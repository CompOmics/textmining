"""Main entry point for running the agentic metadata extraction pipelines"""

import argparse
import logging
from pathlib import Path

import yaml

from .core import resolve_manuscripts, run_extraction, _init_normalizer
from .export_tsv import export_tsv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress noisy third-party loggers
for _name in ("httpx", "httpcore", "urllib3", "transformers", "huggingface_hub",
               "faiss", "faiss.loader"):
    logging.getLogger(_name).setLevel(logging.WARNING)

BENCHMARK_DIR = Path(__file__).parent / "benchmark"


def _resolve_config_path(config_path: Path, value: str) -> Path:
    """Resolve a path from config relative to the config file's directory."""
    p = Path(value)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


# Pipeline for doing actual extraction
def run_pipeline(config_path: Path):
    """Run the full metadata extraction pipeline."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    ollama_cfg = config["ollama"]
    data_cfg = config["data"]

    output_dir = _resolve_config_path(config_path, data_cfg["output_dir"])

    # Resolve relative paths in data config
    if "manuscripts_dir" in data_cfg and data_cfg["manuscripts_dir"]:
        data_cfg["manuscripts_dir"] = str(_resolve_config_path(config_path, data_cfg["manuscripts_dir"]))
    fetch_cfg = data_cfg.get("fetch_manuscripts", {})
    if "pxd_source" in fetch_cfg:
        fetch_cfg["pxd_source"] = str(_resolve_config_path(config_path, fetch_cfg["pxd_source"]))
    if "pxd_dir" in fetch_cfg:
        fetch_cfg["pxd_dir"] = str(_resolve_config_path(config_path, fetch_cfg["pxd_dir"]))

    manuscripts_dir = resolve_manuscripts(data_cfg, default_dir=output_dir / "manuscripts")

    norm_cfg = config.get("normalization", {})
    if "ontology_dir" in norm_cfg:
        norm_cfg["ontology_dir"] = str(_resolve_config_path(config_path, norm_cfg["ontology_dir"]))
    if "cache_dir" in norm_cfg:
        norm_cfg["cache_dir"] = str(_resolve_config_path(config_path, norm_cfg["cache_dir"]))

    extracted_dir = output_dir / "extracted_metadata"
    result = run_extraction(
        manuscripts_dir=manuscripts_dir,
        output_dir=extracted_dir,
        ollama_cfg=ollama_cfg,
        normalization_cfg=norm_cfg,
    )

    # Auto-export agent_metadata.tsv
    tsv_output = output_dir / "agent_metadata.tsv"
    export_tsv(extracted_dir, tsv_output)
    log.info("Exported agent metadata to %s", tsv_output)

    # Write normalization_map.tsv if tracker is available
    tracker = result.get("tracker")
    if tracker:
        map_output = output_dir / "normalization_map.tsv"
        tracker.write_report(map_output)
        log.info("Normalization map written to %s", map_output)


# Benchmarking pipeline
def benchmark_pipeline(config_path: Path, run_name: str = None):
    """Run extraction on benchmark test set and evaluate against ground truth."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    ollama_cfg = config["ollama"]
    bench_cfg = config.get("benchmark", {})
    norm_cfg = config.get("normalization", {})

    if not run_name:
        run_name = bench_cfg.get("run_name", "default")

    # Sanitize run_name for Windows (colons are not allowed in directory names)
    safe_run_name = run_name.replace(":", "_")

    test_set_dir = BENCHMARK_DIR / "test_set"
    results_dir = BENCHMARK_DIR / "results" / safe_run_name

    # 1. Run extraction on benchmark manuscripts
    run_extraction(
        manuscripts_dir=test_set_dir / "manuscripts",
        output_dir=results_dir,
        ollama_cfg=ollama_cfg,
        normalization_cfg=norm_cfg,
    )

    # 2. Evaluate predictions against ground truth
    from .benchmark.evaluate import evaluate_benchmark, print_results, save_results

    result = evaluate_benchmark(
        labels_dir=test_set_dir / "labels",
        predictions_dir=results_dir,
    )
    print_results(result)
    save_results(result, results_dir / "results_summary.json")


def normalize_pipeline(config_path: Path, input_dir: str):
    """Post-process existing JSON results with ontology normalization."""
    import json
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    norm_cfg = config.get("normalization", {})
    norm_cfg["enabled"] = True  # force enable for this command

    normalizer, cell_line_matcher = _init_normalizer(norm_cfg)
    if normalizer is None:
        log.error("Failed to initialize normalizer")
        return

    from .normalization.normalize_extraction import normalize_extraction

    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("PXD*.json"))
    log.info("Normalizing %d files in %s", len(json_files), input_path)

    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        data = normalize_extraction(data, normalizer, cell_line_matcher, config=config)
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.info("Normalized %s", f.name)

    log.info("Done: %d files normalized", len(json_files))


def setup_ontologies(config_path: Path):
    """Build SapBERT FAISS indices from custom vocabulary files."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    norm_cfg = config.get("normalization", {})
    if "ontology_dir" in norm_cfg:
        norm_cfg["ontology_dir"] = str(_resolve_config_path(config_path, norm_cfg["ontology_dir"]))
    if "cache_dir" in norm_cfg:
        norm_cfg["cache_dir"] = str(_resolve_config_path(config_path, norm_cfg["cache_dir"]))

    from .normalization.config import NormalizationConfig
    from .normalization.normalizer import TermNormalizer

    nc = NormalizationConfig.from_dict(norm_cfg)
    normalizer = TermNormalizer(nc)
    normalizer.load_all_ontologies()
    log.info("Setup complete: %d vocabularies indexed", len(normalizer.indices))


def qc_pipeline(config_path: Path, input_dir: str, output: str = None):
    """Generate normalization_map.tsv from existing extracted JSON results.

    Reads all PXD*.json files, re-runs normalization to collect mapping data,
    and writes a report showing what got mapped to what (and what didn't).
    Does NOT modify the JSON files.
    """
    import json
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    norm_cfg = config.get("normalization", {})
    norm_cfg["enabled"] = True
    if "ontology_dir" in norm_cfg:
        norm_cfg["ontology_dir"] = str(_resolve_config_path(config_path, norm_cfg["ontology_dir"]))
    if "cache_dir" in norm_cfg:
        norm_cfg["cache_dir"] = str(_resolve_config_path(config_path, norm_cfg["cache_dir"]))

    normalizer, cell_line_matcher = _init_normalizer(norm_cfg)
    if normalizer is None:
        log.error("Failed to initialize normalizer")
        return

    from .normalization.normalize_extraction import normalize_extraction
    from .normalization.tracker import NormalizationTracker

    tracker = NormalizationTracker()

    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("PXD*.json"))
    log.info("Running normalization QC on %d files in %s", len(json_files), input_path)

    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        # Re-run normalization with tracker but discard the result (don't overwrite files)
        normalize_extraction(data, normalizer, cell_line_matcher,
                             config=config, tracker=tracker)

    # Write report
    if output:
        map_output = Path(output)
    else:
        map_output = input_path.parent / "normalization_map.tsv"
    tracker.write_report(map_output)
    log.info("Normalization map written to %s (%d records)", map_output, len(tracker.records))


def fetch_pipeline(config_path: Path):
    """Fetch manuscripts only, without running extraction.

    Always runs the fetch regardless of manuscripts_dir setting.
    Skips PXDs that already have a manuscript file (incremental).
    """
    from .fetch_manuscripts import fetch_all_manuscripts

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]

    # Determine output directory
    if data_cfg.get("manuscripts_dir"):
        manuscripts_dir = _resolve_config_path(config_path, data_cfg["manuscripts_dir"])
    else:
        output_dir = _resolve_config_path(config_path, data_cfg["output_dir"])
        manuscripts_dir = output_dir / "manuscripts"

    # Determine PXD list
    fetch_cfg = data_cfg.get("fetch_manuscripts", {})
    if "pxd_source" in fetch_cfg:
        import pandas as pd
        pxd_source = _resolve_config_path(config_path, fetch_cfg["pxd_source"])
        df = pd.read_parquet(pxd_source, columns=["pxd"])
        pxd_ids = df["pxd"].unique().tolist()
        log.info("Read %d unique PXDs from %s", len(pxd_ids), pxd_source)
    elif "pxd_dir" in fetch_cfg:
        pxd_file = _resolve_config_path(config_path, fetch_cfg["pxd_dir"])
        pxd_ids = [l.strip() for l in pxd_file.read_text().splitlines()
                    if l.strip().startswith("PXD")]
        log.info("Read %d PXDs from %s", len(pxd_ids), pxd_file)
    else:
        log.error("No pxd_source or pxd_dir configured in fetch_manuscripts")
        return

    fetch_all_manuscripts(pxd_ids=pxd_ids, output_dir=manuscripts_dir)
    log.info("Manuscripts fetched to %s", manuscripts_dir)


def main():
    parser = argparse.ArgumentParser(description="Agentic metadata extraction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the extraction pipeline")
    run_parser.add_argument("--config", default="agentic_metadata/config.yaml", help="Path to config YAML")

    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark pipeline")
    bench_parser.add_argument("--config", default="agentic_metadata/config.yaml", help="Path to config YAML")

    norm_parser = subparsers.add_parser("normalize", help="Normalize existing JSON results")
    norm_parser.add_argument("input_dir", help="Directory containing PXD*.json files")
    norm_parser.add_argument("--config", default="agentic_metadata/config.yaml", help="Path to config YAML")

    setup_parser = subparsers.add_parser("setup", help="Build FAISS indices from vocabulary files")
    setup_parser.add_argument("--config", default="agentic_metadata/config.yaml", help="Path to config YAML")

    qc_parser = subparsers.add_parser("qc", help="Generate normalization quality report from existing JSONs")
    qc_parser.add_argument("input_dir", help="Directory containing PXD*.json files")
    qc_parser.add_argument("--config", default="agentic_metadata/config.yaml", help="Path to config YAML")
    qc_parser.add_argument("--output", default=None, help="Output path for normalization_map.tsv")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch manuscripts only (no extraction)")
    fetch_parser.add_argument("--config", default="agentic_metadata/config.yaml", help="Path to config YAML")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(config_path=Path(args.config))
    elif args.command == "benchmark":
        benchmark_pipeline(config_path=Path(args.config))
    elif args.command == "normalize":
        normalize_pipeline(config_path=Path(args.config), input_dir=args.input_dir)
    elif args.command == "setup":
        setup_ontologies(config_path=Path(args.config))
    elif args.command == "qc":
        qc_pipeline(config_path=Path(args.config), input_dir=args.input_dir, output=args.output)
    elif args.command == "fetch":
        fetch_pipeline(config_path=Path(args.config))


if __name__ == "__main__":
    main()
