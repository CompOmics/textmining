"""Main entry point for the run-level metadata extraction pipeline.

Usage:
    python -m run_metadata --config run_metadata/config.yaml
    python -m run_metadata --config run_metadata/config.yaml  (runs all enabled pipelines)
"""

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

for _name in ("httpx", "httpcore", "urllib3", "transformers", "huggingface_hub",
               "faiss", "faiss.loader"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _resolve(config_path: Path, value: str) -> Path:
    """Resolve a path relative to the config file's directory."""
    p = Path(value)
    return p if p.is_absolute() else (config_path.parent / p).resolve()


def run_pipeline(config_path: Path):
    """Run all enabled run-level metadata pipelines."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    quant_path = _resolve(config_path, config["input"]["quant_parquet"])
    output_dir = _resolve(config_path, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ontology dir (shared across pipelines that need it)
    ontology_dir = None
    sdrf_cfg = config.get("sdrf", {})
    if sdrf_cfg.get("normalization", {}).get("ontology_dir"):
        ontology_dir = _resolve(config_path, sdrf_cfg["normalization"]["ontology_dir"])

    # --- Run Name pipeline ---
    name_cfg = config.get("run_name", {})
    name_output = output_dir / name_cfg.get("output", "run_meta_name.tsv")
    if name_cfg.get("enabled"):
        log.info("=== Run Name Extraction ===")
        from .pipeline_name import run as run_name
        run_name(
            quant_path=quant_path,
            output_path=name_output,
            ontology_dir=ontology_dir,
        )

    # --- SDRF pipeline ---
    sdrf_output = output_dir / sdrf_cfg.get("output", "run_metadata_sdrf.tsv")
    if sdrf_cfg.get("enabled"):
        log.info("=== SDRF Extraction ===")
        from .pipeline_sdrf import run as run_sdrf
        sdrf_cache = _resolve(config_path, sdrf_cfg["sdrf_cache_dir"])
        mapping_path = _resolve(config_path, sdrf_cfg["sample_name_mapping"]) if "sample_name_mapping" in sdrf_cfg else None
        # Resolve normalization paths
        norm_cfg = sdrf_cfg.get("normalization", {})
        if norm_cfg.get("enabled") and "ontology_dir" in norm_cfg:
            norm_cfg = dict(norm_cfg)  # copy to avoid modifying config
            norm_cfg["ontology_dir"] = str(_resolve(config_path, norm_cfg["ontology_dir"]))
            norm_cfg["cache_dir"] = str(_resolve(config_path, norm_cfg["cache_dir"]))
        else:
            norm_cfg = None

        run_sdrf(
            quant_path=quant_path,
            output_path=sdrf_output,
            sdrf_cache_dir=sdrf_cache,
            sample_name_mapping_path=mapping_path,
            normalization_cfg=norm_cfg,
        )

    # --- MLMarker pipeline ---
    mlm_cfg = config.get("mlmarker", {})
    mlm_output = output_dir / mlm_cfg.get("output", "run_meta_mlmarker.tsv")
    if mlm_cfg.get("enabled"):
        log.info("=== MLMarker Tissue Prediction ===")
        from .pipeline_mlmarker import run as run_mlmarker
        run_mlmarker(
            quant_path=quant_path,
            output_path=mlm_output,
            confidence_threshold=mlm_cfg.get("confidence_threshold", 0.3),
        )

    # --- Combine pipeline ---
    combine_cfg = config.get("combine", {})
    if combine_cfg.get("enabled"):
        log.info("=== Combine Metadata ===")
        from .pipeline_combine import run as run_combine
        agent_path = _resolve(config_path, combine_cfg["agent_metadata"])
        mapping_path = _resolve(config_path, sdrf_cfg.get("sample_name_mapping", "")) if sdrf_cfg.get("sample_name_mapping") else Path("nonexistent")
        run_combine(
            quant_path=quant_path,
            sdrf_path=sdrf_output,
            name_path=name_output,
            mlm_path=mlm_output,
            agent_path=agent_path,
            mapping_path=mapping_path,
            output_path=output_dir / combine_cfg.get("output", "run_metadata_combined.tsv"),
        )

    log.info("=== Done ===")


def main():
    parser = argparse.ArgumentParser(description="Run-level metadata extraction pipeline")
    parser.add_argument("--config", default="run_metadata/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    run_pipeline(config_path=Path(args.config))


if __name__ == "__main__":
    main()
