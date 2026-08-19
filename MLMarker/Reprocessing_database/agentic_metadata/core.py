"""Core extraction logic shared between run_pipeline and benchmark_pipeline."""

import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .fetch_manuscripts import fetch_all_manuscripts
from .ollama.ollama_setup import setup_ollama_server
from .ollama.ollama_client import ollama_request
from .prompt_builder.prompt_builder import build_prompt

log = logging.getLogger(__name__)


def resolve_manuscripts(data_cfg: dict, default_dir: Path) -> Path:
    """Resolve manuscript directory from config: use local, fetch, or raise."""
    if data_cfg.get("manuscripts_dir"):
        return Path(data_cfg["manuscripts_dir"])

    if data_cfg.get("fetch_manuscripts", {}).get("enabled"):
        fetch_cfg = data_cfg["fetch_manuscripts"]
        # Support parquet input (pxd_source) or text file (pxd_dir)
        if "pxd_source" in fetch_cfg:
            import pandas as pd
            pxd_source = Path(fetch_cfg["pxd_source"])
            df = pd.read_parquet(pxd_source, columns=["pxd"])
            pxd_ids = df["pxd"].unique().tolist()
            log.info("Read %d unique PXDs from %s", len(pxd_ids), pxd_source)
            fetch_all_manuscripts(pxd_ids=pxd_ids, output_dir=default_dir)
        else:
            fetch_all_manuscripts(
                input_file=Path(fetch_cfg["pxd_dir"]),
                output_dir=default_dir,
            )
        return default_dir

    raise ValueError(
        "No manuscripts available. Either set 'data.manuscripts_dir' to a directory "
        "with existing manuscripts, or set 'data.fetch_manuscripts.enabled' to true in config."
    )


def _init_normalizer(normalization_cfg: dict):
    """Initialize the TermNormalizer and CellLineExactMatcher if normalization is enabled."""
    if not normalization_cfg or not normalization_cfg.get("enabled", False):
        return None, None

    from .normalization.config import NormalizationConfig
    from .normalization.normalizer import TermNormalizer
    from .normalization.normalize_extraction import CellLineExactMatcher

    config = NormalizationConfig.from_dict(normalization_cfg)
    normalizer = TermNormalizer(config)
    normalizer.load_all_ontologies()

    # Cell line exact matcher (bypasses SapBERT)
    cell_line_path = config.get_ontology_path("cell_lines")
    cell_line_matcher = None
    if cell_line_path and cell_line_path.exists():
        cell_line_matcher = CellLineExactMatcher(cell_line_path)
        log.info("Cell line exact matcher: %d entries", len(cell_line_matcher.canonical))

    log.info(
        "Normalization ready: %d ontologies, threshold=%.2f — all embeddings cached in %s \n",
        len(normalizer.indices), config.similarity_threshold, config.cache_dir,
    )
    return normalizer, cell_line_matcher


def run_extraction(
    manuscripts_dir: Path,
    output_dir: Path,
    ollama_cfg: dict,
    normalization_cfg: dict = None,
) -> dict:
    """Run metadata extraction on all manuscripts in a directory.

    Args:
        manuscripts_dir: Directory containing PXD*.txt manuscript files.
        output_dir: Directory to write extracted JSON results.
        ollama_cfg: Ollama config dict with keys: base_url, model, environment, inference.
        normalization_cfg: Optional normalization config dict.

    Returns:
        dict with keys: total_time, total_tokens, n_processed.
    """
    base_url = ollama_cfg["base_url"]
    model = ollama_cfg["model"]
    inference = ollama_cfg.get("inference", {})

    # Setup Ollama server
    setup_ollama_server(
        base_url=base_url,
        model=model,
        env_vars=ollama_cfg.get("environment", {}),
    )

    # Initialize normalizer (once, before the loop)
    normalizer, cell_line_matcher = _init_normalizer(normalization_cfg)

    # Initialize normalization tracker for the normalization_map report
    tracker = None
    if normalizer is not None:
        from .normalization.tracker import NormalizationTracker
        tracker = NormalizationTracker()

    # Process each manuscript
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_files = sorted(manuscripts_dir.glob("PXD*.txt"))
    n_total = len(manuscript_files)
    n_workers = int(ollama_cfg.get("environment", {}).get("OLLAMA_NUM_PARALLEL", 1))

    # Filter out already-processed manuscripts
    to_process = []
    for manuscript_path in manuscript_files:
        pxd_id = manuscript_path.stem
        result_file = output_dir / f"{pxd_id}.json"
        if result_file.exists():
            log.info("  %s: result exists, skipping", pxd_id)
        else:
            to_process.append(manuscript_path)

    if not to_process:
        log.info("All %d PXDs already processed", n_total)
        return {"total_time": 0, "total_tokens": 0, "n_processed": 0}

    log.info("Processing %d/%d PXDs (%d workers)\n", len(to_process), n_total, n_workers)

    # Import normalization here to avoid circular imports
    normalize_fn = None
    if normalizer is not None:
        from .normalization.normalize_extraction import normalize_extraction
        normalize_fn = normalize_extraction

    def _process_one(manuscript_path: Path) -> dict | None:
        """Extract metadata from a single manuscript."""
        pxd_id = manuscript_path.stem
        result_file = output_dir / f"{pxd_id}.json"
        chars = manuscript_path.stat().st_size

        prompt, schema = build_prompt(manuscript_path)
        result = ollama_request(
            prompt=prompt,
            model=model,
            base_url=base_url,
            json_schema=schema,
            num_ctx=inference.get("num_ctx", 32768),
            temperature=inference.get("temperature", 0.0),
            timeout=inference.get("timeout", 600),
            max_tries=inference.get("max_tries", 3),
        )
        if result is None:
            return {"pxd_id": pxd_id, "status": "failed"}

        response_text = result["response"]
        if normalize_fn is not None:
            parsed = json.loads(response_text)
            parsed = normalize_fn(parsed, normalizer, cell_line_matcher,
                                  config={"normalization": normalization_cfg},
                                  tracker=tracker)
            response_text = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"
        result_file.write_text(response_text, encoding="utf-8")

        return {
            "pxd_id": pxd_id,
            "status": "ok",
            "chars": chars,
            "gen_tokens": result["gen_tokens"],
            "tok_s": result["tok_s"],
            "total_duration_s": result["total_duration_s"],
        }

    total_time = 0
    total_tokens = 0
    n_processed = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, mp): mp for mp in to_process}

        for i, future in enumerate(as_completed(futures), 1):
            info = future.result()
            if info is None or info["status"] == "failed":
                pxd_id = info["pxd_id"] if info else "?"
                log.warning("[%d/%d]  %s: SKIPPED — all retries failed", i, len(to_process), pxd_id)
                n_failed += 1
                continue

            log.info(
                "[%d/%d]  %s: %s chars — %d tok, %.0f tok/s, %.1fs",
                i, len(to_process), info["pxd_id"], f"{info['chars']:,}",
                info["gen_tokens"], info["tok_s"], info["total_duration_s"],
            )
            total_time += info["total_duration_s"]
            total_tokens += info["gen_tokens"]
            n_processed += 1

    if n_processed > 0:
        log.info("=" * 50)
        log.info("Extraction complete: %d PXDs (%d failed)", n_processed, n_failed)
        log.info("Total time: %.0fs (%.1f min)", total_time, total_time / 60)
        log.info("Total tokens: %s", f"{total_tokens:,}")
        log.info("Avg time per PXD: %.1fs", total_time / n_processed)

    return {"total_time": total_time, "total_tokens": total_tokens, "n_processed": n_processed, "tracker": tracker}
