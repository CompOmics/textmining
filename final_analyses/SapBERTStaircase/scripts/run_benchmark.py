#!/usr/bin/env python3
"""Re-score the primary Staircase annotations with SapBERT."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAMLET = ROOT.parents[2]
STAIRCASE = ROOT.parent / "Staircase"
AGENTIC = HAMLET / "agentic-metadata"
FRAMEWORK = HAMLET / "textmining/framework"
BENCHMARK_RUNNER = HAMLET / "textmining/agentic-benchmark/benchmark_data/run_sdrf_benchmark.py"
GOLD = HAMLET / "textmining/agentic-benchmark/benchmark_data/SDRFS_github"
INPUTS = STAIRCASE / "benchmark_inputs"
OUTPUT = ROOT / "benchmark_runs"
EMBEDDING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MODELS = ["qwen3_8_27b", "gemma4_31b", "glm4_7"]
ARMS = [f"S{i}" for i in range(6)]


def load_runner():
    sys.path.insert(0, str(AGENTIC))
    import benchmark.semantic_matcher as matcher_module

    original = matcher_module.SemanticMatcher

    class SapBERTSemanticMatcher(original):
        def __init__(self, model_name: str = EMBEDDING_MODEL, threshold: float = 0.75,
                     device: str | None = None):
            super().__init__(model_name=model_name, threshold=threshold, device=device)

    matcher_module.SemanticMatcher = SapBERTSemanticMatcher
    spec = importlib.util.spec_from_file_location("sapbert_staircase_benchmark", BENCHMARK_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCHMARK_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = FRAMEWORK
    return module


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.chdir(FRAMEWORK)
    benchmark = load_runner()
    index = {
        "experiment": "Single-run SapBERT re-scoring of the primary Staircase annotations",
        "inference_replicate": 1,
        "embedding_model": EMBEDDING_MODEL,
        "annotations": str(INPUTS.resolve()),
        "gold": str(GOLD.resolve()),
        "benchmark_runner": str(BENCHMARK_RUNNER.resolve()),
        "semantic_threshold": 0.70,
        "final_metric_threshold": 0.50,
        "filter_extractable": True,
        "runs": [],
    }
    index_path = OUTPUT / "index.json"

    for model in MODELS:
        for arm in ARMS:
            extraction_dir = INPUTS / model / arm
            reports_dir = OUTPUT / model / arm
            reports_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'#' * 78}\nSAPBERT STAIRCASE: {model} {arm}\n{'#' * 78}", flush=True)
            metrics, results, field_stats = benchmark.step_compare(
                GOLD.resolve(), extraction_dir.resolve(), reports_dir.resolve(),
                filter_extractable=True,
            )
            benchmark.step_generate_plots(
                metrics, results, field_stats, reports_dir.resolve(),
                model_name=f"{model} {arm}",
            )
            summary_path = reports_dir / "sdrf_benchmark_summary.json"
            summary = json.loads(summary_path.read_text())
            index["runs"].append({
                "model": model,
                "arm": arm,
                "report": str(summary_path.resolve()),
                "overall": summary.get("overall", {}),
            })
            index_path.write_text(json.dumps(index, indent=2) + "\n")

    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()

