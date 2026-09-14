#!/usr/bin/env python3
"""Compare SciBERT and SapBERT on the frozen standalone 30-paper extraction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
HAMLET = EXPERIMENT.parents[2]
AGENTIC = HAMLET / "agentic-metadata"
FRAMEWORK = HAMLET / "textmining/framework"
BENCHMARK_RUNNER = HAMLET / "textmining/agentic-benchmark/benchmark_data/run_sdrf_benchmark.py"
GOLD = HAMLET / "textmining/agentic-benchmark/benchmark_data/SDRFS_github"
INPUTS = EXPERIMENT / "inputs/qwen3_8_27b"

ENCODERS = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}


def load_runner(embedding_model: str, tag: str):
    sys.path.insert(0, str(AGENTIC))
    import benchmark.semantic_matcher as matcher_module

    original = matcher_module.SemanticMatcher

    class SelectedSemanticMatcher(original):
        def __init__(self, model_name: str = embedding_model, threshold: float = 0.75,
                     device: str | None = None):
            super().__init__(model_name=model_name, threshold=threshold, device=device)

    matcher_module.SemanticMatcher = SelectedSemanticMatcher
    spec = importlib.util.spec_from_file_location(f"standalone30_benchmark_{tag}", BENCHMARK_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCHMARK_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = FRAMEWORK
    return module, matcher_module, original


def run_one(tag: str, embedding_model: str, output_root: Path) -> dict:
    benchmark, matcher_module, original = load_runner(embedding_model, tag)
    report_dir = output_root / tag
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics, results, field_stats = benchmark.step_compare(
            GOLD.resolve(), INPUTS.resolve(), report_dir.resolve(), filter_extractable=True
        )
        benchmark.step_generate_plots(
            metrics,
            results,
            field_stats,
            report_dir.resolve(),
            model_name=f"Qwen 3.8 27B · {tag.capitalize()} semantic matcher",
        )
    finally:
        matcher_module.SemanticMatcher = original

    summary_path = report_dir / "sdrf_benchmark_summary.json"
    summary = json.loads(summary_path.read_text())
    return {
        "encoder": tag,
        "embedding_model": embedding_model,
        "report": str(summary_path.resolve()),
        "overall": summary.get("overall", {}),
        "agents": summary.get("metrics", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "results")
    parser.add_argument("--encoder", action="append", choices=sorted(ENCODERS))
    args = parser.parse_args()
    selected = args.encoder or ["scibert", "sapbert"]
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(FRAMEWORK)

    runs = []
    for tag in selected:
        print(f"\n{'#' * 78}\nSEMANTIC ENCODER: {tag} ({ENCODERS[tag]})\n{'#' * 78}", flush=True)
        runs.append(run_one(tag, ENCODERS[tag], args.output))

    index = {
        "experiment": "Standalone Qwen 3.8 27B test-30 semantic-encoder comparison",
        "annotations": str(INPUTS.resolve()),
        "gold": str(GOLD.resolve()),
        "benchmark_runner": str(BENCHMARK_RUNNER.resolve()),
        "matcher": str((AGENTIC / "benchmark/semantic_matcher.py").resolve()),
        "semantic_threshold": 0.70,
        "final_metric_threshold": 0.50,
        "filter_extractable": True,
        "runs": runs,
    }
    index_path = args.output / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()

