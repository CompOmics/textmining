#!/usr/bin/env python3
"""Run the SDRF semantic benchmark on staircase arms."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


STAIRCASE = Path(__file__).resolve().parents[1]
HAMLET = STAIRCASE.parents[2]
AGENTIC = HAMLET / "agentic-metadata"
FRAMEWORK = HAMLET / "textmining/framework"
BENCHMARK_RUNNER = HAMLET / "textmining/agentic-benchmark/benchmark_data/run_sdrf_benchmark.py"


def load_benchmark_runner():
    # Resolve the matcher and mappings from agentic-metadata before the
    # benchmark runner manipulates sys.path.
    sys.path.insert(0, str(AGENTIC))
    __import__("benchmark.semantic_matcher")
    __import__("core.field_mappings")
    spec = importlib.util.spec_from_file_location("hamlet_sdrf_benchmark", BENCHMARK_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCHMARK_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = FRAMEWORK
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--arm", action="append", choices=[f"S{i}" for i in range(6)])
    parser.add_argument(
        "--append", action="store_true",
        help="Preserve unrelated model/arm records already present in output/index.json.",
    )
    args = parser.parse_args()
    arms = args.arm or [f"S{i}" for i in range(6)]
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(FRAMEWORK)
    benchmark = load_benchmark_runner()
    index_path = args.output / "index.json"
    index = {
        "protocol": str(BENCHMARK_RUNNER),
        "matcher": str(AGENTIC / "benchmark/semantic_matcher.py"),
        "semantic_threshold": 0.70,
        "final_metric_threshold": 0.50,
        "filter_extractable": True,
        "runs": [],
    }
    if args.append and index_path.exists():
        existing = json.loads(index_path.read_text())
        for key in ("semantic_threshold", "final_metric_threshold", "filter_extractable"):
            if existing.get(key) != index[key]:
                raise RuntimeError(f"cannot append: {key} differs from existing index")
        replacing = {(model, arm) for model in args.model for arm in arms}
        index["runs"] = [
            run for run in existing.get("runs", [])
            if (run.get("model"), run.get("arm")) not in replacing
        ]
    for model in args.model:
        for arm in arms:
            extraction_dir = args.inputs / model / arm
            reports_dir = args.output / model / arm
            reports_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{'#' * 78}\nSDRF BENCHMARK: {model} {arm}\n{'#' * 78}", flush=True)
            metrics, results, field_stats = benchmark.step_compare(
                args.gold_dir.resolve(), extraction_dir.resolve(), reports_dir.resolve(),
                filter_extractable=True,
            )
            benchmark.step_generate_plots(
                metrics, results, field_stats, reports_dir.resolve(), model_name=f"{model} {arm}"
            )
            summary_path = reports_dir / "sdrf_benchmark_summary.json"
            summary = json.loads(summary_path.read_text())
            index["runs"].append({
                "model": model, "arm": arm, "report": str(summary_path.resolve()),
                "overall": summary.get("overall", {}),
            })
            index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
