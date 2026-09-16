"""Score the six-model benchmark outputs (six_model_benchmark/<model>/NormalizedAgent)
against the 30 SDRF / curated goldens with the current five-level matcher and
write six_model_benchmark/reports_test_v2_<model>_nopride/sdrf_benchmark_detailed.csv,
the input of plot_all_models_f1.py. Six manuscript-excluded fields are dropped.

    python final_analyses/SixModelBenchmark/score_six_models.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "SapBERTStaircase" / "scripts"))
import rescore_replicates as rr  # noqa: E402

RUNS = REPO / "six_model_benchmark"
MODELS = ["llama", "gpt", "claude", "gemini", "gemma", "qwen"]


def staged(model: str, root: Path) -> Path:
    """step_compare expects <PXD>/normalized_output/<Agent>/*.json; stage symlinks."""
    for agent in rr.AGENTS:
        for jf in (RUNS / model / "NormalizedAgent" / agent).glob("PXD*.json"):
            d = root / jf.name.split("_")[0] / "normalized_output" / agent
            d.mkdir(parents=True, exist_ok=True); (d / jf.name).symlink_to(jf)
    return root


def main():
    os.chdir(rr.FRAMEWORK); rr.preload_normalizer()
    import run_sdrf_benchmark as bench
    bench.PROJECT_ROOT = rr.FRAMEWORK
    for model in MODELS:
        with tempfile.TemporaryDirectory() as reports, tempfile.TemporaryDirectory() as stage, open(os.devnull, "w") as quiet:
            stdout, sys.stdout = sys.stdout, quiet
            try:
                _, results, _ = bench.step_compare(rr.GOLD, staged(model, Path(stage)), Path(reports), filter_extractable=True)
            finally:
                sys.stdout = stdout
        pairs = pd.DataFrame([{"agent": a, **r} for a, rs in results.items() for r in rs])
        pairs = pairs[~pairs["field"].isin(rr.EXCLUDED_FIELDS)]
        out = RUNS / f"reports_test_v2_{model}_nopride"; out.mkdir(exist_ok=True)
        pairs.to_csv(out / "sdrf_benchmark_detailed.csv", index=False)
        print(f"{model:8s} {len(pairs)} pairs, {pairs.pxd.nunique() if 'pxd' in pairs else ''} datasets", flush=True)


if __name__ == "__main__":
    main()
