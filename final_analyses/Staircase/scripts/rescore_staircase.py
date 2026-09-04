"""Re-score the staircase against the SDRF goldens with the current matcher.

No inference is re-run. `benchmark_annotations/` holds every model output for
S0-S5 across three replicates, so changing the matcher only means scoring the
same extractions again.

Written for one question: tier 5 of the matcher moved from SciBERT to SapBERT
on 2026-09-04, and the reported staircase F1 was computed with SciBERT. How much
does it move?

    python final_analyses/Staircase/scripts/rescore_staircase.py
    python final_analyses/Staircase/scripts/rescore_staircase.py --arms S0 S5

Writes rescored_summary.csv next to the existing benchmark_results/summary.csv.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STAIRCASE = HERE.parent
REPO = STAIRCASE.parents[1]

ANNOTATIONS = STAIRCASE / "benchmark_annotations"
GOLDEN_DIR = REPO / "agentic-benchmark" / "benchmark_data" / "SDRFS_github"
EXISTING_SUMMARY = STAIRCASE / "benchmark_results" / "summary.csv"
OUT_CSV = STAIRCASE / "benchmark_results" / "rescored_summary.csv"
METRICS_CSV = STAIRCASE / "figures" / "figure3" / "benchmark_metrics.csv"
SUMMARY_CSV = STAIRCASE / "benchmark_results" / "summary.csv"

# The benchmark script imports `benchmark.semantic_matcher` and
# `core.field_mappings` as bare top-level packages, which only resolve with
# framework/ on the path. Its own sys.path setup assumes the pre-split layout.
sys.path.insert(0, str(REPO / "framework"))
sys.path.insert(0, str(REPO / "agentic-benchmark" / "benchmark_data"))

MODEL_LABEL = {
    "qwen3_8_27b": "Qwen3.8-27B",
    "gemma4_31b": "Gemma 4 31B",
    "glm4_7": "GLM-4.7",
}
ARMS = ["S0", "S1", "S2", "S3", "S4", "S5"]
AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]


def replicate_dirs():
    """{replicate_number: directory holding the per-model trees}.

    Replicate 1 sits at the top level, 2 and 3 in their own subdirectories.
    """
    found = {1: ANNOTATIONS}
    for n in (2, 3):
        candidate = ANNOTATIONS / f"replicate_{n}"
        if candidate.exists():
            found[n] = candidate
    return found


def write_benchmark_metrics(scores):
    """figures/figure3/benchmark_metrics.csv, the table Figure 2 and 3 read.

    Same column layout as the file it replaces: overall first, then each agent,
    each as mean and sample SD over the replicates.
    """
    pieces = []
    for (model_dir, arm), group in scores.groupby(["model_dir", "arm"]):
        row = {"model": model_dir, "arm": arm, "n": len(group)}
        for scope in ["overall"] + AGENTS:
            for metric in ("precision", "recall", "f1"):
                values = group[f"{scope}_{metric}"].dropna()
                row[f"{scope}_{metric}_mean"] = values.mean()
                row[f"{scope}_{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else 0.0
        pieces.append(row)

    table = pd.DataFrame(pieces)
    table["arm"] = pd.Categorical(table["arm"], ARMS, ordered=True)
    table = table.sort_values(["model", "arm"])
    table.to_csv(METRICS_CSV, index=False)
    print(f"wrote {METRICS_CSV}")


def write_summary(scores):
    """benchmark_results/summary.csv, with Student-t 95% intervals."""
    from scipy import stats as sps

    pieces = []
    for (model_dir, arm), group in scores.groupby(["model_dir", "arm"]):
        row = {"model": model_dir, "arm": arm, "n": len(group)}
        for metric in ("precision", "recall", "f1"):
            values = group[f"overall_{metric}"].dropna().to_numpy()
            mean = values.mean()
            sd = values.std(ddof=1) if len(values) > 1 else 0.0
            if len(values) > 1 and sd > 0:
                half = sps.t.ppf(0.975, len(values) - 1) * sd / len(values) ** 0.5
            else:
                half = 0.0
            row |= {f"{metric}_mean": mean, f"{metric}_sd": sd,
                    f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half}
        row["f1_values"] = ";".join(f"{v:.4f}" for v in
                                    group.sort_values("replicate")["overall_f1"])
        pieces.append(row)

    table = pd.DataFrame(pieces)
    table["arm"] = pd.Categorical(table["arm"], ARMS, ordered=True)
    table = table.sort_values(["model", "arm"])
    table.to_csv(SUMMARY_CSV, index=False)
    print(f"wrote {SUMMARY_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_LABEL))
    parser.add_argument("--replicates", nargs="+", type=int, default=None,
                        help="default: every replicate present")
    parser.add_argument("--write-tables", action="store_true",
                        help="overwrite figures/figure3/benchmark_metrics.csv and "
                             "benchmark_results/summary.csv, the tables the figures read")
    args = parser.parse_args()

    if not GOLDEN_DIR.exists():
        sys.exit(f"golden set not found: {GOLDEN_DIR}")

    # Imported here, after sys.path is set, and noisy on import.
    from run_sdrf_benchmark import step_compare
    from benchmark.semantic_matcher import SemanticMatcher

    probe = SemanticMatcher()
    print(f"matcher tier 5: {probe.model_name} @ threshold {probe.threshold}")
    print(f"goldens:        {GOLDEN_DIR}")

    replicates = replicate_dirs()
    if args.replicates:
        replicates = {n: d for n, d in replicates.items() if n in args.replicates}
    print(f"replicates:     {sorted(replicates)}\n")

    rows = []
    for replicate, replicate_root in sorted(replicates.items()):
        for model in args.models:
            for arm in args.arms:
                extraction_dir = replicate_root / model / arm
                if not extraction_dir.exists():
                    print(f"  missing, skipped: {extraction_dir}")
                    continue

                started = time.time()
                # step_compare writes reports; it does not need them kept.
                with tempfile.TemporaryDirectory() as reports:
                    with open(os.devnull, "w") as quiet:
                        stdout = sys.stdout
                        sys.stdout = quiet
                        try:
                            agent_metrics, _, _ = step_compare(
                                GOLDEN_DIR, extraction_dir, Path(reports),
                                filter_extractable=True)
                        finally:
                            sys.stdout = stdout

                # The reported headline is the macro-average over the three
                # agents, matching benchmark_results/README.md.
                row = {"model": MODEL_LABEL[model], "model_dir": model,
                       "arm": arm, "replicate": replicate}
                for metric in ("precision", "recall", "f1"):
                    per_agent = [agent_metrics.get(a, {}).get(f"weighted_{metric}")
                                 for a in AGENTS]
                    for agent, value in zip(AGENTS, per_agent):
                        row[f"{agent}_{metric}"] = value
                    scored = [v for v in per_agent if v is not None]
                    row[f"overall_{metric}"] = (sum(scored) / len(scored)
                                                if scored else float("nan"))
                row["n_agents_scored"] = sum(
                    1 for a in AGENTS if agent_metrics.get(a, {}).get("weighted_f1") is not None)
                rows.append(row)
                print(f"  {MODEL_LABEL[model]:14s} {arm}  rep{replicate}  "
                      f"F1 {row['overall_f1']:.3f}   ({time.time() - started:.0f}s)")

    if not rows:
        sys.exit("nothing scored")

    scores = pd.DataFrame(rows)
    scores.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    # Read the previous summary before the writers overwrite it, otherwise the
    # delta column silently compares the new numbers against themselves.
    baseline = None
    if EXISTING_SUMMARY.exists():
        baseline = pd.read_csv(EXISTING_SUMMARY)[["model", "arm", "f1_mean", "f1_sd"]]

    if args.write_tables:
        write_benchmark_metrics(scores)
        write_summary(scores)

    # ---- mean and SD across replicates, the form the manuscript quotes ------
    summary = (scores.groupby(["model", "arm"])["overall_f1"]
               .agg(rescored_mean="mean", rescored_sd=lambda s: s.std(ddof=1),
                    n="size").reset_index())
    summary["arm"] = pd.Categorical(summary["arm"], ARMS, ordered=True)
    summary = summary.sort_values(["model", "arm"])

    # summary.csv keys on the directory name and calls the column f1_mean.
    summary["model_dir"] = summary["model"].map({v: k for k, v in MODEL_LABEL.items()})
    if baseline is not None:
        previous = baseline.rename(columns={"model": "model_dir",
                                            "f1_mean": "baseline_mean",
                                            "f1_sd": "baseline_sd"})
        summary = summary.merge(previous, on=["model_dir", "arm"], how="left")
        summary["delta"] = summary["rescored_mean"] - summary["baseline_mean"]
    summary = summary.drop(columns=["model_dir"])

    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    summary.to_csv(OUT_CSV.with_name("rescored_vs_reported.csv"), index=False)
    print(f"\nwrote {OUT_CSV.with_name('rescored_vs_reported.csv')}")


if __name__ == "__main__":
    main()
