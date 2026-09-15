"""Re-score all staircase replicates with the current matcher.

Runs the SDRF benchmark's own step_compare over benchmark_annotations/
(replicate 1 at the top level, replicate_2/ and replicate_3/ below it) for
every model x arm, with CL/UBERON/DOID loaded for tier 4 and the SapBERT
tier at 0.70. The six fields the manuscript does not report are dropped
before metrics are computed (Tine, 2026-09-14). F1 per agent is
calculate_weighted_metrics over the remaining pairs, the overall F1 the
macro-average over the three agents, exactly as the benchmark reports it.

Why this exists: benchmark_runs/ and figures/benchmark_metrics.csv were
produced before framework/benchmark/semantic_matcher.py compared numbers as
numbers (2026-09-14). This regenerates the tables with the current matcher.

    python final_analyses/SapBERTStaircase/scripts/rescore_replicates.py

Writes benchmark_runs_current_matcher/<replicate>/<model>/<arm>/sdrf_benchmark_detailed.csv
and manuscript_figure2/benchmark_metrics.csv (mean and SD over replicates).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
FRAMEWORK = REPO / "framework"
ANNOTATIONS = ROOT / "benchmark_annotations"
GOLD = REPO / "agentic-benchmark" / "benchmark_data" / "SDRFS_github"
OUT_RUNS = ROOT / "benchmark_runs_current_matcher"
OUT_METRICS = ROOT / "manuscript_figure2" / "benchmark_metrics.csv"
OUT_PAIRS = ROOT / "manuscript_figure2" / "scored_pairs_all_replicates.csv"

EXCLUDED_FIELDS = {"material_type", "ptm", "technology_type", "developmental_stage", "ethnicity", "mass_analyzer"}
MODELS = {"qwen3_8_27b": "Qwen3.8-27B", "gemma4_31b": "Gemma 4 31B", "glm4_7": "GLM-4.7"}
ARMS = [f"S{i}" for i in range(6)]
AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]

sys.path.insert(0, str(FRAMEWORK))
sys.path.insert(0, str(REPO / "agentic-benchmark" / "benchmark_data"))


def preload_normalizer():
    """step_compare builds a TermNormalizer and loads the ontologies on every
    call (about a minute each). Build one, and hand it back every time."""
    import normalization.normalizer as nm
    from normalization.config import NormalizationConfig

    config = NormalizationConfig()
    config.ontology_dir = str(FRAMEWORK / "ontologies")
    config.cache_dir = str(FRAMEWORK / "ontology_cache")
    shared = nm.TermNormalizer(config)
    for ont in ("cl", "uberon", "doid"):
        shared.load_ontology(ont, str(FRAMEWORK / "ontologies" / f"{ont}.obo"))
    shared.load_ontology = lambda *a, **k: None   # already loaded

    class Preloaded:
        def __new__(cls, *a, **k):
            return shared

    nm.TermNormalizer = Preloaded
    return shared


def replicate_dirs():
    found = {1: ANNOTATIONS}
    for n in (2, 3):
        d = ANNOTATIONS / f"replicate_{n}"
        if d.exists():
            found[n] = d
    return found


def main():
    os.chdir(FRAMEWORK)
    preload_normalizer()
    import run_sdrf_benchmark as bench
    bench.PROJECT_ROOT = FRAMEWORK
    from benchmark.semantic_matcher import calculate_weighted_metrics

    rows, all_pairs = [], []
    for rep, root in sorted(replicate_dirs().items()):
        for model, label in MODELS.items():
            for arm in ARMS:
                extraction_dir = root / model / arm
                if not extraction_dir.exists():
                    print(f"missing {extraction_dir}")
                    continue
                t0 = time.time()
                with tempfile.TemporaryDirectory() as reports, open(os.devnull, "w") as quiet:
                    stdout, sys.stdout = sys.stdout, quiet
                    try:
                        _, results, _ = bench.step_compare(GOLD, extraction_dir, Path(reports), filter_extractable=True)
                    finally:
                        sys.stdout = stdout
                pairs = pd.DataFrame([{"agent": a, **r} for a, rs in results.items() for r in rs])
                pairs = pairs[~pairs["field"].isin(EXCLUDED_FIELDS)]
                out_dir = OUT_RUNS / f"replicate_{rep}" / model / arm
                out_dir.mkdir(parents=True, exist_ok=True)
                pairs.to_csv(out_dir / "sdrf_benchmark_detailed.csv", index=False)
                pairs = pairs.assign(replicate=rep, model=label, model_dir=model, arm=arm)
                all_pairs.append(pairs)

                row = {"model": label, "model_dir": model, "arm": arm, "replicate": rep, "n_pairs": len(pairs)}
                for agent in AGENTS:
                    g = pairs[pairs["agent"] == agent]
                    m = calculate_weighted_metrics([{"score": s, "llm_has_value": pd.notna(l), "golden_has_value": True}
                                                    for s, l in zip(g["score"], g["llm"])])
                    for k in ("precision", "recall", "f1"):
                        row[f"{agent}_{k}"] = m[f"weighted_{k}"]
                for k in ("precision", "recall", "f1"):
                    row[f"overall_{k}"] = sum(row[f"{a}_{k}"] for a in AGENTS) / 3
                rows.append(row)
                print(f"rep{rep} {label:12s} {arm}  F1 {row['overall_f1']:.3f}  "
                      f"(bio {row['BiologicalAgent_f1']:.2f} tech {row['TechnicalAgent_f1']:.2f} "
                      f"exp {row['ExperimentalDesignAgent_f1']:.2f})  {time.time()-t0:.0f}s", flush=True)

    per_rep = pd.DataFrame(rows)
    pd.concat(all_pairs, ignore_index=True).to_csv(OUT_PAIRS, index=False)
    per_rep.to_csv(OUT_METRICS.with_name("benchmark_metrics_per_replicate.csv"), index=False)
    metric_cols = [c for c in per_rep.columns if c.endswith(("_precision", "_recall", "_f1"))]
    agg = per_rep.groupby(["model", "model_dir", "arm"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["n_replicates"] = per_rep.groupby(["model", "model_dir", "arm"]).size().values
    agg.to_csv(OUT_METRICS, index=False)
    print(f"\nwrote {OUT_METRICS}")
    print(agg.pivot(index="arm", columns="model", values="overall_f1_mean").round(3).to_string())


if __name__ == "__main__":
    main()
