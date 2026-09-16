"""Figure 2, cumulative ablation: F1 and Lin IC per arm (overall and per
agent), output volume (non-unknown fields per dataset) and three-run
unanimity.

Inputs (all from this directory):
  manuscript_figure2/benchmark_metrics.csv           rescore_replicates.py: F1 mean/SD over replicates
  manuscript_figure2/scored_pairs_all_replicates.csv  rescore_replicates.py: every scored pair
  benchmark_annotations/                              the extractions themselves (3 replicates)

Lin IC: for every pair whose field has an ontology (organ, cell_type,
disease, cell_line, species | instrument, cleavage_agent, label), both values
are resolved with the framework normaliser and scored with Lin's measure on
the is_a (+ part_of) DAG. A missing or unresolvable LLM value scores 0; pairs
whose golden does not resolve are left out of the denominator. The
experimental-design agent has no ontology-backed field, so it has no Lin.

    python final_analyses/SapBERTStaircase/scripts/build_figure2.py

Writes manuscript_figure2/{lin_by_cell.csv, consistency_by_cell.csv, figure2.png, figure2_data.csv}.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
FRAMEWORK = REPO / "framework"
FIG = ROOT / "manuscript_figure2"
ANNOTATIONS = ROOT / "benchmark_annotations"
sys.path.insert(0, str(FRAMEWORK))
sys.path.insert(0, str(REPO / "agentic-benchmark" / "benchmark_data"))
sys.path.insert(0, str(REPO / "final_analyses"))
sys.path.insert(0, str(REPO))
logging.disable(logging.WARNING)

ARMS = [f"S{i}" for i in range(6)]
MODELS = {"qwen3_8_27b": "Qwen3.8-27B", "gemma4_31b": "Gemma 4 31B", "glm4_7": "GLM-4.7"}
AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]
AGENT_LABEL = {"BiologicalAgent": "Biological", "TechnicalAgent": "Technical", "ExperimentalDesignAgent": "Experimental design"}
# cell types and cell lines are looked up in both CL and CLO (a "cell type"
# column often holds a cell line and vice versa); Lin is taken in whichever
# ontology resolves both sides, CL first for cell_type, CLO first for cell_line.
ONTOLOGY_FIELDS = {"organ": ["uberon"], "cell_type": ["cl", "clo"], "disease": ["mondo"], "cell_line": ["clo", "cl"],
                   "species": ["species"], "instrument": ["psi-ms"], "cleavage_agent": ["psi-ms"], "label": ["psi-ms"]}
FLAT = {"species"}
EXCLUDED_FIELDS = {"material_type", "ptm", "technology_type", "developmental_stage", "ethnicity", "mass_analyzer"}


# ----------------------------------------------------------------------------
def lin_per_pair(pairs: pd.DataFrame) -> pd.DataFrame:
    from icgraph import IcGraph
    from normalization.normalizer import TermNormalizer
    from normalization.config import NormalizationConfig
    from normalization.download import create_species_ontology
    from benchmark.semantic_matcher import HierarchicalMatcher
    from run_sdrf_benchmark import _normalize_label_value

    onto_dir = FRAMEWORK / "ontologies"
    create_species_ontology(onto_dir)
    config = NormalizationConfig()
    config.ontology_dir, config.cache_dir = str(onto_dir), str(FRAMEWORK / "ontology_cache")
    normalizer = TermNormalizer(config)
    for ont, fname in (("cl", "cl.obo"), ("uberon", "uberon.obo"), ("mondo", "mondo.obo"), ("psi-ms", "psi-ms.obo"),
                       ("clo", "clo.owl"), ("species", "species.obo")):
        normalizer.load_ontology(ont, str(onto_dir / fname))
    graphs = {"uberon": IcGraph(onto_dir / "uberon.obo"), "cl": IcGraph(onto_dir / "cl.obo"),
              "mondo": IcGraph(onto_dir / "mondo.obo", relations=("is_a",)), "psi-ms": IcGraph(onto_dir / "psi-ms.obo"),
              "clo": IcGraph(onto_dir / "clo.owl"), "species": IcGraph(onto_dir / "species.obo", relations=("is_a",))}
    matcher = HierarchicalMatcher()

    @lru_cache(maxsize=None)
    def resolve(term, ont):
        r = normalizer.normalize(str(term), ontology_id=ont)
        return r.ontology_id if r.is_normalized else None

    def items(v):
        return [x for x in matcher._split_multi_value(matcher._normalize_llm_separators(str(v)))
                if x and not matcher.is_null_value(x)]

    def in_graph(g, tid, ont):
        return bool(tid) and (ont in FLAT or tid in g.terms or tid in g.parents)

    def graded(llm, golden, field):
        """mean over golden items of the best Lin over LLM items and over the
        field's ontologies; None if no golden item resolves anywhere
        (excluded); 0 if the LLM value is missing or cannot be resolved."""
        golden = _normalize_label_value(golden) if field == "label" else golden
        g_items = items(golden)
        l_items = items(llm) if isinstance(llm, str) else []
        out = []
        for gi in g_items:
            resolved_anywhere = False
            best = 0.0
            for ont in ONTOLOGY_FIELDS[field]:
                g = graphs[ont]
                gid = resolve(gi, ont)
                if not in_graph(g, gid, ont):
                    continue
                resolved_anywhere = True
                for li in l_items:
                    lid = resolve(li, ont)
                    if in_graph(g, lid, ont):
                        best = max(best, float(gid == lid) if ont in FLAT else g.lin(lid, gid))
            if resolved_anywhere:
                out.append(best)
        return float(np.mean(out)) if out else None

    sub = pairs[pairs["field"].isin(ONTOLOGY_FIELDS)].copy()
    sub["lin"] = [graded(l, g, f) for l, g, f in zip(sub["llm"], sub["golden"], sub["field"])]
    return sub


# ----------------------------------------------------------------------------
def consistency() -> pd.DataFrame:
    """Non-unknown fields per dataset (mean over PXDs and replicates) and the
    share of (PXD, field) slots on which all three replicates agree."""
    from run_sdrf_benchmark import find_llm_outputs
    from benchmark.semantic_matcher import HierarchicalMatcher
    m = HierarchicalMatcher()

    def value_of(field_data):
        v = m.extract_value(field_data)
        return None if v is None or m.is_null_value(v) else m.normalize_text(v)

    reps = {1: ANNOTATIONS, 2: ANNOTATIONS / "replicate_2", 3: ANNOTATIONS / "replicate_3"}
    slots = defaultdict(dict)          # (model, arm, pxd, field) -> {rep: value}
    volume = []
    for rep, root in reps.items():
        for model in MODELS:
            for arm in ARMS:
                outputs = find_llm_outputs(root / model / arm)
                per_pxd = defaultdict(int)
                for agent, files in outputs.items():
                    for pxd, path in files.items():
                        data = json.load(open(path))
                        for field, fd in data.items():
                            if field.startswith("_") or field in EXCLUDED_FIELDS:
                                continue
                            v = value_of(fd)
                            slots[(model, arm, pxd, f"{agent}:{field}")][rep] = v
                            if v is not None:
                                per_pxd[pxd] += 1
                for pxd, n in per_pxd.items():
                    volume.append({"model_dir": model, "arm": arm, "replicate": rep, "pxd": pxd, "non_unknown_fields": n})
    vol = pd.DataFrame(volume)
    rows = []
    for (model, arm), _ in vol.groupby(["model_dir", "arm"]):
        keys = [k for k in slots if k[0] == model and k[1] == arm and len(slots[k]) == 3]
        vals = [tuple(slots[k][r] for r in (1, 2, 3)) for k in keys]
        unanimous = [len(set(v)) == 1 for v in vals]
        informative = [v for v in vals if any(x is not None for x in v)]
        unan_inf = [len(set(v)) == 1 for v in informative]
        v = vol[(vol.model_dir == model) & (vol.arm == arm)]
        per_rep = v.groupby("replicate").non_unknown_fields.mean()
        rows.append({"model": MODELS[model], "model_dir": model, "arm": arm,
                     "non_unknown_fields_mean": v.non_unknown_fields.mean(),
                     "non_unknown_fields_sd_across_replicates": per_rep.std(ddof=1),
                     "slots": len(vals), "unanimity_all_slots": float(np.mean(unanimous)) if vals else np.nan,
                     "unanimity_informative_slots": float(np.mean(unan_inf)) if unan_inf else np.nan,
                     "n_informative_slots": len(informative)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main():
    metrics = pd.read_csv(FIG / "benchmark_metrics.csv")
    pairs = pd.read_csv(FIG / "scored_pairs_all_replicates.csv")

    print("Lin IC ...", flush=True)
    graded = lin_per_pair(pairs)
    graded.to_csv(FIG / "lin_pairs_all_replicates.csv", index=False)
    scored = graded[graded["lin"].notna()]
    per_rep = (scored.groupby(["model", "model_dir", "arm", "replicate"])
               .apply(lambda d: pd.Series({"lin_overall": d["lin"].mean(),
                                           **{f"lin_{a}": d[d.agent == a]["lin"].mean() for a in AGENTS if (d.agent == a).any()}}))
               .reset_index())
    lin_cols = [c for c in per_rep.columns if c.startswith("lin_")]
    lin = per_rep.groupby(["model", "model_dir", "arm"])[lin_cols].agg(["mean", "std"])
    lin.columns = [f"{a}_{b}" for a, b in lin.columns]
    lin = lin.reset_index()
    lin["n_pairs_per_replicate"] = scored.groupby(["model", "model_dir", "arm", "replicate"]).size().groupby(level=[0, 1, 2]).mean().values
    lin.to_csv(FIG / "lin_by_cell.csv", index=False)

    print("consistency ...", flush=True)
    cons = consistency()
    cons.to_csv(FIG / "consistency_by_cell.csv", index=False)

    data = metrics.merge(lin, on=["model", "model_dir", "arm"]).merge(cons, on=["model", "model_dir", "arm"])
    data["arm"] = pd.Categorical(data["arm"], ARMS, ordered=True)
    data = data.sort_values(["model", "arm"])
    data.to_csv(FIG / "figure2_data.csv", index=False)
    draw(data)


def draw(data: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import COLORS, clean_axes

    palette = {"Qwen3.8-27B": COLORS["blue"], "Gemma 4 31B": COLORS["green"], "GLM-4.7": COLORS["orange"]}
    x = np.arange(len(ARMS))

    def line(ax, col_mean, col_sd=None, ylim=(0, 1), label_models=False):
        for model, sub in data.groupby("model", sort=False):
            sub = sub.set_index("arm").reindex(ARMS)
            y = sub[col_mean].to_numpy(dtype=float)
            err = sub[col_sd].to_numpy(dtype=float) if col_sd and col_sd in sub else None
            ax.errorbar(x, y, yerr=err, fmt="-o", color=palette[model], ms=4, lw=1.6, capsize=2,
                        label=model if label_models else None)
        ax.set_xticks(x); ax.set_xticklabels(ARMS, fontsize=8)
        if ylim: ax.set_ylim(*ylim)
        clean_axes(ax)

    # main figure: a overall F1, b overall Lin IC, c F1 per agent
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0:3]); line(ax, "overall_f1_mean", "overall_f1_std", label_models=True)
    ax.set_title("a  Overall F1 (mean ± SD, 3 inference runs)", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("macro-F1 over agents"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = fig.add_subplot(gs[0, 3:6]); line(ax, "lin_overall_mean", "lin_overall_std")
    ax.set_title("b  Overall Lin IC (ontology-backed fields)", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylabel("mean Lin similarity")
    for i, agent in enumerate(AGENTS):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2]); line(ax, f"{agent}_f1_mean", f"{agent}_f1_std")
        ax.set_title(("c  " if i == 0 else "") + f"F1 · {AGENT_LABEL[agent]}", loc="left", fontsize=10, fontweight="bold")
        if i == 0: ax.set_ylabel("weighted F1")
    fig.savefig(FIG / "figure2_performance.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # supplementary: Lin IC per agent
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, agent in enumerate(AGENTS):
        ax = axes[i]; col = f"lin_{agent}_mean"
        if col in data and data[col].notna().any():
            line(ax, col, f"lin_{agent}_std", label_models=(i == 0))
        else:
            ax.set_xticks(x); ax.set_xticklabels(ARMS, fontsize=8); ax.set_ylim(0, 1); clean_axes(ax)
            ax.text(0.5, 0.5, "no ontology-backed field\n(counts and free text)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title(f"Lin IC · {AGENT_LABEL[agent]}", loc="left", fontsize=10, fontweight="bold")
        if i == 0: ax.set_ylabel("mean Lin similarity"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "figureS_lin_by_agent.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    line(axes[0], "non_unknown_fields_mean", "non_unknown_fields_sd_across_replicates", ylim=None, label_models=True)
    axes[0].set_title("e  Output volume: non-unknown fields per dataset", loc="left", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("non-unknown fields per dataset"); axes[0].legend(frameon=False, fontsize=8)
    line(axes[1], "unanimity_all_slots", None)
    axes[1].set_title("f  Three-run unanimity", loc="left", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("share of (dataset, field) slots identical in all 3 runs")
    fig.tight_layout(); fig.savefig(FIG / "figure2_consistency.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {FIG / 'figure2_performance.png'}, figure2_consistency.png, figureS_lin_by_agent.png")


if __name__ == "__main__":
    main()
