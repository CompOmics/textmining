"""HAMLET vs MLMarker, one figure in two halves.

Top: HAMLET proposes ONE tissue -> does MLMarker agree?
  a  concordance tier of every healthy-native run with a single HAMLET
     tissue, split by coverage (>= 0.10 usable / below)
  b  calibration: mean Lin similarity per MLMarker confidence bin, with counts
Bottom: HAMLET proposes SEVERAL tissues -> MLMarker chooses
  c  accuracy against SDRF / run-name truth, MLMarker alone vs restricted to
     HAMLET's list, held-out vs training projects
  d  per multi-tissue project, share of usable runs whose top-1 is one of
     HAMLET's tissues, against chance k/34

Inputs: results/MLMarker_v_HAMLET/healthy_native_human_concordance.tsv (penalised
scoring, biofluids excluded), results/run_meta_mlmarker_all_penalty.tsv (coverage),
results/granularity/A_multi_tissue_samples.csv (scripts/granularity_and_disease.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
RES = ROOT / "results"
OUT = RES / "fig5_concordance_figure"
sys.path.insert(0, str(REPO / "final_analyses"))
from figure_style import (style, panel_title, legend_outside, save_composite, save_panels,  # noqa: E402
                          OUTCOME, NEUTRAL, AGENT, LIN_KIND)

OOV = "HAMLET tissue not among MLMarker's classes"
TIERS = ["Exact Match", "Near-Agreement (Subpart / Secretome)", "Distant Concordance (Organ System)", "True Conflict", OOV]
TIER_LABEL = {"Exact Match": "exact", "Near-Agreement (Subpart / Secretome)": "near\n(subpart)",
              "Distant Concordance (Organ System)": "distant\n(organ system)", "True Conflict": "conflict",
              OOV: "no matching\nMLMarker class"}
TIER_COLOR = {"Exact Match": LIN_KIND["same term"], "Near-Agreement (Subpart / Secretome)": LIN_KIND["related (Lin 0.5-0.99)"],
              "Distant Concordance (Organ System)": LIN_KIND["distant (Lin 0.01-0.5)"], "True Conflict": LIN_KIND["unrelated (Lin <= 0.01)"],
              OOV: NEUTRAL}
import os
EXCLUDE_TRAINING = os.environ.get("EXCLUDE_TRAINING", "0") == "1"   # drop MLMarker training projects from panel c
BIOFLUIDS_AS_OOV = os.environ.get("BIOFLUIDS_AS_OOV", "0") == "1"   # panel a: biofluid-only runs as "no matching class"
SUFFIX = ("_notrain" if EXCLUDE_TRAINING else "") + ("_biofluids" if BIOFLUIDS_AS_OOV else "")
BINS = [(0, 0.10, "< 0.10"), (0.10, 0.15, "0.10-0.15"), (0.15, 0.20, "0.15-0.20"), (0.20, 0.30, "0.20-0.30"), (0.30, 1.01, "≥ 0.30")]


def main():
    OUT.mkdir(exist_ok=True)
    plt = style()
    conc = pd.read_csv(RES / "MLMarker_v_HAMLET/healthy_native_human_concordance.tsv", sep="\t")
    cov = pd.read_csv(RES / "run_meta_mlmarker_all_penalty.tsv", sep="\t", usecols=["pxd", "run", "coverage"])
    if BIOFLUIDS_AS_OOV:
        bio = pd.read_csv(RES / "MLMarker_v_HAMLET/healthy_native_biofluid_excluded.tsv", sep="\t")
        bio["is_in_mlm_vocab"] = False; bio["concordance_tier"] = "biofluid"; bio["lin_similarity"] = np.nan; bio["is_agree"] = False
        conc = pd.concat([conc, bio], ignore_index=True)
    conc = conc.merge(cov, on=["pxd", "run"], how="left")
    conc["usable"] = conc.coverage >= 0.10
    single = conc[~conc.tissue_llm.astype(str).str.contains(";")].copy()
    # a HAMLET tissue outside the 34 classes cannot be matched by construction:
    # report it as its own category instead of as conflict / distant
    single.loc[~single.is_in_mlm_vocab.astype(bool), "concordance_tier"] = OOV
    cancer = pd.read_csv(RES / "MLMarker_v_HAMLET/cancer_diseased_human_concordance.tsv", sep="\t")
    cancer = cancer.merge(cov, on=["pxd", "run"], how="left"); cancer["usable"] = cancer.coverage >= 0.10
    multi = pd.read_csv(RES / "granularity/A_multi_tissue_samples.csv")

    tiers = single.groupby(["usable", "concordance_tier"]).size().unstack("concordance_tier", fill_value=0).reindex(columns=TIERS, fill_value=0)
    tiers.to_csv(OUT / f"a_tiers_single_tissue_by_coverage{SUFFIX}.csv")
    single["bin"] = pd.cut(single.confidence_mlm, [b[0] for b in BINS] + [1.01], labels=[b[2] for b in BINS], right=False)
    calib = single.groupby("bin", observed=True).agg(n=("lin_similarity", "size"), mean_lin=("lin_similarity", "mean"),
                                                       exact=("is_agree", "mean")).reindex([b[2] for b in BINS])
    calib.to_csv(OUT / "b_calibration_single_tissue.csv")

    def p_a(ax):
        x = np.arange(len(TIERS)); w = 0.38
        for i, (u, lab, alpha) in enumerate([(True, "coverage ≥ 0.10", 1.0), (False, "coverage < 0.10", 0.45)]):
            v = tiers.loc[u] if u in tiers.index else pd.Series(0, index=TIERS)
            ax.bar(x + (i - 0.5) * w, v.values, w, color=[TIER_COLOR[t] for t in TIERS], alpha=alpha, label=lab)
            for xi, vv in zip(x + (i - 0.5) * w, v.values):
                ax.text(xi, vv + 5, f"{vv / v.sum():.0%}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([TIER_LABEL[t] for t in TIERS], fontsize=7.5, rotation=25, ha="right"); ax.set_ylabel("runs"); ax.grid(False, axis="x")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=NEUTRAL, alpha=1.0, label="high coverage"),
                           Patch(color=NEUTRAL, alpha=0.45, label="low coverage")], loc="upper left")
        panel_title(ax, "a")

    def p_b(ax):
        x = np.arange(len(calib))
        ax2 = ax.twinx()
        ax2.bar(x, calib.n, color=NEUTRAL, alpha=.25, width=0.6); ax2.set_ylabel("runs", color=NEUTRAL); ax2.grid(False)
        ax.plot(x, calib.mean_lin, "-o", color=AGENT["BiologicalAgent"], lw=1.8, ms=4, label="Lin similarity", zorder=3)
        ax.plot(x, calib.exact, "--s", color=AGENT["BiologicalAgent"], lw=1.4, ms=4, label="exact match", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(calib.index, fontsize=7.5, rotation=25, ha="right"); ax.set_ylim(0, 1); ax.set_xlabel("MLMarker confidence"); ax.set_ylabel("agreement with HAMLET")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.legend(loc="upper left"); panel_title(ax, "b")

    def p_c(ax):
        u = multi[multi.usable].copy()
        if EXCLUDE_TRAINING:
            tr = pd.read_csv(RES / "run_labels.tsv", sep="\t", usecols=["pxd", "in_old_atlas"]).drop_duplicates("pxd").set_index("pxd").in_old_atlas
            u = u[~u.pxd.map(tr).fillna(False).astype(bool)]
        cat = np.select([u.assigned.isna(), u.pred_tissue.eq(u.assigned)], ["no agreement", "top-1 agreement"], "top-5 agreement")
        counts = pd.Series(cat).value_counts().reindex(["top-1 agreement", "top-5 agreement", "no agreement"], fill_value=0)
        share = counts / counts.sum()
        colors = [LIN_KIND["same term"], LIN_KIND["related (Lin 0.5-0.99)"], LIN_KIND["unrelated (Lin <= 0.01)"]]
        x = np.arange(3)
        ax.bar(x, share, 0.6, color=colors)
        for xi, sh, n in zip(x, share, counts):
            ax.text(xi, sh + 0.02, f"{sh:.0%}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["top-1", "top-5", "none"], fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("share of usable files"); ax.grid(False, axis="x")
        ax.set_xlabel("MLMarker match with HAMLET's tissue list")
        panel_title(ax, "c")
        pd.DataFrame({"files": counts, "share": share.round(3)}).to_csv(OUT / f"c_multi_tissue_agreement{SUFFIX}.csv")

    def p_d(ax):
        """Arnaud's panel B: Lin similarity between HAMLET's tissue and
        MLMarker's call, healthy native vs cancer / diseased cohort (biofluids
        excluded), same per-run values as cohort_comparison_healthy_vs_cancer.tsv.
        Drawn as histograms with usable coverage on top of all runs."""
        from scipy.stats import gaussian_kde
        xs = np.linspace(-0.05, 1.2, 500)
        for cohort, df, color in [("healthy", conc, OUTCOME["accepted"]), ("cancer", cancer, OUTCOME["wrong"])]:
            v_all = df.lin_similarity.dropna().to_numpy(); v_use = df[df.usable].lin_similarity.dropna().to_numpy()
            ax.fill_between(xs, gaussian_kde(v_use, bw_method=0.12)(xs), color=color, alpha=.35, lw=0)
            ax.plot(xs, gaussian_kde(v_use, bw_method=0.12)(xs), color=color, lw=1.6, label=cohort)
        ax.set_ylim(0, None)
        ax.set_xlabel("Lin similarity (HAMLET vs MLMarker)"); ax.set_ylabel("density"); ax.set_xlim(0, 1.2)
        ax.legend(loc="upper left")
        panel_title(ax, "d")
        pd.DataFrame({"cohort": ["healthy native", "cancer / diseased"],
                      "n_all": [conc.lin_similarity.notna().sum(), cancer.lin_similarity.notna().sum()],
                      "mean_lin_all": [conc.lin_similarity.mean(), cancer.lin_similarity.mean()],
                      "n_high_coverage": [conc[conc.usable].lin_similarity.notna().sum(), cancer[cancer.usable].lin_similarity.notna().sum()],
                      "mean_lin_high_coverage": [conc[conc.usable].lin_similarity.mean(), cancer[cancer.usable].lin_similarity.mean()],
                      "exact_high_coverage": [conc[conc.usable].is_agree.mean(), cancer[cancer.usable].is_agree.mean()]}).round(3).to_csv(OUT / "d_healthy_vs_diseased_lin.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), gridspec_kw={"wspace": 0.35, "hspace": 0.45})
    p_a(axes[0, 0]); p_b(axes[0, 1]); p_c(axes[1, 0]); p_d(axes[1, 1])
    save_composite(fig, OUT, "figure_hamlet_vs_mlmarker" + SUFFIX); plt.close(fig)
    save_panels(plt, OUT, "figure_hamlet_vs_mlmarker" + SUFFIX, {"a": p_a, "b": p_b, "c": p_c, "d": p_d}, {"a": (6.5, 3.8), "b": (5.5, 3.8), "c": (5, 3.8), "d": (6, 3.8)})
    print(tiers.to_string()); print(calib.round(3).to_string()); print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
