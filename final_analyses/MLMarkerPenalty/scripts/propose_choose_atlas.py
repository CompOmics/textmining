"""HAMLET proposes, MLMarker chooses.

HAMLET gives a project a tissue list; MLMarker scores every run. The rule:
a run is assigned the HAMLET-listed tissue that ranks highest in MLMarker's
top-5 (single-tissue projects need no choice). Two parts:

1. Validation against run-level truth (SDRF or run-name tissue labels from
   build_run_labels.py; never HAMLET), split by whether the project is in
   MLMarker's training set (SupplementaryTableS1, via run_labels.in_old_atlas).
2. Application to every usable human run (coverage >= 0.10, no cell line)
   to produce a run-level tissue atlas with provenance.

    python final_analyses/MLMarkerPenalty/scripts/propose_choose_atlas.py

Writes results/propose_choose/{validation_*.csv, atlas_runs.tsv, atlas_summary_by_tissue.csv, summary.md}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
OUT = ROOT / "results" / "propose_choose"
sys.path.insert(0, str(HERE))
from granularity_and_disease import load, to_class, HEALTHY, MIN_COVERAGE  # noqa: E402
from hamlet_classes import MATERIAL_OK, MATERIAL_BAD, BIOFLUIDS  # noqa: E402

TOP = ["pred_tissue", "tissue_2", "tissue_3", "tissue_4", "tissue_5"]
CONF = ["confidence", "confidence_2", "confidence_3", "confidence_4", "confidence_5"]


def choose(row):
    """Best HAMLET candidate in MLMarker's top-5: (tissue, confidence, rank)."""
    for r, (t, c) in enumerate(zip(TOP, CONF), 1):
        if row[t] in row.cands:
            return row[t], row[c], r
    return None, np.nan, np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, ham = load()
    df = df.rename(columns={"tissue_x": "pred_tissue", "tissue_y": "label_tissue"})
    df["truth"] = np.where(df.tissue_source.isin(["sdrf", "name"]), df.label_tissue.map(to_class), None)
    df["truth_raw"] = np.where(df.tissue_source.isin(["sdrf", "name"]), df.label_tissue, None)
    df["cands"] = df.pxd.map(ham.cand_classes)
    df["hamlet_terms"] = df.pxd.map(ham.cands)
    df = df[df.cands.notna()].copy()
    df["k"] = df.cands.map(len)
    df["in_training"] = df.in_old_atlas.fillna(False).astype(bool)
    ch = df.apply(choose, axis=1, result_type="expand")
    df["chosen"], df["chosen_conf"], df["chosen_rank"] = ch[0], ch[1], ch[2]
    mt = df.material_type.fillna("").str.lower()
    df["native"] = mt.str.contains(MATERIAL_OK, regex=True) & ~mt.str.contains(MATERIAL_BAD, regex=True)
    df["biofluid_only"] = df.pxd.map(ham.biofluid_only).fillna(False)

    # ------------------------------------------------------------ 1. validation
    v = df[df.truth.notna() & df.usable].copy()
    v["truth_in_list"] = [t in c for t, c in zip(v.truth, v.cands)]
    v["unrestricted_ok"] = v.pred_tissue == v.truth
    v["chosen_ok"] = v.chosen == v.truth
    rows = []
    for name, sub in [("all", v), ("HAMLET lists >= 2 tissues", v[v.k >= 2]), ("HAMLET lists 1 tissue", v[v.k == 1])]:
        for tr_name, s in [("all projects", sub), ("in MLMarker training set", sub[sub.in_training]), ("not in training set", sub[~sub.in_training])]:
            if not len(s):
                continue
            rows.append({"subset": name, "training": tr_name, "projects": s.pxd.nunique(), "runs": len(s),
                         "truth within HAMLET list": round(s.truth_in_list.mean(), 3),
                         "MLMarker unrestricted top-1 == truth": round(s.unrestricted_ok.mean(), 3),
                         "chosen (restricted to list) == truth": round(s.chosen_ok.mean(), 3),
                         "chosen == truth | truth in list": round(s[s.truth_in_list].chosen_ok.mean(), 3) if s.truth_in_list.any() else np.nan,
                         "chance 1/k (k>=2 only)": round((1 / s[s.k >= 2].k).mean(), 3) if (s.k >= 2).any() else np.nan,
                         "chance unrestricted (1/34)": round(1 / 34, 3),
                         "chance restricted (mean 1/k over runs)": round((1 / s.k).mean(), 3),
                         "runs with conf >= 0.3": int((s.confidence >= 0.3).sum()),
                         "chosen == truth | conf >= 0.3": round(s[s.confidence >= 0.3].chosen_ok.mean(), 3) if (s.confidence >= 0.3).any() else np.nan})
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "validation_summary.csv", index=False)
    per = (v.groupby("pxd").apply(lambda g: pd.Series({
        "in_training": g.in_training.iloc[0], "truth_source": g.tissue_source.iloc[0], "k": g.k.iloc[0],
        "hamlet_list": "; ".join(g.cands.iloc[0]), "truth_tissues": "; ".join(sorted(set(g.truth))),
        "n_runs": len(g), "truth_in_list": g.truth_in_list.mean(), "unrestricted_acc": g.unrestricted_ok.mean(),
        "chosen_acc": g.chosen_ok.mean(), "mean_conf": g.confidence.mean()}), include_groups=False)
           .reset_index().sort_values(["k", "n_runs"], ascending=[False, False]))
    per.to_csv(OUT / "validation_per_project.csv", index=False)
    # truth projects with >= 2 tissues where HAMLET listed fewer: what HAMLET misses
    miss = per[(per.truth_tissues.str.count(";") >= 1) & (per.truth_in_list < 1)]
    miss.to_csv(OUT / "validation_hamlet_incomplete_lists.csv", index=False)

    # ------------------------------------------------------------ 2. atlas
    a = df[df.usable & ~df.biofluid_only & (df.k >= 1)].copy()
    n_out_of_vocab = int((df.usable & ~df.biofluid_only & (df.k == 0)).sum())
    a["disease_state"] = a.pxd.map(ham.diseases).map(
        lambda ds: "healthy" if ds and all(HEALTHY.match(x.lower().strip()) for x in ds)
        else ("diseased" if ds and not any(HEALTHY.match(x.lower().strip()) for x in ds) else ("mixed" if ds else "unknown")))
    a["tissue_assigned"] = np.where(a.truth.notna(), a.truth, np.where(a.k == 1, a.cands.map(lambda c: c[0]), a.chosen))
    a["label_source"] = np.select([a.truth.notna(), a.k == 1, a.chosen.notna()],
                                  ["SDRF / run name", "HAMLET single tissue", "HAMLET list + MLMarker choice"], "unresolved (no candidate in top-5)")
    a["mlmarker_agrees"] = a.pred_tissue == a.tissue_assigned
    cols = ["pxd", "run", "coverage", "scoring_mode", "tissue_assigned", "label_source", "k", "hamlet_terms",
            "pred_tissue", "confidence", "chosen_conf", "chosen_rank", "mlmarker_agrees", "in_training", "native",
            "material_type", "disease_state"]
    a[cols].to_csv(OUT / "atlas_runs.tsv", sep="\t", index=False)

    res = a[a.tissue_assigned.notna()]
    summ = (res.groupby(["tissue_assigned", "in_training"]).agg(
        runs=("run", "size"), projects=("pxd", "nunique"),
        healthy_runs=("disease_state", lambda s: int((s == "healthy").sum())),
        native_runs=("native", "sum"),
        from_sdrf_or_name=("label_source", lambda s: int((s == "SDRF / run name").sum())),
        from_hamlet_single=("label_source", lambda s: int((s == "HAMLET single tissue").sum())),
        from_mlmarker_choice=("label_source", lambda s: int((s == "HAMLET list + MLMarker choice").sum())),
        mlmarker_agrees=("mlmarker_agrees", "mean"), mean_confidence=("confidence", "mean")).reset_index())
    summ["in_training"] = summ.in_training.map({True: "in MLMarker training", False: "new"})
    summ = summ.sort_values(["tissue_assigned", "in_training"])
    summ.round(3).to_csv(OUT / "atlas_summary_by_tissue.csv", index=False)
    wide = summ.pivot(index="tissue_assigned", columns="in_training", values="runs").fillna(0).astype(int)
    wide["healthy_native_new_runs"] = res[(~res.in_training) & res.native & (res.disease_state == "healthy")].groupby("tissue_assigned").size()
    wide["new_projects"] = res[~res.in_training].groupby("tissue_assigned").pxd.nunique()
    wide = wide.fillna(0).astype(int).sort_values("new", ascending=False)
    wide.to_csv(OUT / "atlas_runs_by_tissue_wide.csv")

    overview = pd.Series({
        "usable human runs, not biofluid-only, HAMLET tissue in MLMarker vocabulary": len(a), "projects": a.pxd.nunique(),
        "usable runs whose HAMLET tissue is outside the 34 classes (excluded)": n_out_of_vocab,
        "assigned a tissue": int(a.tissue_assigned.notna().sum()),
        "  by SDRF / run name": int((a.label_source == "SDRF / run name").sum()),
        "  by HAMLET single tissue": int((a.label_source == "HAMLET single tissue").sum()),
        "  by HAMLET list + MLMarker choice": int((a.label_source == "HAMLET list + MLMarker choice").sum()),
        "unresolved (multi-tissue, no candidate in top-5)": int(a.tissue_assigned.isna().sum()),
        "runs in MLMarker training projects": int(a.in_training.sum()), "new runs": int((~a.in_training).sum()),
        "new runs, healthy, native": int(((~a.in_training) & a.native & (a.disease_state == "healthy") & a.tissue_assigned.notna()).sum()),
        "new runs, healthy, native, MLMarker agrees": int(((~a.in_training) & a.native & (a.disease_state == "healthy") & a.mlmarker_agrees).sum()),
        "MLMarker top-1 agrees with assigned label (all)": round(res.mlmarker_agrees.mean(), 3),
        "  in training projects": round(res[res.in_training].mlmarker_agrees.mean(), 3),
        "  new projects": round(res[~res.in_training].mlmarker_agrees.mean(), 3),
    })
    overview.to_csv(OUT / "atlas_overview.csv", header=["value"])

    print(val.to_string(index=False)); print(); print(per.head(20).to_string(index=False)); print()
    print(overview.to_string()); print(); print(wide.head(40).to_string())


def draw():
    sys.path.insert(0, str(REPO / "final_analyses"))
    from figure_style import style, panel_title, legend_outside, save_composite, save_panels, OUTCOME, NEUTRAL, AGENT, MODEL
    plt = style()
    val = pd.read_csv(OUT / "validation_summary.csv")
    wide = pd.read_csv(OUT / "atlas_runs_by_tissue_wide.csv", index_col=0)
    res = pd.read_csv(OUT / "atlas_runs.tsv", sep="\t")
    res = res[res.tissue_assigned.notna()]

    def p_a(ax):
        sub = val[(val.subset == "all") & (val.training != "all projects")]
        x = np.arange(len(sub)); w = 0.26
        series = [("chance 1/k (k>=2 only)", NEUTRAL, "chance (multi-tissue projects)"),
                  ("MLMarker unrestricted top-1 == truth", AGENT["TechnicalAgent"], "MLMarker alone"),
                  ("chosen (restricted to list) == truth", OUTCOME["accepted"], "HAMLET list + MLMarker choice")]
        for i, (col, color, lab) in enumerate(series):
            v = sub[col].fillna(0).to_numpy()
            ax.bar(x + (i - 1) * w, v, w, color=color, label=lab)
            for xi, vv in zip(x + (i - 1) * w, v):
                if vv > 0:
                    ax.text(xi, vv + 0.02, f"{vv:.0%}", ha="center", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{t}\n({p} projects, {r} runs)" for t, p, r in zip(sub.training, sub.projects, sub.runs)], fontsize=8)
        ax.set_ylim(0, 1.08); ax.set_ylabel("run-level accuracy vs SDRF / run-name truth"); ax.grid(False, axis="x")
        legend_outside(ax, "below", ncol=3); panel_title(ax, "a", "Validation on runs with independent tissue labels")

    def p_b(ax):
        w = wide[(wide["in MLMarker training"] + wide["new"]) > 0].sort_values("new", ascending=True)
        y = np.arange(len(w))
        ax.barh(y, w["in MLMarker training"], color=NEUTRAL, height=0.65, label="runs from MLMarker training projects")
        ax.barh(y, w["new"], left=w["in MLMarker training"], color=MODEL["Gemma 4 31B"], height=0.65, label="new runs (diseased or healthy)")
        ax.barh(y, w["healthy_native_new_runs"], left=w["in MLMarker training"], color=OUTCOME["accepted"], height=0.65, label="of which healthy, native tissue")
        for yi, (t, n_new, n_p) in enumerate(zip(w.index, w["new"], w["new_projects"])):
            ax.text(w.loc[t, "in MLMarker training"] + n_new + 5, yi, f"{n_p} new projects" if n_p else "", va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(w.index, fontsize=8); ax.set_xlabel("usable runs assigned to the tissue"); ax.grid(False, axis="y")
        legend_outside(ax, "below", ncol=1); panel_title(ax, "b", "Run-level tissue atlas from HAMLET + MLMarker")

    def p_c(ax):
        src = res.label_source.value_counts()
        cols = {"HAMLET single tissue": AGENT["BiologicalAgent"], "HAMLET list + MLMarker choice": OUTCOME["accepted"], "SDRF / run name": NEUTRAL}
        ax.pie(src.values, labels=[f"{k}\n{v:,}" for k, v in src.items()], colors=[cols.get(k, NEUTRAL) for k in src.index],
               startangle=90, textprops={"fontsize": 8}, wedgeprops={"edgecolor": "white"})
        ax.grid(False); panel_title(ax, "c", "Where each run's label comes from")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.5, height_ratios=[1, 1.1])
    p_a(fig.add_subplot(gs[0, 0])); p_c(fig.add_subplot(gs[0, 1])); p_b(fig.add_subplot(gs[1, :]))
    save_composite(fig, OUT, "figure_propose_choose"); plt.close(fig)
    save_panels(plt, OUT, "figure_propose_choose", {"a": p_a, "b": p_b, "c": p_c}, {"a": (6.5, 4), "b": (9, 6), "c": (4.5, 4)})


if __name__ == "__main__":
    main()
    draw()
