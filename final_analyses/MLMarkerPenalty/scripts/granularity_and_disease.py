"""Two questions on MLMarker as a complement to HAMLET's project-level labels.

A. File-level tissue granularity. HAMLET labels a project; a multi-tissue
   project gets a list. Can MLMarker (per-run prediction) say which run is
   which tissue? Evaluated where run-level truth exists: SDRF or run-name
   tissue labels (build_run_labels.py, sources sdrf / name), never HAMLET.
B. Healthy vs diseased. MLMarker has no disease class; the only signal is
   confidence and agreement with the expected tissue. Within projects that
   mix healthy and diseased runs (run-level disease from SDRF / run name),
   does that signal separate the two?

Inputs
  results/run_meta_mlmarker_all_penalty.tsv   62,329 fraction-collapsed samples, coverage,
                                              scoring_mode, top-5 classes (predict_mlmarker_coverage.py)
  results/run_labels.tsv (build_run_labels.py)   run-level labels with provenance
  ../../mlmarker_hamlet/output/HAMLET_normalized.tsv   HAMLET project-level lists
  MLMarker/Reprocessing_database/parition_combiner/fraction_groups.tsv

    python final_analyses/MLMarkerPenalty/scripts/granularity_and_disease.py

Writes results/granularity/*.csv and summary.md.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
OUT = ROOT / "results" / "granularity"
PRED = ROOT / "results" / "run_meta_mlmarker_all_penalty.tsv"
LABELS = ROOT / "results" / "run_labels.tsv"
HAMLET = REPO / "mlmarker_hamlet/output/HAMLET_normalized.tsv"
FRACTIONS = REPO / "MLMarker/Reprocessing_database/parition_combiner/fraction_groups.tsv"
from hamlet_classes import CLASS_MAP, OLD_CLASSES, BIOFLUIDS  # noqa: E402

MIN_COVERAGE = 0.10
HEALTHY = re.compile(r"^(normal|healthy|disease free|control|none|no disease|not diseased|wild type)$")
CLASSES_LOWER = {c.lower(): c for c in OLD_CLASSES}


def to_class(term):
    """HAMLET / SDRF tissue term -> MLMarker class name, or None."""
    if not isinstance(term, str):
        return None
    t = term.strip().lower()
    if t in CLASS_MAP:
        return CLASS_MAP[t]
    return CLASSES_LOWER.get(t)


def hamlet_terms(cell):
    """HAMLET_normalized cell -> list of ontology names (or values)."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        obj = ast.literal_eval(cell)
    except Exception:
        return [cell]
    items = obj if isinstance(obj, list) else [obj]
    out = []
    for it in items:
        if isinstance(it, dict):
            v = it.get("ontology_name") or it.get("value")
            if isinstance(v, str) and v.lower() not in ("unknown", "false", ""):
                out.append(v)
        elif isinstance(it, str):
            out.append(it)
    return out


def load():
    pred = pd.read_csv(PRED, sep="\t")
    lab = pd.read_csv(LABELS, sep="\t", low_memory=False)
    fg = pd.read_csv(FRACTIONS, sep="\t", usecols=["pxd", "run", "sample_base"]).drop_duplicates()
    lab = lab.merge(fg, on=["pxd", "run"], how="left")
    lab["sample"] = lab.sample_base.fillna(lab.run)
    per = (lab.groupby(["pxd", "sample"]).agg(
        tissue=("tissue", "first"), tissue_source=("tissue_source", "first"),
        disease=("disease", "first"), disease_source=("disease_source", "first"),
        is_human=("is_human", "max"), has_cell_line=("has_cell_line", "max"), in_old_atlas=("in_old_atlas", "max"),
        material_type=("material_type", "first")).reset_index().rename(columns={"sample": "run"}))
    df = pred.merge(per, on=["pxd", "run"], how="inner")
    df = df[df.is_human.fillna(False).astype(bool) & ~df.has_cell_line.fillna(False).astype(bool)]
    df["usable"] = df.coverage >= MIN_COVERAGE

    ham = pd.read_csv(HAMLET, sep="\t", low_memory=False)
    ham["cands"] = ham.tissue.map(hamlet_terms)
    ham["cand_classes"] = ham.cands.map(lambda ts: sorted({c for c in (to_class(t) for t in ts) if c}))
    ham["biofluid_only"] = ham.cands.map(lambda ts: bool(ts) and all(t.lower() in BIOFLUIDS for t in ts))
    ham["diseases"] = ham.disease_state.map(hamlet_terms)
    ham = ham.set_index("source_file")
    return df, ham


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, ham = load()
    # prediction column 'tissue' vs label column 'tissue' collide on merge: rename first
    df = df.rename(columns={"tissue_x": "pred_tissue", "tissue_y": "label_tissue"}) if "tissue_x" in df else df
    if "pred_tissue" not in df:
        raise SystemExit("unexpected columns: " + ", ".join(df.columns))
    df["label_class"] = df.label_tissue.map(to_class)

    # ------------------------------------------------------------- A
    d = df.copy()
    d["cand_classes"] = d.pxd.map(ham.cand_classes)
    d = d[d.cand_classes.notna()]
    d["k"] = d.cand_classes.map(len)
    top = ["pred_tissue", "tissue_2", "tissue_3", "tissue_4", "tissue_5"]
    conf = ["confidence", "confidence_2", "confidence_3", "confidence_4", "confidence_5"]

    def restricted(row):
        for t, c in zip(top, conf):
            if row[t] in row.cand_classes:
                return row[t], row[c]
        return None, np.nan
    multi = d[d.k >= 2].copy()
    r = multi.apply(restricted, axis=1, result_type="expand")
    multi["assigned"], multi["assigned_conf"] = r[0], r[1]
    multi["top1_in_cands"] = multi.pred_tissue.eq(multi.assigned) & multi.assigned.notna()
    multi["chance"] = multi.k / 34
    u = multi[multi.usable]
    overview = pd.Series({
        "multi_tissue_projects (>=2 in-vocab HAMLET tissues)": multi.pxd.nunique(), "samples": len(multi),
        "samples usable (coverage >= 0.10)": int(multi.usable.sum()), "usable share": round(multi.usable.mean(), 3),
        "top-1 within HAMLET candidates, usable": round(u.top1_in_cands.mean(), 3),
        "chance (k/34), usable": round(u.chance.mean(), 3),
        "some candidate in top-5, usable": round(u.assigned.notna().mean(), 3),
        "top-1 within candidates, low coverage": round(multi[~multi.usable].top1_in_cands.mean(), 3),
    })
    overview.to_csv(OUT / "A_overview.csv", header=["value"])

    # truth-based accuracy: SDRF / run-name tissue, in vocab, in projects with >= 2 candidates
    t = multi[multi.tissue_source.isin(["sdrf", "name"]) & multi.label_class.notna()].copy()
    t["truth_in_cands"] = [c in cs for c, cs in zip(t.label_class, t.cand_classes)]
    tu = t[t.usable]
    acc = {
        "projects": t.pxd.nunique(), "samples": len(t), "usable": int(t.usable.sum()),
        "truth within HAMLET candidates": round(t.truth_in_cands.mean(), 3),
        "unrestricted top-1 == truth, usable": round((tu.pred_tissue == tu.label_class).mean(), 3),
        "restricted-to-candidates == truth, usable": round((tu.assigned == tu.label_class).mean(), 3),
        "restricted == truth, usable & conf >= 0.3": round((tu[tu.confidence >= 0.3].assigned == tu[tu.confidence >= 0.3].label_class).mean(), 3),
        "share usable with conf >= 0.3": round((tu.confidence >= 0.3).mean(), 3),
        "chance (1/k)": round((1 / tu.k).mean(), 3),
        "unrestricted top-1 == truth, low coverage": round((t[~t.usable].pred_tissue == t[~t.usable].label_class).mean(), 3),
    }
    pd.Series(acc).to_csv(OUT / "A_truth_accuracy.csv", header=["value"])
    per_proj = (tu.groupby("pxd").apply(lambda g: pd.Series({
        "n_usable": len(g), "k": g.k.iloc[0], "candidates": "; ".join(g.cand_classes.iloc[0]),
        "truth_classes": "; ".join(sorted(set(g.label_class))),
        "acc_unrestricted": (g.pred_tissue == g.label_class).mean(),
        "acc_restricted": (g.assigned == g.label_class).mean(),
        "mean_conf": g.confidence.mean()})).reset_index().sort_values("n_usable", ascending=False))
    per_proj.to_csv(OUT / "A_per_project.csv", index=False)
    # confusion of restricted assignment vs truth
    conf_tab = pd.crosstab(tu.label_class, tu.assigned.fillna("none in top-5"))
    conf_tab.to_csv(OUT / "A_confusion_restricted.csv")
    multi[["pxd", "run", "coverage", "scoring_mode", "usable", "k", "cand_classes", "pred_tissue", "confidence",
           "assigned", "assigned_conf", "tissue_source", "label_class"]].to_csv(OUT / "A_multi_tissue_samples.csv", index=False)

    # ------------------------------------------------------------- B
    from scipy import stats as sps
    from sklearn.metrics import roc_auc_score
    b = df[df.disease_source.isin(["sdrf", "name"]) & df.disease.notna()].copy()
    b["healthy"] = b.disease.str.lower().str.strip().map(lambda x: bool(HEALTHY.match(x)))
    b["expected"] = b.pxd.map(ham.cand_classes)
    b = b[b.expected.notna()]
    b["hit"] = [p in e for p, e in zip(b.pred_tissue, b.expected)]
    mixed = b.groupby("pxd").healthy.nunique()
    mixed = mixed[mixed == 2].index
    bm = b[b.pxd.isin(mixed) & b.usable & (b.expected.map(len) > 0)]
    rows = []
    for pxd, g in bm.groupby("pxd"):
        if g.healthy.sum() < 3 or (~g.healthy).sum() < 3:
            continue
        rows.append({"pxd": pxd, "n_healthy": int(g.healthy.sum()), "n_diseased": int((~g.healthy).sum()),
                     "expected": "; ".join(g.expected.iloc[0]),
                     "auc_confidence": roc_auc_score(g.healthy, g.confidence),
                     "auc_hit": roc_auc_score(g.healthy, g.hit.astype(int)),
                     "hit_rate_healthy": g[g.healthy].hit.mean(), "hit_rate_diseased": g[~g.healthy].hit.mean(),
                     "conf_healthy": g[g.healthy].confidence.mean(), "conf_diseased": g[~g.healthy].confidence.mean(),
                     "top_pred_diseased": g[~g.healthy].pred_tissue.value_counts().index[0]})
    per_b = pd.DataFrame(rows).sort_values("n_diseased", ascending=False)
    per_b.to_csv(OUT / "B_within_project_separation.csv", index=False)
    pooled = bm[bm.pxd.isin(per_b.pxd)]
    summ_b = pd.Series({
        "projects with run-level healthy+diseased labels (usable, >=3 each)": len(per_b),
        "runs": len(pooled), "healthy runs": int(pooled.healthy.sum()),
        "pooled AUC confidence (healthy positive)": round(roc_auc_score(pooled.healthy, pooled.confidence), 3) if len(pooled) else np.nan,
        "pooled AUC hit": round(roc_auc_score(pooled.healthy, pooled.hit.astype(int)), 3) if len(pooled) else np.nan,
        "median within-project AUC confidence": round(per_b.auc_confidence.median(), 3),
        "projects with AUC confidence > 0.7": int((per_b.auc_confidence > 0.7).sum()),
        "projects with AUC confidence < 0.3 (diseased more confident)": int((per_b.auc_confidence < 0.3).sum()),
        "hit rate healthy / diseased (pooled)": f"{pooled[pooled.healthy].hit.mean():.3f} / {pooled[~pooled.healthy].hit.mean():.3f}",
        "mean confidence healthy / diseased (pooled)": f"{pooled[pooled.healthy].confidence.mean():.3f} / {pooled[~pooled.healthy].confidence.mean():.3f}",
        "Mann-Whitney p, confidence": sps.mannwhitneyu(pooled[pooled.healthy].confidence, pooled[~pooled.healthy].confidence).pvalue if len(pooled) else np.nan,
    })
    summ_b.to_csv(OUT / "B_summary.csv", header=["value"])

    # project-level: HAMLET says healthy-only vs diseased-only, in-vocab solid tissue, usable
    p = df[df.usable].copy()
    p["expected"] = p.pxd.map(ham.cand_classes); p["dis"] = p.pxd.map(ham.diseases)
    p = p[p.expected.map(lambda e: isinstance(e, list) and len(e) > 0)]
    p["proj_state"] = p.dis.map(lambda ds: "healthy" if ds and all(HEALTHY.match(x.lower().strip()) for x in ds)
                                 else ("diseased" if ds and not any(HEALTHY.match(x.lower().strip()) for x in ds) else "mixed/unknown"))
    p["hit"] = [t in e for t, e in zip(p.pred_tissue, p.expected)]
    p = p[~p.in_old_atlas.fillna(False).astype(bool)]
    # same material rule as the atlas: solid tissue / primary cells, no cultured derivative
    from hamlet_classes import MATERIAL_OK, MATERIAL_BAD
    mt = p.material_type.fillna("").str.lower()
    p = p[mt.str.contains(MATERIAL_OK, regex=True) & ~mt.str.contains(MATERIAL_BAD, regex=True)]
    pl = p.groupby(["pxd", "proj_state"]).agg(n=("hit", "size"), hit=("hit", "mean"), conf=("confidence", "mean")).reset_index()
    pl.to_csv(OUT / "B_project_level_per_project.csv", index=False)
    pl_all = pl.copy()
    pl = pl[pl.proj_state.isin(["healthy", "diseased"])]
    proj_summ = pl.groupby("proj_state").agg(projects=("pxd", "nunique"), runs=("n", "sum"), mean_hit=("hit", "mean"), mean_conf=("conf", "mean")).round(3)
    proj_summ["auc_project_level_conf"] = roc_auc_score(pl.proj_state == "healthy", pl.conf)
    proj_summ["auc_project_level_hit"] = roc_auc_score(pl.proj_state == "healthy", pl.hit)
    proj_summ.to_csv(OUT / "B_project_level.csv")

    draw(multi, tu, pl_all, per_b, bm)
    print(overview.to_string()); print(); print(pd.Series(acc).to_string()); print()
    print(per_proj.head(15).to_string(index=False)); print(); print(summ_b.to_string()); print(); print(proj_summ.to_string())
    print(); print(per_b.head(15).round(2).to_string(index=False))


def draw(multi, tu, pl_all, per_b, bm):
    sys.path.insert(0, str(REPO / "final_analyses"))
    from figure_style import style, panel_title, legend_outside, save_composite, save_panels, OUTCOME, NEUTRAL, AGENT
    plt = style()
    u = multi[multi.usable]
    proj = (u.groupby("pxd").agg(n=("run", "size"), k=("k", "first"), in_cands=("top1_in_cands", "mean"),
                                chance=("chance", "first"), cands=("cand_classes", "first")).sort_values("in_cands"))

    def p_a(ax):
        y = np.arange(len(proj))
        ax.barh(y, proj.in_cands, color=OUTCOME["accepted"], height=0.65, label="MLMarker top-1 within HAMLET's tissue list")
        ax.scatter(proj.chance, y, marker="|", s=120, color="k", label="chance (k / 34)", zorder=3)
        ax.set_yticks(y); ax.set_yticklabels([f"{p}  (k={k}, n={n})" for p, k, n in zip(proj.index, proj.k, proj.n)], fontsize=7.5)
        ax.set_xlim(0, 1); ax.set_xlabel("share of usable runs (coverage ≥ 0.10)"); ax.grid(False)
        legend_outside(ax, "below", ncol=1); panel_title(ax, "a", "Multi-tissue projects: does the run-level prediction land in HAMLET's list?")

    def p_b(ax):
        # subdivision: assigned candidate per project as stacked shares
        sub = u[u.assigned.notna()]
        tab = pd.crosstab(sub.pxd, sub.assigned, normalize="index").reindex(proj.index)
        left = np.zeros(len(tab)); cmap = plt.get_cmap("tab20")
        for i, col in enumerate(tab.columns):
            ax.barh(np.arange(len(tab)), tab[col].fillna(0), left=left, color=cmap(i % 20), height=0.65, label=col, edgecolor="white", lw=.4)
            left += tab[col].fillna(0).to_numpy()
        ax.set_yticks(np.arange(len(tab))); ax.set_yticklabels(tab.index, fontsize=7.5); ax.set_xlim(0, 1); ax.grid(False)
        ax.set_xlabel("share of runs assigned to each candidate tissue (best candidate in top-5)")
        legend_outside(ax, "right", fontsize=7); panel_title(ax, "b", "How MLMarker subdivides each project")

    def p_c(ax):
        rows = [("chance (1 / k)", (1 / tu.k).mean()), ("unrestricted top-1", (tu.pred_tissue == tu.label_class).mean()),
                ("restricted to HAMLET's list", (tu.assigned == tu.label_class).mean())]
        ax.bar(range(3), [v for _, v in rows], color=[NEUTRAL, AGENT["TechnicalAgent"], OUTCOME["accepted"]], width=0.6)
        for i, (_, v) in enumerate(rows):
            ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels([r for r, _ in rows], fontsize=8); ax.set_ylim(0, 1); ax.grid(False, axis="x")
        ax.set_ylabel("run-level accuracy")
        panel_title(ax, "c", f"Against SDRF / run-name truth ({tu.pxd.nunique()} projects, {len(tu)} usable runs)")

    def p_d(ax):
        order = ["healthy", "mixed/unknown", "diseased"]; cols = {"healthy": OUTCOME["accepted"], "mixed/unknown": NEUTRAL, "diseased": OUTCOME["wrong"]}
        rng = np.random.default_rng(0)
        for i, st in enumerate(order):
            g = pl_all[pl_all.proj_state == st]
            ax.scatter(np.full(len(g), i) + rng.uniform(-.18, .18, len(g)), g.hit, s=14 + g.n.clip(upper=200) / 4, color=cols[st], alpha=.6)
            ax.plot([i - .3, i + .3], [g.hit.median()] * 2, color="k", lw=1.5)
            ax.text(i, 1.04, f"{len(g)} projects", ha="center", fontsize=7.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(["healthy", "mixed / unknown", "diseased"]); ax.set_ylim(-0.02, 1.1)
        ax.set_ylabel("share of runs predicted as the expected tissue"); ax.grid(False, axis="x")
        panel_title(ax, "d", "Project health state (HAMLET) vs MLMarker hit rate; native tissue, training projects excluded")

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.9)
    p_a(fig.add_subplot(gs[0, 0])); p_b(fig.add_subplot(gs[0, 1])); p_c(fig.add_subplot(gs[1, 0])); p_d(fig.add_subplot(gs[1, 1]))
    save_composite(fig, OUT, "figure_granularity_disease"); plt.close(fig)
    save_panels(plt, OUT, "figure_granularity_disease", {"a": p_a, "b": p_b, "c": p_c, "d": p_d},
                {"a": (6.5, 5), "b": (7, 5), "c": (4.5, 3.6), "d": (6, 3.8)})


if __name__ == "__main__":
    main()
