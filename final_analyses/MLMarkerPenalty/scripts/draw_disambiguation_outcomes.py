"""Per-file decision outcome for every multi-tissue project: how each raw
file was resolved when HAMLET listed several tissues.

  top-1 in HAMLET's list          MLMarker's own call is one of the listed tissues
  rescued from rank 2-5           top-1 was outside the list; a listed tissue sat in the top-5
  no listed tissue in top-5       unresolved
  coverage < 0.10                 no usable signal
Where SDRF / run-name labels exist, the number of correct / wrong decisions
is written on the bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
OUT = ROOT / "results" / "fig5_concordance_figure"
sys.path.insert(0, str(REPO / "final_analyses"))
from figure_style import style, panel_title, legend_outside, save_composite  # noqa: E402

CATS = [("top-1 in HAMLET's list", "#1f6b45"), ("rescued from rank 2-5", "#8fb48f"),
        ("no listed tissue in top-5", "#e6d9a0"), ("coverage < 0.10", "#c8c8c8")]


def main():
    plt = style()
    m = pd.read_csv(ROOT / "results/granularity/A_multi_tissue_samples.csv")
    tr = pd.read_csv(ROOT / "results/propose_choose/atlas_runs.tsv", sep="\t", usecols=["pxd", "in_training"]).drop_duplicates().set_index("pxd").in_training
    m["cat"] = np.select([~m.usable, m.assigned.isna(), m.pred_tissue.eq(m.assigned)],
                         ["coverage < 0.10", "no listed tissue in top-5", "top-1 in HAMLET's list"], "rescued from rank 2-5")
    proj = m.groupby("pxd").agg(n=("run", "size"), k=("k", "first"))
    proj["training"] = proj.index.map(tr).fillna(False).astype(bool)
    counts = m.groupby(["pxd", "cat"]).size().unstack("cat", fill_value=0).reindex(columns=[c for c, _ in CATS], fill_value=0)
    lab = m[m.usable & m.label_class.notna() & m.assigned.notna()]
    ok = lab.groupby("pxd").apply(lambda g: pd.Series({"ok": int((g.assigned == g.label_class).sum()), "bad": int((g.assigned != g.label_class).sum())}), include_groups=False)
    proj = proj.join(counts).join(ok)
    proj = proj.sort_values(["training", "n"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(proj) + 2.5))
    y = np.arange(len(proj)); left = np.zeros(len(proj))
    for cat, color in CATS:
        v = proj[cat].to_numpy()
        ax.barh(y, v, left=left, color=color, height=0.65, edgecolor="white", lw=.5, label=cat)
        for yi, l, w in zip(y, left, v):
            if w >= 4:
                ax.text(l + w / 2, yi, str(int(w)), ha="center", va="center", fontsize=7, color="white" if cat == CATS[0][0] else "black")
        left += v
    for yi, (p, row) in enumerate(proj.iterrows()):
        if not np.isnan(row.get("ok", np.nan)):
            ax.text(row.n + 1.5, yi, f"labels: {int(row.ok)} right, {int(row.bad)} wrong", va="center", fontsize=7, color="#444444")
    ax.set_yticks(y); ax.set_yticklabels([f"{p}  ({k} listed){'  · training' if t else ''}" for p, k, t in zip(proj.index, proj.k, proj.training)], fontsize=8)
    ax.invert_yaxis()
    n_new = int((~proj.training).sum()); ax.axhline(n_new - 0.5, color="k", lw=0.8)
    ax.set_xlabel("raw files"); ax.grid(False, axis="y"); ax.set_xlim(0, proj.n.max() * 1.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    panel_title(ax, "", "How each raw file of a multi-tissue project was resolved")
    save_composite(fig, OUT, "figure_file_level_disambiguation_outcomes"); plt.close(fig)
    proj.to_csv(OUT / "disambiguation_outcomes_by_project.csv")
    print(proj[[c for c, _ in CATS] + ["ok", "bad"]].to_string())


if __name__ == "__main__":
    main()
