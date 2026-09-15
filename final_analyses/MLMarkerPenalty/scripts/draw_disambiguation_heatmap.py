"""Complementary view of file-level disambiguation: one hue, no per-tissue
colours. Rows = multi-tissue projects (non-training first), columns = the
tissues HAMLET listed; a cell is the share of the project's usable runs that
MLMarker assigned to that tissue, annotated with the run count. Blank = the
tissue was not in HAMLET's list. A ring marks tissues confirmed by SDRF /
run-name labels; the last column is the share left unresolved.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
OUT = ROOT / "results" / "fig5_concordance_figure"
sys.path.insert(0, str(REPO / "final_analyses"))
from figure_style import style, panel_title, save_composite, HEAT_CMAP  # noqa: E402


def main():
    plt = style()
    m = pd.read_csv(ROOT / "results/granularity/A_multi_tissue_samples.csv")
    m["cand_classes"] = m.cand_classes.map(ast.literal_eval)
    tr = pd.read_csv(ROOT / "results/propose_choose/atlas_runs.tsv", sep="\t", usecols=["pxd", "in_training"]).drop_duplicates().set_index("pxd").in_training
    u = m[m.usable].copy()
    projects = (u.groupby("pxd").size().rename("n").to_frame())
    projects["in_training"] = projects.index.map(tr).fillna(False).astype(bool)
    projects["k"] = u.groupby("pxd").k.first()
    projects = projects.sort_values(["in_training", "k", "n"], ascending=[True, True, False])
    tissues = sorted({t for c in u.cand_classes for t in c})
    share = pd.DataFrame(np.nan, index=projects.index, columns=tissues + ["unresolved"])
    count = share.copy()
    for pxd, g in u.groupby("pxd"):
        for t in g.cand_classes.iloc[0]:
            share.loc[pxd, t] = (g.assigned == t).mean(); count.loc[pxd, t] = (g.assigned == t).sum()
        share.loc[pxd, "unresolved"] = g.assigned.isna().mean(); count.loc[pxd, "unresolved"] = g.assigned.isna().sum()
    truth = {(pxd, t) for pxd, g in u.groupby("pxd") for t in set(g.label_class.dropna())}

    fig, ax = plt.subplots(figsize=(0.42 * share.shape[1] + 3, 0.5 * len(share) + 2))
    masked = np.ma.masked_invalid(share.to_numpy(float))
    cmap = plt.get_cmap(HEAT_CMAP).copy(); cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, pxd in enumerate(share.index):
        for j, t in enumerate(share.columns):
            v = share.iloc[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#eeeeee", lw=.5))
                continue
            if count.iloc[i, j] == 0:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#e6e6e6", edgecolor="white", lw=1.2))
                ax.text(j, i, "0", ha="center", va="center", fontsize=6.5, color="#888888")
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="white", lw=1.2))
            if count.iloc[i, j] > 0:
                ax.text(j, i, f"{int(count.iloc[i, j])}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else "black")
            if (pxd, t) in truth:
                ax.add_patch(plt.Rectangle((j - .42, i - .42), .84, .84, fill=False, edgecolor="k", lw=1.6))
    ax.set_xticks(range(share.shape[1])); ax.set_xticklabels(share.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([f"{p}  ({k} listed, {n} files){'  · training' if t else ''}" for p, k, n, t in zip(share.index, projects.k, projects.n, projects.in_training)], fontsize=8)
    n_new = int((~projects.in_training).sum())
    ax.axhline(n_new - 0.5, color="k", lw=1)
    ax.text(-0.5, n_new - 0.5, " projects in MLMarker's training set ↓", va="bottom", ha="left", fontsize=7, color="gray")
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("share of the project's usable files assigned to the tissue")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fill=False, edgecolor="k", lw=1.6, label="tissue confirmed by SDRF / run-name labels"),
                       Patch(facecolor="#e6e6e6", edgecolor="white", label="listed by HAMLET, no file assigned"),
                       Patch(fill=False, edgecolor="#cccccc", label="not in HAMLET's list")],
              loc="upper left", bbox_to_anchor=(0, -0.28), ncol=3)
    panel_title(ax, "", "HAMLET's project-level tissue list, resolved per file by MLMarker")
    save_composite(fig, OUT, "figure_file_level_disambiguation_heatmap"); plt.close(fig)
    share.round(3).to_csv(OUT / "disambiguation_share_by_project_and_tissue.csv"); count.to_csv(OUT / "disambiguation_count_by_project_and_tissue.csv")
    print(f"wrote {OUT / 'figure_file_level_disambiguation_heatmap.png'}")


if __name__ == "__main__":
    main()
