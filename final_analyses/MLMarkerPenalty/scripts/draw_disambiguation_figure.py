"""Headline figure: HAMLET labels a project with several tissues; MLMarker
assigns each raw file to one of them.

One panel per example project. Each column is a raw file. Rows:
  HAMLET   the project-level list (one bar over all files)
  MLMarker the tissue chosen for that file (best HAMLET candidate in the
           top-5), tile opacity = MLMarker confidence, grey = no candidate
           in top-5 or coverage < 0.10
  truth    the run-level label from the SDRF / run name where one exists

Input: results/granularity/A_multi_tissue_samples.csv.
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
from figure_style import style, panel_title, save_composite, save_panels, NEUTRAL  # noqa: E402

EXAMPLES = [("PXD006401", "HAMLET: tonsil, kidney · SDRF truth (human files are all tonsil)"),
            ("PXD048734", "HAMLET: 7 tissues · cancer atlas, tissue of origin per sample"),
            ("PXD048647", "HAMLET: 2 tissues · no run-level truth"),
            ("PXD020192", "HAMLET: 11 tissues · SDRF truth, 40 organism parts · in MLMarker training set"),
            ("PXD010154", "HAMLET: 28 tissues · SDRF truth · in MLMarker training set"),
            ("PXD000561", "HAMLET: 14 tissues · draft human proteome map, SDRF truth · in MLMarker training set")]
TISSUE_COLORS = {  # muted, distinct; anything else falls back to tab20
    "Tonsil": "#6aa56a", "Kidney": "#4f81b9", "Lymph node": "#8f7fbf", "Spleen": "#d99457", "Lung": "#5fa8a0",
    "Liver": "#b8866b", "Colon": "#c9a06c", "Stomach": "#cf6f5f", "Ovary": "#d4a5c9", "Esophagus": "#9bb7d4",
    "Heart": "#a34f4f", "Brain": "#7a9cc6", "Testis": "#8fb48f", "Placenta": "#e6d9a0", "Adrenal gland": "#bfa6d9",
    "Pancreas": "#c8c8c8", "Prostate": "#a9d3ae", "Rectum": "#e0b48c", "Urinary bladder": "#9fc5e8", "Gallbladder": "#d9c28f",
    "Retina": "#b3b3b3", "Bone marrow": "#f0c1a8", "Monocytes": "#d6b8e0", "B-cells": "#c4d6a0", "Small intestine": "#e3c9a0",
    "Adipose tissue": "#f2d38c", "Endometrium": "#e8b4b8", "Appendix": "#b5d1c5", "Duodenum": "#d8c3a5",
    "Pituitary gland": "#c2b6d9", "Salivary gland": "#b9d9c8", "Skeletal muscle": "#cfa88a", "Thyroid": "#a3c4bc", "Oviduct": "#d9b8c4"}


def color(t):
    return TISSUE_COLORS.get(t, NEUTRAL)


def _samples():
    m = pd.read_csv(OUT.parent / "granularity" / "A_multi_tissue_samples.csv")
    m["cand_classes"] = m.cand_classes.map(ast.literal_eval)
    # run-level truth from validate_disambiguation.py replaces the run_labels one
    t = pd.read_csv(OUT / "disambiguation_truth.csv")[["pxd", "run", "truth_class", "truth_raw"]]
    m = m.drop(columns=["label_class"]).merge(t, on=["pxd", "run"], how="left").rename(columns={"truth_class": "label_class"})
    return m


def draw_project(ax, pxd, note, letter, m=None):
    from matplotlib.patches import Patch
    if m is None:
        m = _samples()
    if True:
        d = m[m.pxd == pxd].copy()
        d["shown"] = np.where(d.usable & d.assigned.notna(), d.assigned, None)
        d = d.sort_values(["shown", "assigned_conf"], ascending=[True, False], na_position="last").reset_index(drop=True)
        n = len(d); cands = d.cand_classes.iloc[0]
        has_truth = d.label_class.notna().any()
        rows = ["HAMLET (project)", "MLMarker (per file)"] + (["truth (per file)"] if has_truth else [])
        # HAMLET bar
        ax.barh(0, n, left=0, height=0.7, color="#e9e9e9", edgecolor="white")
        ax.text(n / 2, 0, "; ".join(cands) if len(cands) <= 7 else f"{len(cands)} tissues listed", ha="center", va="center", fontsize=8)
        # MLMarker tiles
        for i, r in d.iterrows():
            if r.shown is None:
                ax.barh(1, 1, left=i, height=0.7, color="white", edgecolor="#bbbbbb", hatch="////", lw=0.4)
            else:
                ax.barh(1, 1, left=i, height=0.7, color=color(r.shown), alpha=float(np.clip(0.35 + r.assigned_conf, 0.35, 1.0)), edgecolor="white", lw=0.3)
        if has_truth:
            for i, r in d.iterrows():
                if isinstance(r.label_class, str):
                    ok = r.label_class == r.shown
                    in_list = r.label_class in cands
                    ax.barh(2, 1, left=i, height=0.7, color=color(r.label_class), edgecolor="white", lw=0.3)
                    if not in_list:
                        ax.text(i + 0.5, 2, "+", ha="center", va="center", fontsize=7, color="k")
                    elif not ok and r.shown is not None:
                        ax.text(i + 0.5, 2, "×", ha="center", va="center", fontsize=7, color="k")
                elif isinstance(r.truth_raw, str):
                    ax.barh(2, 1, left=i, height=0.7, color="white", edgecolor="#999999", lw=0.5, hatch="....")
                else:
                    ax.barh(2, 1, left=i, height=0.7, color="white", edgecolor="#dddddd", lw=0.3)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8); ax.invert_yaxis()
        ax.set_xlim(0, n); ax.set_xticks([]); ax.grid(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        panel_title(ax, letter, f"{pxd}: {note}  ({n} files)")
        used = set(d.shown.dropna()) | (set(d.label_class.dropna()) if has_truth else set())
        flags = {"outside": bool(has_truth and d.truth_raw.notna().gt(d.label_class.notna()).any())}
        return used, flags



def main():
    from matplotlib.patches import Patch
    plt = style()
    m = _samples()
    letters = "abcdefgh"
    fig, axes = plt.subplots(len(EXAMPLES), 1, figsize=(13, 2.0 * len(EXAMPLES)), gridspec_kw={"hspace": 0.9})
    used, outside = set(), False
    for ax, (pxd, note), letter in zip(axes, EXAMPLES, letters):
        u, f = draw_project(ax, pxd, note, letter, m)
        used |= u; outside |= f["outside"]
    handles = [Patch(color=color(t), label=t) for t in sorted(used)]
    handles.append(Patch(facecolor="white", edgecolor="#bbbbbb", hatch="////", label="unresolved / coverage < 0.10"))
    if outside:
        handles.append(Patch(facecolor="white", edgecolor="#999999", hatch="....", label="truth outside MLMarker's classes"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=9, fontsize=7.5, frameon=False)
    save_composite(fig, OUT, "figure_file_level_disambiguation"); plt.close(fig)
    save_panels(plt, OUT, "figure_file_level_disambiguation",
                {l: (lambda ax, p=p, n=n_, l=l: draw_project(ax, p, n, l, m)) for (p, n_), l in zip(EXAMPLES, letters)},
                {l: (12, 2.3) for l in letters})
    print(f"wrote {OUT / 'figure_file_level_disambiguation.png'}")


if __name__ == "__main__":
    main()
