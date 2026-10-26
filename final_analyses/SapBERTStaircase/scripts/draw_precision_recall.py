"""Precision, recall and error decomposition of the cumulative ablation, per model.

Three rows (one per model), two columns: precision and recall per arm with the
F1 line (mean +- SD over 3 runs), and the outcome of every golden value
(matched / wrong value / no value produced).

    python final_analyses/SapBERTStaircase/scripts/draw_precision_recall.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "manuscript_figure2"
sys.path.insert(0, str(ROOT.parents[0]))
from figure_style import (style, panel_title, legend_outside, save_composite, save_panels,  # noqa: E402
                          AGENT, AGENT_LABEL, AGENTS, MODEL as MODEL_COLORS, MODELS, OUTCOME, NEUTRAL)

ARMS = [f"S{i}" for i in range(6)]


def main():
    rep = pd.read_csv(FIG / "benchmark_metrics_per_replicate.csv")
    pairs = pd.read_csv(FIG / "scored_pairs_all_replicates.csv")
    plt = style()

    def pr_panel(ax, model, letter="", legend=False):
        g = rep[rep.model == model].groupby("arm")[["overall_precision", "overall_recall"]].agg(["mean", "std"]).reindex(ARMS)
        x = np.arange(len(ARMS)); w = 0.36
        for i, (metric, color, lab) in enumerate([("overall_precision", "#4f81b9", "precision"), ("overall_recall", "#d99457", "recall")]):
            m, sd = g[(metric, "mean")], g[(metric, "std")]
            ax.bar(x + (i - 0.5) * w, m, w, yerr=sd, color=color, label=lab, capsize=2, error_kw={"lw": .8, "ecolor": "#444444"})
        f1 = rep[rep.model == model].groupby("arm").overall_f1.mean().reindex(ARMS)
        ax.plot(x, f1, "k-o", ms=3.5, lw=1.2, label="F1", zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.grid(False, axis="x")
        ax.text(0.1, 1.02, model, transform=ax.transAxes, fontsize=9, fontweight="bold", color=MODEL_COLORS[model], va="bottom")
        if legend:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
        panel_title(ax, letter)

    def outcome_panel(ax, model, letter="", legend=False):
        p = pairs[pairs.model == model].copy()
        p["has"] = p.llm.notna()
        p["kind"] = np.select([p.has & (p.score >= 0.5), p.has], ["matched", "wrong value"], "no value produced")
        t = (p.groupby(["arm", "kind"]).size().unstack("kind", fill_value=0).reindex(ARMS)
             .reindex(columns=["matched", "wrong value", "no value produced"], fill_value=0))
        t = t.div(t.sum(axis=1), axis=0)
        x = np.arange(len(ARMS)); bottom = np.zeros(len(ARMS))
        for col, color in [("matched", OUTCOME["accepted"]), ("wrong value", OUTCOME["wrong"]), ("no value produced", NEUTRAL)]:
            v = t[col].to_numpy()
            ax.bar(x, v, 0.62, bottom=bottom, color=color, label=col, edgecolor="white", lw=.4)
            for xi, b, w in zip(x, bottom, v):
                if w >= 0.07:
                    ax.text(xi, b + w / 2, f"{w:.0%}", ha="center", va="center", fontsize=7, color="white")
            bottom += v
        ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 1); ax.set_ylabel("share of ground truth values"); ax.grid(False, axis="x")
        if legend:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
        panel_title(ax, letter)
        return t

    def pooled_pr(ax, letter="a"):
        g = rep.groupby("arm")[["overall_precision", "overall_recall"]].agg(["mean", "std"]).reindex(ARMS)
        x = np.arange(len(ARMS)); w = 0.36
        for i, (metric, color, lab) in enumerate([("overall_precision", "#4f81b9", "precision"), ("overall_recall", "#d99457", "recall")]):
            ax.bar(x + (i - 0.5) * w, g[(metric, "mean")], w, yerr=g[(metric, "std")], color=color, label=lab, capsize=2, error_kw={"lw": .8, "ecolor": "#444444"})
        ax.plot(x, rep.groupby("arm").overall_f1.mean().reindex(ARMS), "k-o", ms=4, lw=1.4, label="F1", zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.grid(False, axis="x")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9, frameon=False); panel_title(ax, letter)

    def pooled_outcome(ax, letter="b"):
        q = pairs.copy(); q["has"] = q.llm.notna()
        q["kind"] = np.select([q.has & (q.score >= 0.5), q.has], ["matched", "wrong value"], "no value produced")
        t = (q.groupby(["arm", "kind"]).size().unstack("kind", fill_value=0).reindex(ARMS)
             .reindex(columns=["matched", "wrong value", "no value produced"], fill_value=0))
        t = t.div(t.sum(axis=1), axis=0)
        x = np.arange(len(ARMS)); bottom = np.zeros(len(ARMS))
        for col, color in [("matched", OUTCOME["accepted"]), ("wrong value", OUTCOME["wrong"]), ("no value produced", NEUTRAL)]:
            v = t[col].to_numpy()
            ax.bar(x, v, 0.62, bottom=bottom, color=color, label=col, edgecolor="white", lw=.4)
            for xi, b, wv in zip(x, bottom, v):
                if wv >= 0.06:
                    ax.text(xi, b + wv / 2, f"{wv:.0%}", ha="center", va="center", fontsize=8, color="white")
            bottom += v
        ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 1); ax.set_ylabel("share of ground truth values"); ax.grid(False, axis="x")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9, frameon=False); panel_title(ax, letter)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.3})
    pooled_pr(axes[0]); pooled_outcome(axes[1])
    save_composite(fig, FIG, "figure_precision_recall_pooled"); plt.close(fig)
    save_panels(plt, FIG, "figure_precision_recall_pooled", {"a": pooled_pr, "b": pooled_outcome}, {"a": (5.2, 4.2), "b": (5.2, 4.2)})

    letters = iter("abcdef")
    tabs = {}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    for row, model in enumerate(MODELS):
        last = row == len(MODELS) - 1
        pr_panel(axes[row, 0], model, next(letters), legend=last)
        tabs[model] = outcome_panel(axes[row, 1], model, next(letters), legend=last)
    save_composite(fig, FIG, "figureS2_precision_recall"); plt.close(fig)
    pd.concat(tabs, names=["model"]).round(3).to_csv(FIG / "golden_outcome_by_arm.csv")
    panels = {}
    for row, model in enumerate(MODELS):
        key = model.split()[0].lower().replace("-", "").replace(".", "")
        panels[f"{key}_pr"] = (lambda ax, m=model: pr_panel(ax, m, legend=True))
        panels[f"{key}_outcome"] = (lambda ax, m=model: outcome_panel(ax, m, legend=True))
    save_panels(plt, FIG, "figureS2_precision_recall", panels, {k: (5.5, 3.6) for k in panels})
    print(pd.concat(tabs, names=["model"]).round(3).to_string())


if __name__ == "__main__":
    main()
