"""Figure 2: the framework and what its components are worth.

Plotted from the aggregated staircase tables in ../figures/figure3/. Note the
directory name uses the OLD manuscript figure numbering; this is the current
Figure 2. Output goes to ../figures/manuscript_figure2/.

This replaces an earlier draft that pasted rendered PNGs, which could not be
restyled, relabelled or exported at publication resolution and carried no
Source Data.

  a  framework schematic (PLACEHOLDER, still to be drawn)
  b  overall F1 by ablation arm and model
  c  F1 by metadata class
  d  output consistency: fields per dataset, and three-run unanimity
  e  unknown rate
  f  framework-model crossover

Source: ../figures/figure3/
  benchmark_metrics.csv                     3 models x 6 arms, mean and SD over
                                            three independent full-inference runs
  output_consistency_and_unknown_rate.csv   fields per PXD, unanimity, unknown rate
  provenance.json                           thresholds and replicate paths

Axes read "Ablation arm", not "Staircase arm": the analysis was renamed to a
cumulative ablation study throughout the manuscript.

Usage: python final_analyses/Staircase/scripts/build_manuscript_figure2.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent          # final_analyses/Staircase/scripts
ANALYSIS = HERE.parent                          # final_analyses/Staircase
REPO = ANALYSIS.parents[1]                      # repository root
sys.path.insert(0, str(REPO))
from plot_style import COLORS, clean_axes, save_fig  # noqa: E402

# Source tables live in figures/figure3/, which keeps the name it was pushed
# under. That "3" is the OLD manuscript numbering; in the current numbering this
# is Figure 2. The directory is left as-is so it does not conflict with further
# pushes, and the output goes to figures/manuscript_figure2/ instead.
DATA = ANALYSIS / "figures/figure3"
METRICS = DATA / "benchmark_metrics.csv"
CONSIST = DATA / "output_consistency_and_unknown_rate.csv"
PROVENANCE = DATA / "provenance.json"
OUT = ANALYSIS / "figures/manuscript_figure2/figure2.png"
PANELS = ANALYSIS / "figures/manuscript_figure2/panels"

ARMS = ["S0", "S1", "S2", "S3", "S4", "S5"]
# display names; the CSV uses lowercase identifiers
MODELS = [
    ("qwen3_8_27b", "Qwen3.8-27B", COLORS["dark_blue"], "o"),
    ("gemma4_31b", "Gemma 4 31B", COLORS["orange"], "s"),
    ("glm4_7", "GLM-4.7", COLORS["green"], "^"),
]
AGENTS = [
    ("BiologicalAgent", "Biological"),
    ("TechnicalAgent", "Technical"),
    ("ExperimentalDesignAgent", "Experimental design"),
]

# The stages the schematic must show. Kept here so the placeholder doubles as
# the spec for whoever draws it; mirrors arms/S0..S5 in the staircase tree.
STAGES = [
    ("Input", "abstract + Materials and Methods\n+ PRIDE project record"),
    ("S1  Agents", "biological / technical /\nexperimental design"),
    ("S2  Schema", "field definitions, rules,\nexamples, explicit unknown"),
    ("S3  Validation", "structural checks\nand retries"),
    ("S4  Gleaning", "second pass over\nmissed fields"),
    ("S5  Normalisation", "SapBERT + FAISS over\n19 ontologies"),
]


def load(path):
    return list(csv.DictReader(open(path)))


def series(rows, model, col):
    """Value per arm for one model, in ARMS order."""
    idx = {r["arm"]: r for r in rows if r["model"] == model}
    return np.array([float(idx[a][col]) for a in ARMS])


def draw_schematic_placeholder(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.008",
        facecolor="0.975", edgecolor=COLORS["red"], lw=1.6,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(0.5, 0.94, "PLACEHOLDER\nSCHEMATIC TO BE DRAWN",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=COLORS["red"], zorder=3, linespacing=1.3)
    ax.text(0.5, 0.855,
            "Must make the separability of the components visible,\n"
            "since that is what panels b to f exploit.",
            ha="center", va="top", fontsize=8, style="italic",
            color="0.4", zorder=3)

    n = len(STAGES)
    top, bot = 0.80, 0.09
    h = (top - bot) / n
    for i, (name, detail) in enumerate(STAGES):
        y = top - (i + 1) * h + h * 0.12
        ax.add_patch(FancyBboxPatch(
            (0.08, y), 0.84, h * 0.74, boxstyle="round,pad=0.006",
            facecolor="white", edgecolor="0.72", lw=0.9, zorder=2))
        ax.text(0.5, y + h * 0.60, name, ha="center", va="center",
                fontsize=8.4, fontweight="bold", color="0.25", zorder=3)
        ax.text(0.5, y + h * 0.26, detail, ha="center", va="center",
                fontsize=6.6, color="0.45", zorder=3, linespacing=1.2)
        if i < n - 1:
            ax.annotate("", xy=(0.5, y - h * 0.10), xytext=(0.5, y),
                        arrowprops=dict(arrowstyle="-|>", color="0.6", lw=1.0),
                        zorder=3)
    ax.text(0.5, 0.055, "every value carries the sentence it came from",
            ha="center", va="center", fontsize=7.4, style="italic",
            color="0.45", zorder=3)
    ax.set_title("a  The framework", loc="left", fontsize=10.5,
                 fontweight="bold")


def arm_axis(ax, ylabel, title, letter):
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS)
    ax.set_xlabel("Ablation arm")
    ax.set_ylabel(ylabel)
    clean_axes(ax)
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=10.5,
                 fontweight="bold")


def panel_b(ax, m):
    x = np.arange(len(ARMS))
    for key, label, c, mk in MODELS:
        y = series(m, key, "overall_f1_mean")
        sd = series(m, key, "overall_f1_sd")
        ax.errorbar(x, y, yerr=sd, color=c, marker=mk, ms=6, lw=1.8,
                    capsize=3, label=label)
    ax.set_ylim(0, 1.05)
    arm_axis(ax, "Weighted F1", "Overall performance", "b")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    per_arm = np.array([series(m, key, "overall_f1_mean") for key, *_ in MODELS])
    spread_by_arm = per_arm.max(axis=0) - per_arm.min(axis=0)

    lowest_at_s0 = per_arm[:, 0].min()
    ax.annotate(f"{spread_by_arm[0]:.2f} F1 spread\nat S0",
                xy=(0, lowest_at_s0 + 0.02), xytext=(0.55, 0.30),
                fontsize=8, style="italic", color="0.35",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.9))

    # Read the convergence arm off the data rather than hardcoding it: the
    # first arm from which the between-model spread never again exceeds 0.05.
    converged = next((i for i in range(len(ARMS))
                      if all(s <= 0.05 for s in spread_by_arm[i:])), None)
    if converged is not None:
        ax.text((converged + len(ARMS) - 1) / 2, 0.99,
                f"models converge from {ARMS[converged]} on", fontsize=8,
                style="italic", color="0.35", ha="center", va="top")


def panel_c(axes, m):
    for ax, (agent, label) in zip(axes, AGENTS):
        x = np.arange(len(ARMS))
        for key, mlabel, c, mk in MODELS:
            y = series(m, key, f"{agent}_f1_mean")
            sd = series(m, key, f"{agent}_f1_sd")
            ax.errorbar(x, y, yerr=sd, color=c, marker=mk, ms=4.5, lw=1.5,
                        capsize=2.5, label=mlabel)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(ARMS, fontsize=8)
        ax.set_xlabel("Ablation arm", fontsize=8.5)
        clean_axes(ax)
        ax.set_title(label, fontsize=9)
    axes[0].set_ylabel("Weighted F1")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].text(0.0, 1.20, "c  Performance by metadata class",
                 transform=axes[0].transAxes, fontsize=10.5, fontweight="bold")


def panel_d(axes, c):
    ax1, ax2 = axes
    x = np.arange(len(ARMS))
    for key, label, col, mk in MODELS:
        y = series(c, key, "fields_per_pxd_mean")
        sd = series(c, key, "fields_per_pxd_sd")
        ax1.errorbar(x, y, yerr=sd, color=col, marker=mk, ms=5, lw=1.6,
                     capsize=3, label=label)
    arm_axis(ax1, "Non-unknown fields per dataset", "Output volume", "d")
    ax1.set_title("d  Output consistency: fields per dataset", loc="left",
                  fontsize=10.5, fontweight="bold")

    for key, label, col, mk in MODELS:
        ax2.plot(x, series(c, key, "repeat_unanimous_agreement"),
                 color=col, marker=mk, ms=5, lw=1.6, label=label)
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks(range(len(ARMS)))
    ax2.set_xticklabels(ARMS)
    ax2.set_xlabel("Ablation arm")
    ax2.set_ylabel("Three-run unanimity")
    clean_axes(ax2)
    ax2.set_title("    and value agreement across three runs", loc="left",
                  fontsize=9.5, style="italic", color="0.4")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")


def panel_e(ax, c):
    x = np.arange(len(ARMS))
    for key, label, col, mk in MODELS:
        y = series(c, key, "unknown_rate_mean")
        sd = series(c, key, "unknown_rate_sd")
        ax.errorbar(x, y, yerr=sd, color=col, marker=mk, ms=5, lw=1.6,
                    capsize=3, label=label)
    ax.set_ylim(0, 1.0)
    arm_axis(ax, "Unknown rate", "Cost of declining to assert", "e")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")


def panel_f(ax, m):
    bars = [
        ("Qwen3.8-27B\nS5 (full framework)", "qwen3_8_27b", "S5", COLORS["dark_blue"]),
        ("Gemma 4 31B\nS1", "gemma4_31b", "S1", COLORS["orange"]),
        ("GLM-4.7\nS1", "glm4_7", "S1", COLORS["green"]),
        ("Gemma 4 31B\nS0", "gemma4_31b", "S0", COLORS["orange"]),
        ("GLM-4.7\nS0", "glm4_7", "S0", COLORS["green"]),
    ]
    idx = {(r["model"], r["arm"]): r for r in m}
    vals = [float(idx[(k, a)]["overall_f1_mean"]) for _, k, a, _ in bars]
    sds = [float(idx[(k, a)]["overall_f1_sd"]) for _, k, a, _ in bars]
    cols = [c for *_, c in bars]
    alphas = [1.0, 0.55, 0.55, 0.55, 0.55]

    x = np.arange(len(bars))
    for xi, (v, sd, c, al) in enumerate(zip(vals, sds, cols, alphas)):
        ax.bar(xi, v, yerr=sd, color=c, alpha=al, width=0.62, capsize=4)
        ax.text(xi, v + sd + 0.03, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.8)
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0, 1.12)
    clean_axes(ax)
    ax.axhline(vals[0], color=COLORS["dark_blue"], ls=(0, (4, 3)), lw=1.0,
               zorder=0)
    ax.set_title("f  The weakest model inside the framework beats stronger models without it",
                 loc="left", fontsize=10.5, fontweight="bold")


def main():
    m = load(METRICS)
    c = load(CONSIST)
    prov = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}
    n_rep = len(prov.get("replicates", [])) or 3

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 1.0],
                          height_ratios=[1.0, 0.85, 0.9],
                          hspace=0.52, wspace=0.36)

    draw_schematic_placeholder(fig.add_subplot(gs[0:2, 0]))
    panel_b(fig.add_subplot(gs[0, 1:3]), m)
    panel_e(fig.add_subplot(gs[0, 3]), c)
    panel_c([fig.add_subplot(gs[1, i]) for i in (1, 2, 3)], m)
    panel_d([fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])], c)
    panel_f(fig.add_subplot(gs[2, 2:]), m)

    fig.suptitle(
        "Framework structure, not model identity, carries extraction performance\n"
        f"30 held-out datasets, three open-weights models, {n_rep} independent runs per arm",
        fontsize=13, fontstyle="italic", y=0.975)

    fig.text(0.5, 0.008,
             "Source data: final_analyses/Staircase/figures/figure3/. "
             f"Semantic threshold {prov.get('semantic_classification_threshold', 0.7)}, "
             f"final metric threshold {prov.get('effective_final_metric_threshold', 0.5)}. "
             "Panel a is a placeholder and still has to be drawn.",
             ha="center", va="bottom", fontsize=8.5, style="italic",
             color="0.4")

    save_fig(fig, OUT)
    build_individual_panels(m, c)


def build_individual_panels(m, c):
    """Also render each panel on its own, for talks and for co-authors who want
    one result rather than six. Same styling as in the composite."""
    PANELS.mkdir(parents=True, exist_ok=True)

    def one(name, size, fn):
        f = plt.figure(figsize=size)
        fn(f)
        f.tight_layout()
        out = PANELS / f"figure2_panel_{name}.png"
        f.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(f)
        print(f"Saved: {out}")

    one("a_framework_schematic", (4.5, 8.0),
        lambda f: draw_schematic_placeholder(f.add_subplot(111)))
    one("b_overall_f1", (7.0, 5.0), lambda f: panel_b(f.add_subplot(111), m))
    one("c_f1_by_metadata_class", (11.0, 4.2),
        lambda f: panel_c([f.add_subplot(1, 3, i + 1) for i in range(3)], m))
    one("d_output_consistency", (10.0, 4.2),
        lambda f: panel_d([f.add_subplot(1, 2, 1), f.add_subplot(1, 2, 2)], c))
    one("e_unknown_rate", (6.0, 5.0), lambda f: panel_e(f.add_subplot(111), c))
    one("f_crossover", (8.0, 5.0), lambda f: panel_f(f.add_subplot(111), m))


if __name__ == "__main__":
    main()
