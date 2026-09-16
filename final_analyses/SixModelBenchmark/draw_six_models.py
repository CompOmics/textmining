"""Six-model benchmark F1 in the manuscript style (reads f1_all_models.csv written
by plot_all_models_f1.py; scoring in score_six_models.py).

    python final_analyses/SixModelBenchmark/draw_six_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import style, panel_title, legend_outside, save_composite, save_panels, SIX_MODEL as MODEL_COLORS  # noqa: E402
GROUPS = [("Biological", "Biological"), ("Technical", "Technical"), ("Experimental\nDesign", "Experimental design"), ("Overall", "Overall")]


MODELS = {"llama": "Llama-4 Scout", "gpt": "GPT-5.4", "claude": "Claude Opus 4.5", "gemini": "Gemini 3.5 Flash", "gemma": "Gemma 4 31B", "qwen": "Qwen 3.7 Max"}
AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]


def f1_table():
    """Same metric as Figure 2 (rescore_replicates.py): framework
    calculate_weighted_metrics per agent (a value string, 'unknown' included,
    counts as an attempt; only a missing value is a plain FN), overall = mean
    over the three agents."""
    sys.path.insert(0, str(HERE.parents[1] / "framework"))
    from benchmark.semantic_matcher import calculate_weighted_metrics
    rows = []
    for m, label in MODELS.items():
        d = pd.read_csv(HERE.parents[1] / "six_model_benchmark" / f"reports_test_v2_{m}_nopride" / "sdrf_benchmark_detailed.csv")
        r = {"model": label}
        for a, col in zip(AGENTS, ["Biological", "Technical", "Experimental\nDesign"]):
            g = d[d.agent == a]
            r[col] = calculate_weighted_metrics([{"score": s, "llm_has_value": pd.notna(l), "golden_has_value": True} for s, l in zip(g.score, g.llm)])["weighted_f1"]
        r["Overall"] = sum(r[c] for c in ["Biological", "Technical", "Experimental\nDesign"]) / 3
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model"); t.round(3).to_csv(HERE / "f1_all_models_framework_metric.csv"); return t


def six_model_panel(ax, f1, letter=""):
    """Grouped bars, one group per agent plus the mean over agents."""
    x = np.arange(len(GROUPS)); n = len(f1); w = 0.8 / n
    for i, (model, row) in enumerate(f1.iterrows()):
        ax.bar(x + (i - (n - 1) / 2) * w, [row[c] for c, _ in GROUPS], w, color=MODEL_COLORS[model], label=model, edgecolor="white", lw=.4)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in GROUPS])
    ax.set_ylabel("weighted F1"); ax.set_ylim(0.4, 1.0); ax.grid(False, axis="x")
    ax.axvline(2.5, color="#bbbbbb", lw=0.8, ls=":")
    legend_outside(ax, "right"); panel_title(ax, letter)


def main():
    f1 = f1_table(); print(f1.round(3).to_string())
    plt = style()

    panel = lambda ax: six_model_panel(ax, f1, letter="")
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    panel(ax)
    save_composite(fig, HERE, "figure_six_models"); plt.close(fig)
    save_panels(plt, HERE, "figure_six_models", {"a": panel}, {"a": (9.5, 4.2)})
    print("wrote", HERE / "figure_six_models.png")


if __name__ == "__main__":
    main()
