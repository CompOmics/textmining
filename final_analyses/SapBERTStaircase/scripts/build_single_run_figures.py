#!/usr/bin/env python3
"""Build non-replicate SapBERT Staircase figures from the primary run."""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STAIRCASE = ROOT.parent / "Staircase"
INDEX = ROOT / "benchmark_runs/index.json"
INPUTS = STAIRCASE / "benchmark_inputs"
OUTPUT = ROOT / "figures"
ORIGINAL_FIGURE_SCRIPT = STAIRCASE / "scripts/build_figure3.py"
MODELS = ["qwen3_8_27b", "gemma4_31b", "glm4_7"]
ARMS = [f"S{i}" for i in range(6)]
AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]
MODEL_LABELS = {"qwen3_8_27b": "Qwen3.8-27B", "gemma4_31b": "Gemma 4 31B", "glm4_7": "GLM-4.7"}
AGENT_LABELS = {"BiologicalAgent": "Biological", "TechnicalAgent": "Technical", "ExperimentalDesignAgent": "Experimental design"}
COLORS = {"qwen3_8_27b": "#0072B2", "gemma4_31b": "#D55E00", "glm4_7": "#009E73"}
MARKERS = {"qwen3_8_27b": "o", "gemma4_31b": "s", "glm4_7": "^"}


def load_helpers():
    spec = importlib.util.spec_from_file_location("staircase_figure_helpers", ORIGINAL_FIGURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ORIGINAL_FIGURE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.png", dpi=300)
    fig.savefig(OUTPUT / f"{stem}.pdf")
    plt.close(fig)


def benchmark_rows() -> list[dict]:
    index = json.loads(INDEX.read_text())
    rows = []
    for run in index["runs"]:
        report_path = Path(run["report"].replace("${HAMLET_ROOT}", str(ROOT.parents[2])))
        report = json.loads(report_path.read_text())
        row = {"model": run["model"], "arm": run["arm"], **run["overall"]}
        for agent in AGENTS:
            metrics = report["metrics"][agent]
            row[f"{agent}_precision"] = metrics["weighted_precision"]
            row[f"{agent}_recall"] = metrics["weighted_recall"]
            row[f"{agent}_f1"] = metrics["weighted_f1"]
        rows.append(row)
    expected = {(model, arm) for model in MODELS for arm in ARMS}
    observed = {(row["model"], row["arm"]) for row in rows}
    if observed != expected:
        raise RuntimeError(f"incomplete benchmark index: missing={sorted(expected - observed)}")
    return rows


def output_rows(helpers) -> tuple[list[dict], list[dict]]:
    loaded = {}
    schemas = {agent: set() for agent in AGENTS}
    for model in MODELS:
        for arm in ARMS:
            records = helpers.load_arm(INPUTS / model / arm)
            loaded[(model, arm)] = records
            if arm != "S0":
                for (_, agent), fields in records.items():
                    schemas[agent].update(field for field in fields if not field.startswith("_"))

    overall, by_field = [], []
    for model in MODELS:
        for arm in ARMS:
            records = loaded[(model, arm)]
            pxds = sorted({pxd for pxd, _ in records})
            field_counts, unknown, total = [], 0, 0
            field_unknown: dict[tuple[str, str], int] = defaultdict(int)
            for pxd in pxds:
                extracted = 0
                for agent in AGENTS:
                    fields = records[(pxd, agent)]
                    for field in sorted(schemas[agent]):
                        present = fields.get(field) is not None
                        extracted += int(present)
                        unknown += int(not present)
                        total += 1
                        field_unknown[(agent, field)] += int(not present)
                field_counts.append(extracted)
            overall.append({
                "model": model, "arm": arm, "n_inference_runs": 1,
                "n_pxds": len(pxds),
                "fields_per_pxd_mean": statistics.mean(field_counts),
                "fields_per_pxd_sd_across_pxds": statistics.stdev(field_counts),
                "unknown_rate": unknown / total,
            })
            for (agent, field), count in sorted(field_unknown.items()):
                by_field.append({
                    "model": model, "arm": arm, "agent": agent, "field": field,
                    "n_pxds": len(pxds), "unknown_rate": count / len(pxds),
                })
    return overall, by_field


def plot_performance(rows: list[dict]) -> None:
    lookup = {(row["model"], row["arm"]): row for row in rows}
    x = list(range(6))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model in MODELS:
        ax.plot(x, [lookup[(model, arm)]["f1"] for arm in ARMS], marker=MARKERS[model],
                linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set(xticks=x, xticklabels=ARMS, ylim=(0, 1.02), xlabel="Staircase arm",
           ylabel="Weighted F1", title="SDRF benchmark")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, loc="lower right")
    save(fig, "staircase_f1")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model in MODELS:
        ax.plot(x, [lookup[(model, arm)]["f1"] for arm in ARMS], marker=MARKERS[model],
                linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set(xticks=x, xticklabels=ARMS, ylim=(0, 1.02), xlabel="Staircase arm",
           ylabel="Weighted F1", title="a  Overall performance")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, loc="lower right")
    save(fig, "figure3_panel_a_overall_f1")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    for ax, agent in zip(axes, AGENTS):
        for model in MODELS:
            ax.plot(x, [lookup[(model, arm)][f"{agent}_f1"] for arm in ARMS],
                    marker=MARKERS[model], linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
        ax.set(xticks=x, xticklabels=ARMS, ylim=(0, 1.02), xlabel="Arm", title=AGENT_LABELS[agent])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Weighted F1"); axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("b  Performance by metadata class")
    save(fig, "figure3_panel_b_f1_by_class")

    comparisons = [("qwen3_8_27b", "S5"), ("gemma4_31b", "S0"), ("gemma4_31b", "S1"),
                   ("glm4_7", "S0"), ("glm4_7", "S1")]
    labels = ["Qwen\nS5", "Gemma\nS0", "Gemma\nS1", "GLM\nS0", "GLM\nS1"]
    means = [lookup[key]["f1"] for key in comparisons]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar(labels, means, color=[COLORS[model] for model, _ in comparisons], width=0.65)
    ax.set(ylabel="Weighted F1", ylim=(0, 1.02), title="f  Framework–model crossover comparison")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.018, f"{value:.3f}", ha="center")
    save(fig, "figure3_panel_f_crossover")


def plot_output_metrics(rows: list[dict]) -> None:
    lookup = {(row["model"], row["arm"]): row for row in rows}
    x = list(range(6))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model in MODELS:
        ax.plot(x, [lookup[(model, arm)]["fields_per_pxd_mean"] for arm in ARMS],
                marker=MARKERS[model], linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set(xticks=x, xticklabels=ARMS, xlabel="Staircase arm", ylabel="Non-unknown fields per PXD",
           title="c  Output volume")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    save(fig, "figure3_panel_c_output_volume")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model in MODELS:
        ax.plot(x, [lookup[(model, arm)]["unknown_rate"] for arm in ARMS],
                marker=MARKERS[model], linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax.set(xticks=x, xticklabels=ARMS, xlabel="Staircase arm", ylabel="Unknown rate",
           ylim=(0, 1.02), title="d  Unknown rate")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    save(fig, "figure3_panel_d_unknown_rate")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    helpers = load_helpers()
    metrics = benchmark_rows()
    outputs, fields = output_rows(helpers)
    write_csv(OUTPUT / "benchmark_metrics.csv", metrics)
    write_csv(OUTPUT / "output_volume_and_unknown_rate.csv", outputs)
    write_csv(OUTPUT / "unknown_rate_by_field.csv", fields)
    plot_performance(metrics)
    plot_output_metrics(outputs)
    provenance = {
        "benchmark": "SDRF benchmark",
        "semantic_embedding_model": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "semantic_classification_threshold": 0.70,
        "effective_final_metric_threshold": 0.50,
        "n_inference_runs": 1,
        "benchmark_index": str(INDEX.resolve()),
        "annotations": str(INPUTS.resolve()),
        "repeat_consistency_status": "not generated: requires multiple independent inference runs",
        "uncertainty_status": "no between-run SD or confidence interval for a single inference run",
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "figures": 6}, indent=2))


if __name__ == "__main__":
    main()
