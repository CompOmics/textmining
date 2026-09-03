#!/usr/bin/env python3
"""Build supported Figure 3 panels from the frozen benchmark runs."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RUNS = ROOT / "benchmark_runs"
BENCHMARK_INPUTS = ROOT / "benchmark_inputs"
OUTPUT = ROOT / "figures" / "figure3"
ARMS = [f"S{i}" for i in range(6)]
MODELS = ["qwen3_8_27b", "gemma4_31b", "glm4_7"]
MODEL_LABELS = {
    "qwen3_8_27b": "Qwen3.8-27B",
    "gemma4_31b": "Gemma 4 31B",
    "glm4_7": "GLM-4.7",
}
COLORS = {"qwen3_8_27b": "#0072B2", "gemma4_31b": "#D55E00", "glm4_7": "#009E73"}
MARKERS = {"qwen3_8_27b": "o", "gemma4_31b": "s", "glm4_7": "^"}
AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]
AGENT_LABELS = {
    "BiologicalAgent": "Biological",
    "TechnicalAgent": "Technical",
    "ExperimentalDesignAgent": "Experimental design",
}
REPLICATES = [
    ("replicate_1", BENCHMARK_RUNS / "index.json", BENCHMARK_INPUTS),
    ("replicate_2", BENCHMARK_RUNS / "replicate_2" / "index.json", BENCHMARK_INPUTS / "replicate_2"),
    ("replicate_3", BENCHMARK_RUNS / "replicate_3" / "index.json", BENCHMARK_INPUTS / "replicate_3"),
]
PXD_RE = re.compile(r"PXD\d+")


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


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


def benchmark_metrics() -> list[dict]:
    collected: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for replicate, index_path, _ in REPLICATES:
        index = json.loads(index_path.read_text())
        for run in index["runs"]:
            model, arm = run["model"], run["arm"]
            report = json.loads(Path(run["report"]).read_text())
            for agent in AGENTS:
                metrics = report["metrics"][agent]
                for metric in ("weighted_precision", "weighted_recall", "weighted_f1"):
                    collected[(model, arm, f"{agent}:{metric}")].append(float(metrics[metric]))
            for metric in ("precision", "recall", "f1"):
                collected[(model, arm, f"overall:{metric}")].append(float(run["overall"][metric]))

    rows = []
    for model in MODELS:
        for arm in ARMS:
            row = {"model": model, "arm": arm, "n": 3}
            for scope in ["overall", *AGENTS]:
                names = ("precision", "recall", "f1") if scope == "overall" else (
                    "weighted_precision", "weighted_recall", "weighted_f1"
                )
                for name in names:
                    values = collected[(model, arm, f"{scope}:{name}")]
                    mean, sd = mean_sd(values)
                    short = name.removeprefix("weighted_")
                    row[f"{scope}_{short}_mean"] = mean
                    row[f"{scope}_{short}_sd"] = sd
            rows.append(row)
    return rows


def infer_agent(path: Path) -> str:
    lowered = str(path).lower()
    if "biological" in lowered:
        return "BiologicalAgent"
    if "technical" in lowered:
        return "TechnicalAgent"
    if "experimental" in lowered:
        return "ExperimentalDesignAgent"
    raise ValueError(f"cannot infer agent from {path}")


def raw_value(value):
    if isinstance(value, dict):
        return value.get("resolved", value.get("value"))
    if isinstance(value, list):
        return value[0] if value else None
    return value


def canonical(value) -> str | None:
    value = raw_value(value)
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = " ".join(value.strip().casefold().split())
    if not value or value in {"unknown", "null", "none"}:
        return None
    # Treat semicolon-delimited output as a set so order does not reduce
    # repeat agreement. Commas are retained because they often occur inside
    # scientific names and descriptions.
    parts = sorted({" ".join(part.split()) for part in value.split(";") if part.strip()})
    return "; ".join(parts) if parts else None


def load_arm(directory: Path) -> dict[tuple[str, str], dict[str, str | None]]:
    records: dict[tuple[str, str], dict[str, str | None]] = {}
    files = sorted(path for path in directory.rglob("*.json") if path.is_file())
    if len(files) != 90:
        raise RuntimeError(f"expected 90 adapted agent files under {directory}, found {len(files)}")
    for path in files:
        match = PXD_RE.search(path.name)
        if not match:
            raise RuntimeError(f"missing PXD identifier in {path}")
        agent = infer_agent(path)
        payload = json.loads(path.read_text())
        records[(match.group(), agent)] = {field: canonical(value) for field, value in payload.items()}
    if len(records) != 90:
        raise RuntimeError(f"expected 90 unique PXD-agent records under {directory}, found {len(records)}")
    return records


def output_metrics() -> tuple[list[dict], list[dict]]:
    loaded: dict[tuple[str, str, str], dict] = {}
    schemas: dict[str, set[str]] = {agent: set() for agent in AGENTS}
    for replicate, _, adapted_root in REPLICATES:
        for model in MODELS:
            for arm in ARMS:
                records = load_arm(adapted_root / model / arm)
                loaded[(replicate, model, arm)] = records
                if arm != "S0":
                    for (_, agent), fields in records.items():
                        schemas[agent].update(field for field in fields if not field.startswith("_"))

    overall_rows, field_rows = [], []
    for model in MODELS:
        for arm in ARMS:
            replicate_rates, field_counts = [], []
            per_field_rep: dict[tuple[str, str], list[float]] = defaultdict(list)
            for replicate, _, _ in REPLICATES:
                records = loaded[(replicate, model, arm)]
                pxds = sorted({pxd for pxd, _ in records})
                unknown, total = 0, 0
                for pxd in pxds:
                    extracted = 0
                    for agent in AGENTS:
                        fields = records[(pxd, agent)]
                        for field in sorted(schemas[agent]):
                            present = fields.get(field) is not None
                            extracted += int(present)
                            unknown += int(not present)
                            total += 1
                    field_counts.append(extracted)
                replicate_rates.append(unknown / total)
                for agent in AGENTS:
                    for field in sorted(schemas[agent]):
                        values = [records[(pxd, agent)].get(field) for pxd in pxds]
                        per_field_rep[(agent, field)].append(sum(v is None for v in values) / len(values))

            # Exact normalized unanimity across all three repeats, excluding
            # units that are unknown in all repeats.
            comparable, unanimous = 0, 0
            first_records = loaded[(REPLICATES[0][0], model, arm)]
            for pxd, agent in sorted(first_records):
                for field in sorted(schemas[agent]):
                    values = [loaded[(rep, model, arm)][(pxd, agent)].get(field) for rep, _, _ in REPLICATES]
                    if all(value is None for value in values):
                        continue
                    comparable += 1
                    unanimous += int(values[0] == values[1] == values[2])

            unknown_mean, unknown_sd = mean_sd(replicate_rates)
            overall_rows.append({
                "model": model, "arm": arm, "n_replicates": 3,
                "fields_per_pxd_mean": statistics.mean(field_counts),
                "fields_per_pxd_sd": statistics.stdev(field_counts),
                "repeat_unanimous_units": unanimous,
                "repeat_comparable_units": comparable,
                "repeat_unanimous_agreement": unanimous / comparable if comparable else 0.0,
                "unknown_rate_mean": unknown_mean,
                "unknown_rate_sd": unknown_sd,
            })
            for (agent, field), rates in sorted(per_field_rep.items()):
                mean, sd = mean_sd(rates)
                field_rows.append({
                    "model": model, "arm": arm, "agent": agent, "field": field,
                    "n_replicates": 3, "unknown_rate_mean": mean, "unknown_rate_sd": sd,
                })
    return overall_rows, field_rows


def plot_panel_a(metric_rows: list[dict]) -> None:
    lookup = {(row["model"], row["arm"]): row for row in metric_rows}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = range(len(ARMS))
    for model in MODELS:
        ax.errorbar(x, [lookup[(model, arm)]["overall_f1_mean"] for arm in ARMS],
                    yerr=[lookup[(model, arm)]["overall_f1_sd"] for arm in ARMS],
                    marker=MARKERS[model], linewidth=2, capsize=4, color=COLORS[model],
                    label=MODEL_LABELS[model])
    ax.set(xticks=list(x), xticklabels=ARMS, ylim=(0, 1.02), xlabel="Staircase arm",
           ylabel="Weighted F1", title="a  Overall performance (mean ± SD, n=3)")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, loc="lower right")
    save(fig, "figure3_panel_a_overall_f1")


def plot_panel_b(metric_rows: list[dict]) -> None:
    lookup = {(row["model"], row["arm"]): row for row in metric_rows}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    x = range(len(ARMS))
    for ax, agent in zip(axes, AGENTS):
        for model in MODELS:
            ax.errorbar(x, [lookup[(model, arm)][f"{agent}_f1_mean"] for arm in ARMS],
                        yerr=[lookup[(model, arm)][f"{agent}_f1_sd"] for arm in ARMS],
                        marker=MARKERS[model], linewidth=2, capsize=3, color=COLORS[model],
                        label=MODEL_LABELS[model])
        ax.set(xticks=list(x), xticklabels=ARMS, ylim=(0, 1.02), xlabel="Arm",
               title=AGENT_LABELS[agent]); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Weighted F1")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("b  Performance by metadata class (mean ± SD, n=3)")
    save(fig, "figure3_panel_b_f1_by_class")


def plot_panel_c(rows: list[dict]) -> None:
    lookup = {(row["model"], row["arm"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = range(len(ARMS))
    for model in MODELS:
        axes[0].errorbar(x, [lookup[(model, arm)]["fields_per_pxd_mean"] for arm in ARMS],
                         yerr=[lookup[(model, arm)]["fields_per_pxd_sd"] for arm in ARMS],
                         marker=MARKERS[model], linewidth=2, capsize=3, color=COLORS[model],
                         label=MODEL_LABELS[model])
        axes[1].plot(x, [lookup[(model, arm)]["repeat_unanimous_agreement"] for arm in ARMS],
                     marker=MARKERS[model], linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    axes[0].set(xticks=list(x), xticklabels=ARMS, xlabel="Arm", ylabel="Non-unknown fields per PXD",
                title="Output volume (mean ± SD across PXD-runs)")
    axes[1].set(xticks=list(x), xticklabels=ARMS, xlabel="Arm", ylabel="Three-run exact agreement",
                ylim=(0, 1.02), title="Normalized field-value unanimity")
    for ax in axes: ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("c  Output consistency")
    save(fig, "figure3_panel_c_consistency")


def plot_panel_d(rows: list[dict]) -> None:
    lookup = {(row["model"], row["arm"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = range(len(ARMS))
    for model in MODELS:
        ax.errorbar(x, [lookup[(model, arm)]["unknown_rate_mean"] for arm in ARMS],
                    yerr=[lookup[(model, arm)]["unknown_rate_sd"] for arm in ARMS],
                    marker=MARKERS[model], linewidth=2, capsize=4, color=COLORS[model],
                    label=MODEL_LABELS[model])
    ax.set(xticks=list(x), xticklabels=ARMS, xlabel="Staircase arm", ylabel="Unknown rate",
           ylim=(0, 1.02), title="d  Unknown rate (mean ± SD across runs, n=3)")
    ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False)
    save(fig, "figure3_panel_d_unknown_rate")


def plot_panel_f(metric_rows: list[dict]) -> None:
    lookup = {(row["model"], row["arm"]): row for row in metric_rows}
    comparisons = [
        ("qwen3_8_27b", "S5"),
        ("gemma4_31b", "S0"), ("gemma4_31b", "S1"),
        ("glm4_7", "S0"), ("glm4_7", "S1"),
    ]
    labels = ["Qwen\nS5", "Gemma\nS0", "Gemma\nS1", "GLM\nS0", "GLM\nS1"]
    means = [lookup[key]["overall_f1_mean"] for key in comparisons]
    sds = [lookup[key]["overall_f1_sd"] for key in comparisons]
    colors = [COLORS[model] for model, _ in comparisons]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar(labels, means, yerr=sds, capsize=5, color=colors, width=0.65)
    ax.set(ylabel="Weighted F1", ylim=(0, 1.02),
           title="f  Framework–model crossover comparison (mean ± SD, n=3)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.018, f"{value:.3f}", ha="center")
    save(fig, "figure3_panel_f_crossover")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric_rows = benchmark_metrics()
    output_rows, field_rows = output_metrics()
    write_csv(OUTPUT / "benchmark_metrics.csv", metric_rows)
    write_csv(OUTPUT / "output_consistency_and_unknown_rate.csv", output_rows)
    write_csv(OUTPUT / "unknown_rate_by_field.csv", field_rows)
    plot_panel_a(metric_rows)
    plot_panel_b(metric_rows)
    plot_panel_c(output_rows)
    plot_panel_d(output_rows)
    plot_panel_f(metric_rows)
    provenance = {
        "benchmark": "SDRF benchmark",
        "semantic_classification_threshold": 0.70,
        "effective_final_metric_threshold": 0.50,
        "models": MODELS,
        "arms": ARMS,
        "replicates": [{"label": label, "index": str(index), "adapted_outputs": str(adapted)}
                       for label, index, adapted in REPLICATES],
        "panel_e_status": "not generated: S5 leave-one-out inference arms are unavailable",
        "consistency_definition": "fraction of PXD-agent-field units with identical normalized values in all three runs, excluding units unknown in all runs",
        "unknown_definition": "missing, empty, null/none, or literal unknown across the frozen union schema",
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "figures": 5, "panel_e": provenance["panel_e_status"]}, indent=2))


if __name__ == "__main__":
    main()
