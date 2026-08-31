#!/usr/bin/env python3
"""Aggregate SDRF benchmark reports across inference replicates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import math
from collections import defaultdict
from pathlib import Path


# Two-sided 95% Student-t critical value for the three-replicate design (df=2).
T_975_DF2 = 4.302652729911275

MODEL_LABELS = {
    "qwen3_8_27b": "Qwen3.8-27B",
    "gemma4_31b": "Gemma 4 31B",
    "glm4_7": "GLM-4.7",
}
MODEL_ORDER = ["qwen3_8_27b", "gemma4_31b", "glm4_7"]
ARM_ORDER = [f"S{i}" for i in range(6)]


def plot_f1(rows: list[dict], output: Path) -> None:
    """Plot benchmark F1 across staircase arms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(row["model"], row["arm"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"qwen3_8_27b": "#0072B2", "gemma4_31b": "#D55E00", "glm4_7": "#009E73"}
    markers = {"qwen3_8_27b": "o", "gemma4_31b": "s", "glm4_7": "^"}
    x = list(range(len(ARM_ORDER)))

    for model in MODEL_ORDER:
        means = [by_key[(model, arm)]["f1_mean"] for arm in ARM_ORDER]
        sds = [by_key[(model, arm)]["f1_sd"] for arm in ARM_ORDER]
        ax.errorbar(
            x, means, yerr=sds, marker=markers[model], markersize=6,
            linewidth=2, capsize=4, color=colors[model],
            label=MODEL_LABELS[model],
        )

    ax.set_xticks(x, ARM_ORDER)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Staircase arm")
    ax.set_ylabel("Weighted F1")
    ax.set_title("SDRF benchmark (mean ± SD, n=3)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "staircase_f1.png", dpi=300)
    fig.savefig(output / "staircase_f1.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate", action="append", required=True,
                        help="replicate_label=/absolute/path/to/index.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.replicate) < 2:
        raise SystemExit("at least two replicate indices are required")
    if len(args.replicate) != 3:
        raise SystemExit("this confidence-interval report requires exactly three replicates")

    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sources = []
    expected_keys: set[tuple[str, str]] | None = None
    for spec in args.replicate:
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path).resolve()
        data = json.loads(path.read_text())
        runs = {(run["model"], run["arm"]): run for run in data["runs"]}
        keys = set(runs)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise RuntimeError(f"replicate {label} key mismatch; missing={missing}, extra={extra}")
        sources.append({"label": label, "index": str(path)})
        for key, run in runs.items():
            overall = run["overall"]
            for metric in ("precision", "recall", "f1"):
                values[key][metric].append(float(overall[metric]))

    rows = []
    for model, arm in sorted(expected_keys or set()):
        row = {"model": model, "arm": arm, "n": len(sources)}
        for metric in ("precision", "recall", "f1"):
            observed = values[(model, arm)][metric]
            row[f"{metric}_mean"] = statistics.mean(observed)
            row[f"{metric}_sd"] = statistics.stdev(observed)
            half_width = T_975_DF2 * row[f"{metric}_sd"] / math.sqrt(len(observed))
            row[f"{metric}_ci95_low"] = row[f"{metric}_mean"] - half_width
            row[f"{metric}_ci95_high"] = row[f"{metric}_mean"] + half_width
            row[f"{metric}_values"] = observed
        rows.append(row)

    result = {
        "method": "mean, sample standard deviation, and two-sided 95% Student-t confidence interval across independent full inference runs",
        "ddof": 1,
        "n_replicates": len(sources),
        "confidence_interval": {
            "level": 0.95,
            "method": "Student-t interval for the mean",
            "degrees_of_freedom": 2,
            "critical_value": T_975_DF2,
        },
        "matcher_behavior": {
            "semantic_threshold": 0.70,
            "effective_final_metric_threshold": 0.50,
            "source_modified": False,
        },
        "replicates": sources,
        "results": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as handle:
        fieldnames = [
            "model", "arm", "n",
            "precision_mean", "precision_sd", "precision_ci95_low", "precision_ci95_high",
            "recall_mean", "recall_sd", "recall_ci95_low", "recall_ci95_high",
            "f1_mean", "f1_sd", "f1_ci95_low", "f1_ci95_high", "f1_values",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["f1_values"] = ";".join(f"{v:.4f}" for v in row["f1_values"])
            writer.writerow(rendered)
    plot_f1(rows, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
