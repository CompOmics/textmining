#!/usr/bin/env python3
"""
Plot the impact of semantic matching on benchmark metrics.

Compares "Before" (only EXACT + NORMALIZED matches) vs "After" (full semantic
matching including ONTOLOGY, HIERARCHICAL, SEMANTIC) for each model and agent.

Output format matches overall_agent_metrics.png style.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).parent

MODELS = {
    "Llama-4-Scout": BENCHMARK_DIR / "reports",
    "GPT 5.2":       BENCHMARK_DIR / "reports_gpt",
    "Gemini 2.5 Pro": BENCHMARK_DIR / "reports_gemini",
}

AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]
AGENT_SHORT = {
    "BiologicalAgent": "Biological",
    "TechnicalAgent": "Technical",
    "ExperimentalDesignAgent": "Exp. Design",
}

PLAIN_MATCH_TYPES = {"EXACT", "NORMALIZED"}
MATCH_THRESHOLD = 0.5

OUT = BENCHMARK_DIR / "reports_comparison"


# ─── Metric Calculation ────────────────────────────────────────────
def compute_prf(df: pd.DataFrame, use_semantic: bool) -> dict:
    tp = fp = fn = 0
    for _, row in df.iterrows():
        match_type = row["match_type"]
        score = row["score"]
        llm_val = row.get("llm")
        golden_val = row.get("golden")
        llm_has = llm_val is not None and str(llm_val).strip() != "" and str(llm_val).lower() not in ("none", "nan")
        golden_has = golden_val is not None and str(golden_val).strip() != "" and str(golden_val).lower() not in ("none", "nan")
        if not use_semantic and match_type not in PLAIN_MATCH_TYPES:
            score = 0.0
        is_match = score >= MATCH_THRESHOLD
        if llm_has and golden_has:
            if is_match:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif not llm_has and golden_has:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def load_all_metrics():
    data = {}
    for model_name, reports_dir in MODELS.items():
        csv_path = reports_dir / "sdrf_benchmark_detailed.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping {model_name}")
            continue
        df = pd.read_csv(csv_path)
        data[model_name] = {}
        for agent in AGENTS:
            agent_df = df[df["agent"] == agent]
            if agent_df.empty:
                continue
            data[model_name][agent] = {
                "before": compute_prf(agent_df, use_semantic=False),
                "after":  compute_prf(agent_df, use_semantic=True),
            }
        if data[model_name]:
            n = len(data[model_name])
            for mode in ("before", "after"):
                data[model_name].setdefault("Overall", {})
                data[model_name]["Overall"][mode] = {
                    k: sum(data[model_name][a][mode][k] for a in data[model_name] if a != "Overall") / n
                    for k in ("precision", "recall", "f1")
                }
    return data


# ─── Plot 1: Per-model Before/After (matching overall_agent_metrics.png) ──
def plot_per_model(data: dict):
    """One plot per model: Before and After side by side, same format as reference."""
    labels = ["Biological", "Technical", "Exp. Design", "Overall"]
    agent_keys = AGENTS + ["Overall"]

    for model_name in data:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        fig.suptitle(f"SDRF Benchmark — {model_name}  |  107 PXDs\n"
                     f"Before vs After Semantic Matching",
                     fontsize=14, fontweight='bold')

        for col, (mode, title_suffix) in enumerate([
            ("before", "Before (Exact + Normalized only)"),
            ("after",  "After (Full Semantic Matching)"),
        ]):
            ax = axes[col]
            p_vals, r_vals, f1_vals = [], [], []
            for ak in agent_keys:
                m = data[model_name].get(ak, {}).get(mode, {})
                p_vals.append(m.get("precision", 0))
                r_vals.append(m.get("recall", 0))
                f1_vals.append(m.get("f1", 0))

            x = np.arange(len(labels))
            w = 0.25

            b1 = ax.bar(x - w, p_vals,  w, label='Precision', color='#3498db', alpha=0.85, edgecolor='white')
            b2 = ax.bar(x,     r_vals,  w, label='Recall',    color='#2ecc71', alpha=0.85, edgecolor='white')
            b3 = ax.bar(x + w, f1_vals, w, label='F1 Score',  color='#9b59b6', alpha=0.85, edgecolor='white')

            for bars in [b1, b2, b3]:
                for bar in bars:
                    h = bar.get_height()
                    fw = 'bold' if bars is b3 else 'normal'
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                            f'{h:.3f}', ha='center', va='bottom', fontsize=10,
                            fontweight=fw, color='#2c3e50')

            ax.set_ylim(0, 1.15)
            ax.set_ylabel('Score', fontsize=13)
            ax.set_title(title_suffix, fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=12)
            if col == 0:
                ax.legend(fontsize=10, loc='upper left')
            ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)

        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
        path = OUT / f"semantic_impact_{safe_name}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')


# ─── Plot 2: Cross-model F1 comparison (Before vs After) ──────────
def plot_cross_model_f1(data: dict):
    """All 3 models on one chart, Before/After F1 per agent + Overall."""
    labels = ["Biological", "Technical", "Exp. Design", "Overall"]
    agent_keys = AGENTS + ["Overall"]
    models = [m for m in MODELS if m in data]

    fig, ax = plt.subplots(figsize=(14, 7))

    n_models = len(models)
    x = np.arange(len(labels))
    total_bars_per_group = n_models * 2  # before + after for each model
    w = 0.10
    gap = 0.02

    model_colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for mi, model in enumerate(models):
        before_vals, after_vals = [], []
        for ak in agent_keys:
            m = data[model].get(ak, {})
            before_vals.append(m.get("before", {}).get("f1", 0))
            after_vals.append(m.get("after", {}).get("f1", 0))

        offset_b = (mi * 2) * (w + gap) - (total_bars_per_group * (w + gap)) / 2 + w / 2
        offset_a = offset_b + w + gap

        color_before = model_colors[mi]
        color_after = model_colors[mi]

        bars_b = ax.bar(x + offset_b, before_vals, w, color=color_before, alpha=0.4,
                        edgecolor='white', linewidth=0.5, hatch='///')
        bars_a = ax.bar(x + offset_a, after_vals, w, color=color_after, alpha=0.85,
                        edgecolor='white', linewidth=0.5)

        # Labels on after bars only
        for bar, val in zip(bars_a, after_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold', color='#2c3e50')

    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = []
    for mi, model in enumerate(models):
        legend_handles.append(Patch(facecolor=model_colors[mi], alpha=0.4, hatch='///', label=f'{model} (Before)'))
        legend_handles.append(Patch(facecolor=model_colors[mi], alpha=0.85, label=f'{model} (After)'))
    ax.legend(handles=legend_handles, fontsize=8.5, loc='upper left', ncol=2)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('Semantic Matching Impact on F1 — All Models\n'
                 '107 PXDs · Before (Exact+Normalized) vs After (Full Semantic)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)

    plt.tight_layout()
    path = OUT / "semantic_impact_cross_model_f1.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─── Plot 3: Delta improvement chart ─────────────────────────────
def plot_delta(data: dict):
    """F1 improvement (After − Before) per agent per model."""
    labels = ["Biological", "Technical", "Exp. Design", "Overall"]
    agent_keys = AGENTS + ["Overall"]
    models = [m for m in MODELS if m in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    n = len(models)
    w = 0.22
    model_colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for mi, model in enumerate(models):
        deltas = []
        for ak in agent_keys:
            m = data[model].get(ak, {})
            delta = m.get("after", {}).get("f1", 0) - m.get("before", {}).get("f1", 0)
            deltas.append(delta)

        offset = mi * w - (n * w) / 2 + w / 2
        bars = ax.bar(x + offset, deltas, w, label=model,
                      color=model_colors[mi], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f'+{val:.3f}' if val >= 0 else f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('ΔF1 (After − Before)', fontsize=13)
    ax.set_title('F1 Improvement from Semantic Matching\n107 PXDs · Per Agent & Overall',
                 fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
    ax.axvline(x=2.5, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    path = OUT / "semantic_impact_delta_f1.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─── Summary Table ─────────────────────────────────────────────────
def print_summary(data: dict):
    print("\n" + "=" * 90)
    print("  SEMANTIC MATCHING IMPACT SUMMARY")
    print("=" * 90)
    header = f"{'Model':<18} {'Agent':<16} {'Before F1':>10} {'After F1':>10} {'ΔF1':>8} {'Δ%':>7}"
    print(header)
    print("-" * 90)
    for model in data:
        for ak in AGENTS + ["Overall"]:
            if ak not in data[model]:
                continue
            b = data[model][ak]["before"]["f1"]
            a = data[model][ak]["after"]["f1"]
            d = a - b
            pct = (d / b * 100) if b > 0 else 0
            label = AGENT_SHORT.get(ak, ak)
            if ak == "Overall":
                label = "** Overall **"
            print(f"{model:<18} {label:<16} {b:>10.4f} {a:>10.4f} {d:>+8.4f} {pct:>+6.1f}%")
        print("-" * 90)


# ─── Main ──────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading benchmark data...")
    data = load_all_metrics()
    if not data:
        print("ERROR: No data found.")
        sys.exit(1)
    print(f"  Loaded {len(data)} models: {list(data.keys())}")

    print_summary(data)

    plot_per_model(data)
    plot_cross_model_f1(data)
    plot_delta(data)

    print(f"\n✅ All plots saved to: {OUT}")


if __name__ == "__main__":
    main()
