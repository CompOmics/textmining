#!/usr/bin/env python3
"""
F1 scores for all 6 benchmarked models.

"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODELS       = ["llama", "gpt", "claude", "gemini", "gemma", "qwen"]
MODEL_LABELS = ["Llama-4\nScout", "GPT-5.4", "Claude\nOpus 4.5", "Gemini\n3.5 Flash", "Gemma 4\n31B", "Qwen\n3.7 Max"]
AGENT_LABELS = ["Biological", "Technical", "Experimental\nDesign", "Overall"]
AGENTS_SRC   = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]
BASE         = Path(__file__).resolve().parents[2] / "six_model_benchmark"
OUT          = Path(__file__).resolve().parent
NULL_VALS    = {"", "unknown", "not applicable", "n/a", "none", "nan"}
THRESHOLD    = 0.5
COLORS       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def prf(df):
    golden_has = df["golden"].notna() & ~df["golden"].astype(str).str.strip().str.lower().isin(NULL_VALS)
    llm_has    = df["llm"].notna()    & ~df["llm"].astype(str).str.strip().str.lower().isin(NULL_VALS)
    both = llm_has & golden_has
    tp = (both & (df["score"] >= THRESHOLD)).sum()
    fp = (both & (df["score"] < THRESHOLD)).sum()
    fn = fp + (~llm_has & golden_has).sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


f1_data = {ag: [] for ag in AGENT_LABELS}
for model in MODELS:
    df = pd.read_csv(BASE / f"reports_test_v2_{model}_nopride" / "sdrf_benchmark_detailed.csv")
    for ag_src, ag_lbl in zip(AGENTS_SRC, AGENT_LABELS[:3]):
        _, _, f1 = prf(df[df["agent"] == ag_src])
        f1_data[ag_lbl].append(f1)
    _, _, f1 = prf(df)
    f1_data["Overall"].append(f1)
pd.DataFrame(f1_data, index=[l.replace("\n", " ") for l in MODEL_LABELS]).round(3).to_csv(OUT / "f1_all_models.csv")
print(pd.DataFrame(f1_data, index=[l.replace("\n", " ") for l in MODEL_LABELS]).round(3).to_string())

n_agents = len(AGENT_LABELS)
n_models = len(MODELS)
x        = np.arange(n_agents)
w        = 0.12
offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * w

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.suptitle("Benchmark F1 scores — all models, 30-PXD test set",
             fontsize=13, fontweight="bold")

for i, (label, color) in enumerate(zip(MODEL_LABELS, COLORS)):
    f1s = [f1_data[ag][i] for ag in AGENT_LABELS]
    ax.bar(x + offsets[i], f1s, w, label=label, color=color, alpha=0.88,
           edgecolor="white", lw=0.4)

ax.set_xticks(x)
ax.set_xticklabels(AGENT_LABELS, fontsize=11)
ax.set_ylabel("F1 Score", fontsize=11)
ymin = min(min(v) for v in f1_data.values())
ax.set_ylim(max(0.0, np.floor((ymin - 0.05) * 10) / 10), 1.02)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.legend(fontsize=9, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.18), frameon=True)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, ls="--")
ax.set_axisbelow(True)
ax.axvline(2.5, color="grey", lw=0.8, ls=":", alpha=0.6)

plt.tight_layout()
out = OUT / "plots_all_models_f1.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
