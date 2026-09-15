"""Error analysis for the cumulative ablation: which entities are easy, which
are hard, and when a value is wrong, how wrong is it (Lin IC).

Inputs: manuscript_figure2/scored_pairs_all_replicates.csv (every pair, 3
runs x 3 models x S0-S5) and manuscript_figure2/lin_pairs_all_replicates.csv
(Lin IC on the ontology-backed fields). Outcome per pair, using the
benchmark's own acceptance:
  accepted   LLM value present and score >= 0.5 (counts as TP in F1)
  wrong      LLM value present and score < 0.5
  missing    LLM gave unknown / nothing

    python final_analyses/SapBERTStaircase/scripts/build_error_analysis.py

Writes manuscript_figure2/error_analysis/{E1..E4}.png and the tables behind them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
FIG = ROOT / "manuscript_figure2"
OUT = FIG / "error_analysis"
sys.path.insert(0, str(REPO))

ARMS = [f"S{i}" for i in range(6)]
FINAL_ARM = "S5"
AGENT_OF = {}
FIELD_ORDER_AGENT = {"BiologicalAgent": 0, "TechnicalAgent": 1, "ExperimentalDesignAgent": 2}


def outcome(df):
    present = df["llm"].notna()
    return np.select([~present, df["score"] >= 0.5], ["missing", "accepted"], "wrong")


def main():
    OUT.mkdir(exist_ok=True)
    pairs = pd.read_csv(FIG / "scored_pairs_all_replicates.csv")
    pairs["outcome"] = outcome(pairs)
    lin = pd.read_csv(FIG / "lin_pairs_all_replicates.csv")
    lin["outcome"] = outcome(lin)
    for f, a in pairs.groupby("field").agent.first().items():
        AGENT_OF[f] = a

    # ---- E1: acceptance per field x arm, mean over models and runs ----------
    e1 = (pairs.assign(acc=pairs.outcome == "accepted")
          .groupby(["field", "arm"]).acc.mean().unstack("arm").reindex(columns=ARMS))
    e1["agent"] = e1.index.map(AGENT_OF)
    e1 = e1.sort_values(["agent", FINAL_ARM], ascending=[True, False])
    e1.to_csv(OUT / "E1_acceptance_by_field_and_arm.csv")

    # ---- E2: outcome composition per field at the final arm -----------------
    final = pairs[pairs.arm == FINAL_ARM]
    e2 = (final.groupby(["field", "outcome"]).size().unstack("outcome", fill_value=0))
    e2 = e2.div(e2.sum(axis=1), axis=0)
    e2["n_pairs"] = final.groupby("field").size()
    e2["agent"] = e2.index.map(AGENT_OF)
    e2 = e2.sort_values("accepted", ascending=False)
    e2.to_csv(OUT / "E2_outcome_by_field_final_arm.csv")

    # ---- E3: how wrong are the mistakes (Lin on ontology fields, final arm) --
    lf = lin[(lin.arm == FINAL_ARM) & lin.lin.notna()].copy()
    lf["kind"] = np.select(
        [lf.outcome == "missing", lf.lin >= 0.99, lf.lin >= 0.5, lf.lin > 0.01],
        ["missing", "same term", "related (Lin 0.5-0.99)", "distant (Lin 0.01-0.5)"], "unrelated (Lin <= 0.01)")
    e3 = lf.groupby(["field", "kind"]).size().unstack("kind", fill_value=0)
    e3 = e3.div(e3.sum(axis=1), axis=0)
    e3["mean_lin_of_mistakes"] = lf[lf.outcome != "accepted"].groupby("field").lin.mean()
    e3["mean_lin_of_wrong_values"] = lf[lf.outcome == "wrong"].groupby("field").lin.mean()
    e3 = e3.sort_values("same term", ascending=False)
    e3.to_csv(OUT / "E3_mistake_severity_by_field.csv")
    # accepted-but-not-the-same-term: what F1 counts as right that the ontology does not
    acc_not_same = lf[(lf.outcome == "accepted") & (lf.lin < 0.99)].groupby("field").agg(
        n=("lin", "size"), share_of_accepted=("lin", lambda s: len(s)), mean_lin=("lin", "mean"))
    acc_not_same["share_of_accepted"] = acc_not_same.n / lf[lf.outcome == "accepted"].groupby("field").size()
    acc_not_same.to_csv(OUT / "E3b_accepted_but_not_same_term.csv")

    # ---- E4: most frequent confusions at the final arm -----------------------
    wrong = final[final.outcome == "wrong"].copy()
    wrong["golden_s"] = wrong.golden.astype(str).str.lower().str.strip().str.slice(0, 45)
    wrong["llm_s"] = wrong.llm.astype(str).str.lower().str.strip().str.slice(0, 45)
    conf = (wrong.groupby(["field", "golden_s", "llm_s"]).size().rename("n").reset_index()
            .sort_values("n", ascending=False))
    lin_lookup = lf[lf.outcome == "wrong"].assign(
        golden_s=lambda d: d.golden.astype(str).str.lower().str.strip().str.slice(0, 45),
        llm_s=lambda d: d.llm.astype(str).str.lower().str.strip().str.slice(0, 45)
    ).groupby(["field", "golden_s", "llm_s"]).lin.mean()
    conf["lin"] = [lin_lookup.get((f, g, l), np.nan) for f, g, l in zip(conf.field, conf.golden_s, conf.llm_s)]
    conf.to_csv(OUT / "E4_top_confusions_final_arm.csv", index=False)

    draw(e1, e2, e3, lf, conf)


def draw(e1, e2, e3, lf, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import COLORS, clean_axes

    # E1 heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    m = e1[ARMS].to_numpy(dtype=float)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
    ax.set_yticks(range(len(e1))); ax.set_yticklabels([f"{f}  ({a[:3].lower()})" for f, a in zip(e1.index, e1.agent)], fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] > 0.6 else "black")
    # separators between agents
    agents = list(e1.agent)
    for i in range(1, len(agents)):
        if agents[i] != agents[i - 1]:
            ax.axhline(i - 0.5, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="share of pairs accepted (score ≥ 0.5)")
    ax.set_title("E1  Acceptance per field along the staircase (mean of 3 models × 3 runs)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "E1_acceptance_heatmap.png", dpi=220); plt.close(fig)

    # E2 stacked outcome bars
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(e2))
    left = np.zeros(len(e2))
    for col, color, lab in [("accepted", COLORS["green"], "accepted (TP)"), ("wrong", COLORS["red"], "wrong value"),
                            ("missing", "#bbbbbb", "unknown / missing")]:
        v = e2[col].to_numpy() if col in e2 else np.zeros(len(e2))
        ax.barh(y, v, left=left, color=color, label=lab, height=0.7)
        left += v
    ax.set_yticks(y); ax.set_yticklabels([f"{f}  (n={n}, {a[:3].lower()})" for f, n, a in zip(e2.index, e2.n_pairs, e2.agent)], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel(f"share of golden values, {FINAL_ARM}, 3 models × 3 runs")
    ax.legend(frameon=False, fontsize=8, loc="lower right"); clean_axes(ax, grid_axis="x")
    ax.set_title("E2  Best to hardest entities: outcome per field at S5", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "E2_outcome_by_field.png", dpi=220); plt.close(fig)

    # E3 mistake severity: stacked kinds per ontology field + Lin strip of mistakes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    kinds = ["same term", "related (Lin 0.5-0.99)", "distant (Lin 0.01-0.5)", "unrelated (Lin <= 0.01)", "missing"]
    colors = [COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["red"], "#bbbbbb"]
    y = np.arange(len(e3)); left = np.zeros(len(e3))
    for k, c in zip(kinds, colors):
        v = e3[k].to_numpy() if k in e3 else np.zeros(len(e3))
        ax.barh(y, v, left=left, color=c, label=k, height=0.7); left += v
    ax.set_yticks(y); ax.set_yticklabels(e3.index, fontsize=9); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("share of golden values (ontology-resolved), S5"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    clean_axes(ax, grid_axis="x"); ax.set_title("E3  How wrong is wrong: ontology distance of every answer", fontsize=10, loc="left")
    ax = axes[1]
    mistakes = lf[lf.outcome == "wrong"]
    order = list(e3.index)
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        v = mistakes[mistakes.field == f].lin.to_numpy()
        if len(v):
            ax.scatter(v, np.full(len(v), i) + rng.uniform(-.25, .25, len(v)), s=10, alpha=.5, color=COLORS["purple"])
            ax.plot([np.median(v)] * 2, [i - .35, i + .35], color="k", lw=1.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(-.02, 1.02); ax.set_xlabel("Lin IC of values the matcher rejected (bar = median)")
    ax.axvline(0.5, ls=":", color="gray", lw=.8); clean_axes(ax, grid_axis="x")
    ax.set_title("rejected values only", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "E3_mistake_severity.png", dpi=220); plt.close(fig)

    # E4 top confusions as a table figure
    top = conf.head(22)
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(top) + 1.2)); ax.axis("off")
    cells = [[r.field, r.golden_s, r.llm_s, str(r.n), "" if pd.isna(r.lin) else f"{r.lin:.2f}"] for r in top.itertuples()]
    tbl = ax.table(cellText=cells, colLabels=["field", "SDRF golden", "HAMLET value", "n (of 9 runs)", "Lin IC"],
                   loc="center", cellLoc="left", colWidths=[0.14, 0.36, 0.36, 0.08, 0.06])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.25)
    ax.set_title(f"E4  Most frequent rejected (golden, HAMLET) pairs at {FINAL_ARM}, pooled over models and runs", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "E4_top_confusions.png", dpi=220); plt.close(fig)
    print(f"wrote E1-E4 to {OUT}")


if __name__ == "__main__":
    main()
