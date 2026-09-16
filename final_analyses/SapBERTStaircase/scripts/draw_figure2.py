"""Draw Figure 2 (main) and Figure S2 (entity level) from the tables built by
rescore_replicates.py, build_figure2.py and build_error_analysis.py.

Main figure
  a  overall F1 per arm                     b  overall Lin IC per arm
  c  F1 per agent (mean over models,        d  Lin IC per agent (same style)
     band = min-max over models)
  e  how values are accepted at S5, per agent: exact/normalised, ontology or
     hierarchy, SapBERT tier, accepted below the tier by cosine >= 0.5,
     rejected, missing (the SapBERT shift)
  f  non-unknown fields per dataset          g  three-run unanimity

Supplementary figure (entity level)
  a  acceptance per field along the staircase (heatmap)
  b  outcome per field at S5 with the human annotators' own disagreement on
     the same label overlaid (final_analyses/HumanAnnotation)
  c  ontology distance of every answer per ontology-backed field (Lin)
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
EA = FIG / "error_analysis"
HUMAN = REPO / "final_analyses" / "HumanAnnotation" / "results"
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / "final_analyses"))
from figure_style import (metric_group_legend, AGENT as AGENT_COLORS, AGENT_LABEL, AGENTS, MODEL as MODEL_COLORS, MODELS, TIER as TIER_COLORS,
                          OUTCOME, LIN_KIND, HEAT_CMAP, style, panel_title, legend_outside, save_composite, save_panels)

ARMS = [f"S{i}" for i in range(6)]
FINAL = "S5"
FIELD_TO_HUMAN = {"species": "Organism", "organ": "OrganismPart", "cell_type": "CellType", "cell_line": "CellLine",
                  "disease": "Disease", "strain": "Strain", "instrument": "Instrument", "cleavage_agent": "CleavageAgent",
                  "label": "Label", "reduction_reagent": "ReductionReagent", "collision_energy": "CollisionEnergy",
                  "replicates": "NumberOfBiologicalReplicates", "technical_replicates": "NumberOfTechnicalReplicates",
                  "number_of_samples": "NumberOfSamples", "fractions": "NumberOfFractions", "factor_value": "FactorValue"}


def trend_by_model(ax, data, col_mean, col_sd):
    x = np.arange(len(ARMS))
    for model in MODELS:
        sub = data[data.model == model].set_index("arm").reindex(ARMS)
        y, e = sub[col_mean].to_numpy(float), sub[col_sd].to_numpy(float)
        ax.fill_between(x, y - e, y + e, color=MODEL_COLORS[model], alpha=.18, lw=0)
        ax.plot(x, y, "-o", color=MODEL_COLORS[model], ms=4, lw=1.8, label=model)
    ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 1)


def per_replicate_tables():
    """Per replicate x model x arm: F1 overall and per agent (from the
    benchmark tables) and Lin overall and per agent (from the graded pairs;
    a missing LLM value counts 0, pairs whose golden does not resolve are
    left out)."""
    f1 = pd.read_csv(FIG / "benchmark_metrics_per_replicate.csv")
    lin = pd.read_csv(FIG / "lin_pairs_all_replicates.csv")
    lin = lin[lin.lin.notna()]
    g = lin.groupby(["model", "arm", "replicate"])
    lt = g.lin.mean().rename("lin_overall").to_frame()
    for agent in AGENTS:
        sub = lin[lin.agent == agent]
        if len(sub):
            lt[f"lin_{agent}"] = sub.groupby(["model", "arm", "replicate"]).lin.mean()
    return f1.merge(lt.reset_index(), on=["model", "arm", "replicate"], how="left")


def trend_with_runs(ax, rep, series, group_col, groups, colors, labels, ls, ylabel):
    """series: list of (column, linestyle, legend suffix). For every group
    (model or agent) mean over runs, band = SD over the three runs (for
    agent panels, the run value is first averaged over models)."""
    x = np.arange(len(ARMS))
    for grp in groups:
        for col, style_, suffix in series:
            if col.format(g=grp) not in rep:
                continue
            c = col.format(g=grp)
            if group_col == "model":
                d = rep[rep.model == grp]
                per_run = d.groupby(["arm", "replicate"])[c].mean()
            else:
                per_run = rep.groupby(["arm", "replicate"])[c].mean()   # mean over models within a run
            per_run = per_run.unstack("replicate").reindex(ARMS)
            m, sd = per_run.mean(axis=1).to_numpy(), per_run.std(axis=1, ddof=1).to_numpy()
            ax.fill_between(x, m - sd, m + sd, color=colors[grp], alpha=.15, lw=0)
            ax.plot(x, m, style_, color=colors[grp], ms=4, lw=1.8)
    ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 1); ax.set_ylabel(ylabel)


def acceptance_decomposition(pairs):
    """Share of golden values per agent at the final arm by how they were
    accepted. Columns ordered from strict to lenient."""
    p = pairs[pairs.arm == FINAL].copy()
    present = p.llm.notna()
    p["how"] = np.select(
        [~present, p.match_type.isin(["EXACT", "NORMALIZED"]), p.match_type.isin(["ONTOLOGY", "HIERARCHICAL"]),
         p.match_type.eq("SEMANTIC"), (p.match_type.eq("NO_MATCH")) & (p.score >= 0.5)],
        ["missing", "exact / normalised", "ontology / hierarchy", "SapBERT tier (cosine ≥ 0.70)", "below tier, cosine ≥ 0.50"],
        "rejected")
    out = p.groupby(["agent", "how"]).size().unstack("how", fill_value=0)
    out = out.reindex(columns=list(TIER_COLORS), fill_value=0)
    return out.div(out.sum(axis=1), axis=0).reindex(AGENTS)


def main():
    plt = style()
    data = pd.read_csv(FIG / "figure2_data.csv")
    pairs = pd.read_csv(FIG / "scored_pairs_all_replicates.csv")
    e1 = pd.read_csv(EA / "E1_acceptance_by_field_and_arm.csv", index_col=0)
    e2 = pd.read_csv(EA / "E2_outcome_by_field_final_arm.csv", index_col=0)
    e3 = pd.read_csv(EA / "E3_mistake_severity_by_field.csv", index_col=0)
    human = pd.read_csv(HUMAN / "value_match_by_label.csv").set_index("label")
    hk = pd.read_csv(HUMAN / "label_agreement_presence.csv")
    hk = hk[hk.pair_category.str.contains("human", case=False) & ~hk.pair_category.str.contains("Harmonized|GPT|model", case=False)]
    hk = hk.groupby("label").kappa.mean()

    # ------------------------------------------------------------ main figure
    dec = acceptance_decomposition(pairs)

    rep = per_replicate_tables()
    F1_LIN = [("{g}", "-o", ", F1"), ("lin_{g}", "--s", ", Lin IC")]

    def p_a(ax):
        rep_m = rep.rename(columns={"overall_f1": "F1"})
        trend_with_runs(ax, rep_m.assign(lin_F1=rep_m["lin_overall"]),
                        [("F1", "-o", "  F1"), ("lin_F1", "--s", "  Lin IC")], "model", MODELS, MODEL_COLORS,
                        {m: m for m in MODELS}, None, "weighted F1  /  Lin IC")
        metric_group_legend(ax, MODEL_COLORS, fontsize=8)
        panel_title(ax, "a", "Overall, mean ± SD over 3 runs")

    def p_b(ax):
        cols = {f"{ag}_f1": ag for ag in AGENTS}
        rep_a = rep.rename(columns=cols)
        trend_with_runs(ax, rep_a, [("{g}", "-o", "  F1"), ("lin_{g}", "--s", "  Lin IC")], "agent", AGENTS, AGENT_COLORS,
                        AGENT_LABEL, None, "weighted F1  /  Lin IC")
        metric_group_legend(ax, {AGENT_LABEL[a]: AGENT_COLORS[a] for a in AGENTS}, fontsize=8)
        panel_title(ax, "b", "Per agent, mean over models ± SD over 3 runs")

    def p_e(ax):
        left = np.zeros(len(dec))
        for how, color in TIER_COLORS.items():
            v = dec[how].to_numpy()
            ax.barh(np.arange(len(dec)), v, left=left, color=color, label=how, height=0.62, edgecolor="white", lw=.5)
            for i, (l, w) in enumerate(zip(left, v)):
                if w >= 0.06:
                    ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7.5,
                            color="white" if how in ("exact / normalised", "rejected") else "black")
            left += v
        ax.set_yticks(range(len(dec))); ax.set_yticklabels([AGENT_LABEL[a] for a in dec.index]); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_xlabel(f"share of values at {FINAL}"); ax.grid(False)
        legend_outside(ax, "right", fontsize=7.5)
        panel_title(ax, "c", "How values are accepted at S5")

    def p_f(ax):
        trend_by_model(ax, data, "non_unknown_fields_mean", "non_unknown_fields_sd_across_replicates")
        ax.set_ylim(0, data["non_unknown_fields_mean"].max() * 1.15); panel_title(ax, "d", "Output volume, mean ± SD over 3 runs")
        ax.set_ylabel("values per dataset"); legend_outside(ax, "right")

    def p_g(ax):
        x = np.arange(len(ARMS))
        for model in MODELS:
            sub = data[data.model == model].set_index("arm").reindex(ARMS)
            ax.plot(x, sub["unanimity_all_slots"], "-o", color=MODEL_COLORS[model], ms=4, lw=1.8, label=model)
        ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 1)
        panel_title(ax, "e", "Three-run unanimity"); ax.set_ylabel("identical values")

    fig = plt.figure(figsize=(13, 10.5))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.75, height_ratios=[1, 0.85, 1])
    p_a(fig.add_subplot(gs[0, 0])); p_b(fig.add_subplot(gs[0, 1]))
    p_e(fig.add_subplot(gs[1, :]))
    p_f(fig.add_subplot(gs[2, 0])); p_g(fig.add_subplot(gs[2, 1]))
    save_composite(fig, FIG, "figure2"); plt.close(fig)
    save_panels(plt, FIG, "figure2", {"a": p_a, "b": p_b, "c": p_e, "d": p_f, "e": p_g}, {"c": (11, 3.2)})

    # --------------------------------------------------- supplementary figure
    e2s = e2.sort_values(["agent", "accepted"], ascending=[True, False])
    e3s = e3.sort_values("same term", ascending=False)

    def s_a(ax):
        m = e1[ARMS].to_numpy(float)
        im = ax.imshow(m, cmap=HEAT_CMAP, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS)
        ax.set_yticks(range(len(e1))); ax.set_yticklabels(e1.index, fontsize=8)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] > 0.6 else "black")
        agents = list(e1.agent)
        for i in range(1, len(agents)):
            if agents[i] != agents[i - 1]:
                ax.axhline(i - 0.5, color="k", lw=0.8)
        for agent in AGENTS:
            idx = [i for i, a in enumerate(agents) if a == agent]
            if idx:
                ax.text(-0.55, (min(idx) + max(idx)) / 2, AGENT_LABEL[agent], transform=ax.get_yaxis_transform(),
                        ha="right", va="center", rotation=90, fontsize=8, color=AGENT_COLORS[agent])
        ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="share of values accepted")
        panel_title(ax, "a", "Acceptance per entity along the staircase")

    def s_b(ax):
        y = np.arange(len(e2s)); left = np.zeros(len(e2s))
        for col, color, lab in [("accepted", OUTCOME["accepted"], "accepted"), ("wrong", OUTCOME["wrong"], "wrong value"), ("missing", OUTCOME["missing"], "unknown")]:
            v = e2s[col].to_numpy() if col in e2s else np.zeros(len(e2s))
            ax.barh(y, v, left=left, color=color, label=lab, height=0.6, edgecolor="white", lw=.5); left += v
        hx = [1 - human.loc[FIELD_TO_HUMAN[f], "no_match_pct"] / 100 if f in FIELD_TO_HUMAN and FIELD_TO_HUMAN[f] in human.index else np.nan for f in e2s.index]
        ax.scatter(hx, y, marker="D", s=28, color="k", zorder=5, label="human annotators: share of values in agreement")
        ax.set_yticks(y); ax.set_yticklabels([f"{f}  (n={n})" for f, n in zip(e2s.index, e2s.n_pairs)], fontsize=8); ax.invert_yaxis()
        agents = list(e2s.agent)
        for i in range(1, len(agents)):
            if agents[i] != agents[i - 1]:
                ax.axhline(i - 0.5, color="k", lw=0.8)
        ax.set_xlim(0, 1); ax.set_xlabel(f"share of values at {FINAL}"); ax.grid(False)
        legend_outside(ax, "below", ncol=2)
        panel_title(ax, "b", "Outcome per entity, with the human ceiling")

    def s_c(ax):
        kinds = list(LIN_KIND.items())
        y = np.arange(len(e3s)); left = np.zeros(len(e3s))
        for k, c in kinds:
            v = e3s[k].to_numpy() if k in e3s else np.zeros(len(e3s))
            ax.barh(y, v, left=left, color=c, label=k, height=0.6, edgecolor="white", lw=.5); left += v
        ax.set_yticks(y); ax.set_yticklabels(e3s.index, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False)
        ax.set_xlabel(f"share of values that resolve to an ontology term, {FINAL}"); legend_outside(ax, "below", ncol=5)
        panel_title(ax, "c", "How wrong is wrong: ontology distance to the golden (Lin IC)")

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.45, height_ratios=[1.15, 1])
    s_a(fig.add_subplot(gs[0, 0])); s_b(fig.add_subplot(gs[0, 1])); s_c(fig.add_subplot(gs[1, :]))
    save_composite(fig, FIG, "figureS2_entities"); plt.close(fig)
    save_panels(plt, FIG, "figureS2_entities", {"a": s_a, "b": s_b, "c": s_c}, {"a": (6.5, 6), "b": (7, 6), "c": (11, 4.5)})
    dec.to_csv(FIG / "acceptance_decomposition_by_agent.csv")
    print(f"wrote figure2 and figureS2_entities as png+svg, panels under {FIG/'panels'}")


if __name__ == "__main__":
    main()
