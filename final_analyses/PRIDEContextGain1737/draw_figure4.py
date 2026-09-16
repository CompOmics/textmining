"""Figure 4, record vs article: what the manuscript, the PRIDE descriptor and
their combination each yield, in the style of Figure 2.

Three-way (manuscript / PRIDE descriptor / manuscript + PRIDE) wherever the
committed tables allow:
  a  populated fields per dataset, by input and by cohort (with / without methods)
  b  distinct values per dataset, same
  c  per-dataset gain of the combined input over each single input
Two-way, because only manuscript vs combined value pairs are committed:
  d  per-agent agreement of values on fields both inputs populate
  e  per-field coverage: manuscript only / both / PRIDE-added, grouped by agent

Inputs: three_source_comparison_excluding_PXD001017_v2/dataset_coverage.csv,
matched_prompt_comparison/job_966224/results/field_value_agreement.csv.
A three-way per-field / per-value panel needs the raw PRIDE-only outputs
(pride_only_1737/), which are on the cluster, not in the repository.

    python final_analyses/PRIDEContextGain1737/draw_figure4.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "manuscript_figure4"
COV = HERE / "three_source_comparison_excluding_PXD001017_v2" / "dataset_coverage.csv"
AGREE = HERE / "matched_prompt_comparison" / "job_966224" / "results" / "field_value_agreement.csv"

import sys
sys.path.insert(0, str(HERE.parent))
from figure_style import (SOURCE as SOURCE_COLORS, SOURCE_LABEL, COHORT_LABEL, AGENT as AGENT_COLORS, AGENT_LABEL, AGENTS,
                          AGREEMENT, NEUTRAL, style, panel_title, legend_outside, save_composite, save_panels)

SOURCES = [(k, SOURCE_LABEL[k]) for k in ("manuscript", "pride", "combined")]
COHORTS = [(k, COHORT_LABEL[k]) for k in ("abstract_only", "with_methods")]
STATUS = [(k, lab, col) for k, (lab, col) in AGREEMENT.items()]


def grouped_bars(ax, cov, what, ylabel):
    """mean per dataset with SD whiskers; groups = cohorts, bars = inputs."""
    x = np.arange(len(COHORTS)); w = 0.26
    for i, (src, lab) in enumerate(SOURCES):
        means = [cov[cov.cohort == c][f"{src}_{what}"].mean() for c, _ in COHORTS]
        sds = [cov[cov.cohort == c][f"{src}_{what}"].std() for c, _ in COHORTS]
        ax.bar(x + (i - 1) * w, means, w, yerr=sds, color=SOURCE_COLORS[src], label=lab, capsize=2, error_kw={"lw": .8})
        for xi, m in zip(x + (i - 1) * w, means):
            ax.text(xi, m + 1.2, f"{m:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([lab.replace(" manuscript", "") for c, lab in COHORTS])
    ax.set_ylabel(ylabel.replace(" per dataset", "")); ax.grid(False, axis="x")


def main():
    plt = style()
    FIG.mkdir(exist_ok=True)
    cov = pd.read_csv(COV)
    agree = pd.read_csv(AGREE)

    # ---- tables behind the panels
    gain = cov.assign(over_manuscript=cov.combined_fields - cov.manuscript_fields,
                      over_pride=cov.combined_fields - cov.pride_fields)
    t = (cov.groupby("cohort")[[f"{s}_{w}" for s, _ in SOURCES for w in ("fields", "values")]]
         .agg(["mean", "std"]).round(2))
    t.to_csv(FIG / "coverage_by_cohort.csv")
    shared = agree[agree.status.isin([s for s, _, _ in STATUS])]
    ag = shared.groupby(["agent", "status"]).size().unstack("status", fill_value=0)
    ag = ag.div(ag.sum(axis=1), axis=0).reindex(AGENTS)
    ag.to_csv(FIG / "value_agreement_by_agent.csv")
    def field_cov(a):
        t = a.groupby(["agent", "field", "status"]).size().unstack("status", fill_value=0)
        t["manuscript_only"] = t.get("earlier_only_field", 0); t["pride_added"] = t.get("expanded_only_field", 0)
        t["both"] = t[[s_ for s_, _, _ in STATUS if s_ in t]].sum(axis=1)
        t["n_datasets_populated"] = t[["manuscript_only", "pride_added", "both"]].sum(axis=1)
        return t[["manuscript_only", "both", "pride_added", "n_datasets_populated"]].reset_index()
    fc_cohort = {c: field_cov(agree[agree.cohort == c]) for c, _ in COHORTS}
    n_cohort = {c: agree[agree.cohort == c].pxd.nunique() for c, _ in COHORTS}
    for c, t in fc_cohort.items():
        t.to_csv(FIG / f"field_coverage_manuscript_vs_combined_{c}.csv", index=False)
    fc = agree.groupby(["agent", "field", "status"]).size().unstack("status", fill_value=0)
    fc["manuscript_only"] = fc.get("earlier_only_field", 0)
    fc["pride_added"] = fc.get("expanded_only_field", 0)
    fc["both"] = fc[[s for s, _, _ in STATUS if s in fc]].sum(axis=1)
    fc["n_datasets_populated"] = fc[["manuscript_only", "pride_added", "both"]].sum(axis=1)
    fc = fc[["manuscript_only", "both", "pride_added", "n_datasets_populated"]].reset_index()
    fc = fc.sort_values(["agent", "n_datasets_populated"], ascending=[True, False])
    fc.to_csv(FIG / "field_coverage_manuscript_vs_combined.csv", index=False)

    # ---- panels
    def p_a(ax):
        grouped_bars(ax, cov, "fields", "populated fields per dataset")
        legend_outside(ax, "right"); panel_title(ax, "a")

    def p_b(ax):
        grouped_bars(ax, cov, "values", "distinct values per dataset")
        legend_outside(ax, "right"); panel_title(ax, "b")

    def p_c(ax):
        data, labels, colors = [], [], []
        for c, clab in COHORTS:
            sub = gain[gain.cohort == c]
            data += [sub.over_manuscript, sub.over_pride]
            labels += [f"vs abstract + M&M\n{clab.split('-')[0]}", f"vs PRIDE\n{clab.split('-')[0]}"]
            colors += [SOURCE_COLORS["manuscript"], SOURCE_COLORS["pride"]]
        parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col); body.set_alpha(.55); body.set_edgecolor("none")
        parts["cmedians"].set_color("k")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("fields gained"); ax.grid(False, axis="x")
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.25)
        for i, d in enumerate(data, 1):
            ax.text(i, ax.get_ylim()[1] * 0.93, f"{d.median():+.0f}", ha="center", va="top", fontsize=7.5)
        panel_title(ax, "c")

    def p_d(ax):
        """Per agent, two bars: HAMLET's matcher tiers collapsed to three
        (agree = equivalent + hierarchy, partial = SapBERT candidate +
        partial overlap, disagree = no accepted overlap) and Lin IC on the
        ontology-backed fields (same term / related / distant)."""
        from figure_style import LIN_KIND
        lp = FIG / "lin_agreement_pairs.csv"
        lin = pd.read_csv(lp) if lp.exists() else pd.DataFrame(columns=["agent", "lin"])
        lin = lin[lin.lin.notna()]
        m3 = pd.DataFrame({"agree": ag.get("equivalent_value_sets", 0) + ag.get("hierarchically_related_value_sets", 0),
                           "partial": ag.get("semantic_candidate_value_sets", 0) + ag.get("partial_overlap", 0),
                           "disagree": ag.get("no_accepted_overlap_review", 0)}).reindex(AGENTS)
        l3 = pd.DataFrame({a_: {"agree": (g.lin >= 0.99).mean(), "partial": ((g.lin >= 0.5) & (g.lin < 0.99)).mean(), "disagree": (g.lin < 0.5).mean()}
                           for a_, g in lin.groupby("agent")}).T.reindex(AGENTS)
        # experimental design: field-appropriate rule (Lin on EFO/Mondo/UBERON/CL/ChEBI for technology type and
        # factor values, identity for the design vocabulary, equal / within one / off for counts), expdesign_agreement.py
        ep = FIG / "expdesign_agreement_pairs.csv"
        if ep.exists():
            e = pd.read_csv(ep); e = e[e.category.notna()]
            l3.loc["ExperimentalDesignAgent"] = e.category.value_counts(normalize=True).reindex(["agree", "partial", "disagree"]).fillna(0)
        lin_cols = [("agree", LIN_KIND["same term"]), ("partial", LIN_KIND["related (Lin 0.5-0.99)"]), ("disagree", LIN_KIND["unrelated (Lin <= 0.01)"])]
        m5 = ag.reindex(columns=[k for k, _, _ in STATUS], fill_value=0)
        y = np.arange(len(AGENTS)); h = 0.36
        # matcher: five tiers
        left = np.zeros(len(AGENTS))
        for k, lab, color in STATUS:
            v = m5[k].to_numpy()
            ax.barh(y - h / 2, v, h, left=left, color=color, edgecolor="white", lw=.5, label=f"matcher: {lab}")
            for yi, l, w in zip(y - h / 2, left, v):
                if w >= 0.08:
                    ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7, color="white" if k in ("equivalent_value_sets", "no_accepted_overlap_review") else "black")
            left += v
        # Lin: three
        left = np.zeros(len(AGENTS))
        for col, color in lin_cols:
            v = l3[col].fillna(0).to_numpy()
            ax.barh(y + h / 2, v, h, left=left, color=color, edgecolor="white", lw=.5, label=f"Lin IC: {col}")
            for yi, l, w in zip(y + h / 2, left, v):
                if w >= 0.08:
                    ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7, color="white" if col != "partial" else "black")
            left += v
        for yi, a_ in zip(y, AGENTS):
            ax.text(-0.02, yi - h / 2, "HAMLET matcher", ha="right", va="center", fontsize=7, color="#555555")
            ax.text(-0.02, yi + h / 2, "Lin IC" if a_ != "ExperimentalDesignAgent" else "Lin IC / rule", ha="right", va="center", fontsize=7, color="#555555")
            if l3.loc[a_].isna().all() or l3.loc[a_].sum() == 0:
                ax.text(0.02, yi + h / 2, "no ontology-backed field", va="center", fontsize=7, color="gray")
        ax.set_yticks(y); ax.set_yticklabels([AGENT_LABEL[a_] for a_ in AGENTS], fontsize=9); ax.tick_params(axis="y", pad=62)
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("share of fields populated by both"); ax.grid(False)
        legend_outside(ax, "right"); panel_title(ax, "d")

    order = list(fc.sort_values("n_datasets_populated", ascending=False).field)   # same order in e and f

    def p_s(ax, cohort="abstract_only", letter="a"):
        """per-field version (supplementary)"""
        t = fc_cohort[cohort].set_index("field").reindex(order).reset_index()
        n = n_cohort[cohort]
        x = np.arange(len(t)); bottom = np.zeros(len(t))
        for col, color, lab in [("both", NEUTRAL, "both inputs"), ("manuscript_only", SOURCE_COLORS["manuscript"], "abstract + M&M only"),
                                ("pride_added", SOURCE_COLORS["pride"], "added by PRIDE")]:
            v = t[col].fillna(0).to_numpy() / n
            ax.bar(x, v, bottom=bottom, color=color, width=0.78, label=lab, zorder=2); bottom += v
        ax.set_xticks(x); ax.set_xticklabels(t.field, fontsize=6.4, rotation=90)
        agent_of = dict(zip(fc.field, fc.agent))
        for tick, f in zip(ax.get_xticklabels(), t.field):
            tick.set_color(AGENT_COLORS[agent_of[f]])
        ax.set_xlim(-0.7, len(t) - 0.3); ax.set_ylim(0, 1)
        ax.set_ylabel(f"share of {COHORT_LABEL[cohort].split('-')[0]}-access datasets")
        ax.grid(False, axis="x")
        ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        panel_title(ax, letter)

    def p_s_open(ax):
        p_s(ax, "with_methods", "b")

    agent_share = {}
    for c, _ in COHORTS:
        a = agree[agree.cohort == c]
        t = a.groupby(["agent", "status"]).size().unstack("status", fill_value=0)
        d = pd.DataFrame({"manuscript_only": t.get("earlier_only_field", 0), "pride_added": t.get("expanded_only_field", 0)})
        d["both"] = t[[s_ for s_, _, _ in STATUS if s_ in t]].sum(axis=1)
        d = d.div(d.sum(axis=1), axis=0).reindex(AGENTS)
        agent_share[c] = d
    pd.concat(agent_share, names=["cohort"]).round(3).to_csv(FIG / "field_population_by_agent_and_cohort.csv")

    def p_e(ax, cohort="abstract_only", letter="e"):
        d = agent_share[cohort]
        y = np.arange(len(d)); left = np.zeros(len(d))
        for col, color, lab in [("both", NEUTRAL, "both inputs"), ("manuscript_only", SOURCE_COLORS["manuscript"], "abstract + M&M only"),
                                ("pride_added", SOURCE_COLORS["pride"], "added by PRIDE")]:
            v = d[col].to_numpy()
            ax.barh(y, v, left=left, color=color, height=0.62, label=lab, edgecolor="white", lw=.5)
            for yi, l, w in zip(y, left, v):
                if w >= 0.06:
                    ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if col != "pride_added" else "black")
            left += v
        ax.set_yticks(y); ax.set_yticklabels([AGENT_LABEL[a_] for a_ in d.index]); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_xlabel(f"share of populated fields, {COHORT_LABEL[cohort].split('-')[0]}-access"); ax.grid(False)
        legend_outside(ax, "right"); panel_title(ax, letter)

    def p_f(ax):
        p_e(ax, "with_methods", "f")

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.7, height_ratios=[1, 1, 0.8])
    p_a(fig.add_subplot(gs[0, 0])); p_b(fig.add_subplot(gs[0, 1]))
    p_c(fig.add_subplot(gs[1, 0])); p_d(fig.add_subplot(gs[1, 1]))
    p_e(fig.add_subplot(gs[2, 0])); p_f(fig.add_subplot(gs[2, 1]))
    save_composite(fig, FIG, "figure4"); plt.close(fig)
    # supplementary: per-field population, both cohorts
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"hspace": 0.95})
    p_s(axes[0], "abstract_only", "a"); p_s_open(axes[1])
    save_composite(fig, FIG, "figureS4_fields"); plt.close(fig)
    save_panels(plt, FIG, "figureS4_fields", {"a": p_s, "b": p_s_open}, {"a": (11, 4.2), "b": (11, 4.2)})
    save_panels(plt, FIG, "figure4", {"a": p_a, "b": p_b, "c": p_c, "d": p_d, "e": p_e,
                                       "f": p_f},
                {"a": (6, 3.6), "b": (6, 3.6), "c": (6.5, 3.8), "d": (7.5, 3.6), "e": (6, 3), "f": (6, 3)})
    print(f"wrote figure4 png+svg and panels under {FIG}")


if __name__ == "__main__":
    main()
