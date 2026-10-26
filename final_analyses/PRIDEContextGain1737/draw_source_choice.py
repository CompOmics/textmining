"""One panel: which input supplies each field, and whether the two inputs agree.

Per field, over the 1,736 datasets: populated by the manuscript run only,
by the PRIDE-descriptor-augmented run only, or by both; where both, agreement
graded by the HAMLET matcher's five tiers (equivalent, hierarchy-related,
SapBERT candidate, partial overlap, no accepted overlap).
Six manuscript-excluded fields dropped.

    python final_analyses/PRIDEContextGain1737/draw_source_choice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "manuscript_figure4"
sys.path.insert(0, str(HERE.parent))
from figure_style import style, panel_title, save_composite, save_panels, SOURCE, LIN_KIND, AGENT, AGENT_LABEL, AGENTS, NEUTRAL  # noqa: E402

AGREE = HERE / "matched_prompt_comparison" / "job_966224" / "results" / "field_value_agreement.csv"
EXCLUDED = {"material_type", "ptm", "technology_type", "developmental_stage", "ethnicity", "mass_analyzer"}
from figure_style import AGREEMENT  # noqa: E402
# five HAMLET matcher tiers for fields populated by both runs, plus the two single-source categories
COLS = {"abstract (+M&M) only": SOURCE["manuscript"]}
COLS.update({lab: col for _, (lab, col) in AGREEMENT.items()})
COLS["PRIDE only"] = SOURCE["pride"]
STATUS_KEYS = {lab: k for k, (lab, _) in AGREEMENT.items()}


COHORTS = [("abstract_only", "closed-access", ".."), ("with_methods", "open-access", None)]


def field_table(a):
    rows = []
    for (agent, field), g in a.groupby(["agent", "field"]):
        r = {"agent": agent, "field": field, "abstract (+M&M) only": (g.status == "earlier_only_field").sum(), "PRIDE only": (g.status == "expanded_only_field").sum()}
        r.update({lab: (g.status == k).sum() for lab, k in STATUS_KEYS.items()})
        rows.append(r)
    t = pd.DataFrame(rows); t["populated"] = t[list(COLS)].sum(axis=1)
    return t


def build():
    """Return (tabs, pooled, keys, n_cohort) for the panel."""
    a = pd.read_csv(AGREE); a = a[~a.field.isin(EXCLUDED)]
    n_cohort = {c: a[a.cohort == c].pxd.nunique() for c, _, _ in COHORTS}
    tabs = {c: field_table(a[a.cohort == c]).set_index(["agent", "field"]) for c, _, _ in COHORTS}
    pooled = field_table(a)
    pooled = pooled[pooled.populated >= 50]
    pooled["agent"] = pd.Categorical(pooled.agent, AGENTS)
    pooled = pooled.sort_values(["agent", "populated"], ascending=[True, False])
    keys = list(zip(pooled.agent.astype(str), pooled.field))
    out = pd.concat({lab: tabs[c].reindex(keys) for c, lab, _ in COHORTS}, names=["cohort"]).reset_index()
    out.to_csv(FIG / "source_choice_by_field_and_cohort.csv", index=False)
    return tabs, pooled, keys, n_cohort


def source_choice_panel(ax, letter="", legend_ncol=4):
    tabs, pooled, keys, n_cohort = build()
    if True:
        y = np.arange(len(keys)); h = 0.38
        for j, (c, clab, hatch) in enumerate(COHORTS):
            off = -h / 2 if j == 0 else h / 2
            t = tabs[c].reindex(keys); left = np.zeros(len(keys))
            for k, col in COLS.items():
                v = (t[k].fillna(0) / n_cohort[c]).to_numpy()
                ax.barh(y + off, v, left=left, color=col, height=h, label=k if j == 1 else None, edgecolor="white", lw=.3, hatch=hatch); left += v
        labels = list(pooled.field)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        for tick, ag in zip(ax.get_yticklabels(), pooled.agent.astype(str)):
            tick.set_color(AGENT[ag])
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False, axis="y")
        ax.set_xlabel(f"share of datasets in which the field is populated (closed-access, dotted, n = {n_cohort['abstract_only']:,}; open-access, solid, n = {n_cohort['with_methods']:,})")
        ag_list = list(pooled.agent.astype(str))
        for i in range(1, len(keys)):
            if ag_list[i] != ag_list[i - 1]:
                ax.axhline(i - 0.5, color="k", lw=0.7)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=col, label=k) for k, col in COLS.items()]
        handles += [Patch(facecolor="white", edgecolor="#777777", hatch="..", label="closed-access"), Patch(facecolor="white", edgecolor="#777777", label="open-access")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=legend_ncol, fontsize=8, frameon=False)
        panel_title(ax, letter)


def agent_panel(ax, cohort, letter=""):
    """Per agent, share of populated (field, dataset) slots by source and agreement tier, one cohort."""
    tabs, pooled, keys, n_cohort = build()
    t = tabs[cohort].reindex(keys).reset_index()
    t["agent"] = t["level_0"] if "level_0" in t else t["agent"]
    g = t.groupby("agent")[list(COLS)].sum().reindex(AGENTS)
    g = g.div(g.sum(axis=1), axis=0)
    g.to_csv(FIG / f"source_choice_by_agent_{cohort}.csv")
    y = np.arange(len(AGENTS)); left = np.zeros(len(AGENTS))
    for k, col in COLS.items():
        v = g[k].to_numpy()
        ax.barh(y, v, left=left, color=col, height=0.62, label=k, edgecolor="white", lw=.5)
        for yi, l, w in zip(y, left, v):
            if w >= 0.06:
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if k in ("equivalent", "no accepted overlap") else "black")
        left += v
    ax.set_yticks(y); ax.set_yticklabels([AGENT_LABEL[a] for a in AGENTS]); ax.invert_yaxis()
    clab = "closed-access" if cohort == "abstract_only" else "open-access"
    ax.set_xlim(0, 1); ax.set_xlabel(f"share of populated fields, {clab}"); ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5); panel_title(ax, letter)


def agent_shares(cohort):
    tabs, pooled, keys, n_cohort = build()
    t = tabs[cohort].reindex(keys).reset_index()
    g = t.groupby("agent")[list(COLS)].sum().reindex(AGENTS)
    return g


def coverage_panel(ax, letter="d"):
    """Per agent, closed (dotted) and open (solid) access: share of populated
    (field, dataset) slots supplied by the manuscript run only, by both, or by
    the PRIDE-augmented run only."""
    from matplotlib.patches import Patch
    tiers = list(STATUS_KEYS)
    parts = [("abstract (+M&M) only", SOURCE["manuscript"]), ("both inputs", NEUTRAL), ("PRIDE only", SOURCE["pride"])]
    y = np.arange(len(AGENTS)); h = 0.36
    for j, (c, clab, hatch) in enumerate(COHORTS):
        g = agent_shares(c); g["both inputs"] = g[tiers].sum(axis=1)
        g = g[[k for k, _ in parts]].div(g[[k for k, _ in parts]].sum(axis=1), axis=0)
        g.to_csv(FIG / f"coverage_by_agent_{c}.csv")
        off = -h / 2 if j == 0 else h / 2; left = np.zeros(len(AGENTS))
        for k, col in parts:
            v = g[k].to_numpy()
            ax.barh(y + off, v, h, left=left, color=col, edgecolor="white", lw=.5)
            for yi, l, w in zip(y + off, left, v):
                if w >= 0.07:
                    ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7, color="white" if k == "both inputs" else "black")
            left += v
        for yi in y:
            ax.text(-0.02, yi + off, clab, ha="right", va="center", fontsize=7, color="#555555")
    ax.set_yticks(y); ax.set_yticklabels([AGENT_LABEL[a] for a in AGENTS], fontsize=9); ax.tick_params(axis="y", pad=58)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("share of populated fields"); ax.grid(False)
    handles = [Patch(facecolor=col, label=k) for k, col in parts]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5); panel_title(ax, letter)


def tier_panel(ax, cohort, letter=""):
    """Per agent, the fields populated by both runs tiered by the HAMLET matcher, one cohort."""
    tiers = list(STATUS_KEYS)
    g = agent_shares(cohort)[tiers]; g = g.div(g.sum(axis=1), axis=0)
    g.to_csv(FIG / f"matcher_tiers_by_agent_{cohort}.csv")
    y = np.arange(len(AGENTS)); left = np.zeros(len(AGENTS))
    for k in tiers:
        v = g[k].to_numpy()
        ax.barh(y, v, left=left, color=COLS[k], height=0.62, label=k, edgecolor="white", lw=.5)
        for yi, l, w in zip(y, left, v):
            if w >= 0.06:
                ax.text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if k in ("equivalent", "no accepted overlap") else "black")
        left += v
    ax.set_yticks(y); ax.set_yticklabels([AGENT_LABEL[a] for a in AGENTS]); ax.invert_yaxis()
    clab = "closed-access" if cohort == "abstract_only" else "open-access"
    ax.set_xlim(0, 1); ax.set_xlabel(f"share of fields populated by both, {clab}"); ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5); panel_title(ax, letter)


def field_tier_panel(ax, cohort, letter="", min_pairs=30):
    """Per field (grouped by agent), fields populated by both runs tiered by the HAMLET matcher, one cohort."""
    tabs, pooled, keys, n_cohort = build()
    tiers = list(STATUS_KEYS)
    t = tabs[cohort].reindex(keys)
    n = t[tiers].sum(axis=1)
    t = t[n >= min_pairs]; n = n[n >= min_pairs]
    g = t[tiers].div(n, axis=0)
    g.assign(n_pairs=n).to_csv(FIG / f"matcher_tiers_by_field_{cohort}.csv")
    y = np.arange(len(g)); left = np.zeros(len(g))
    for k in tiers:
        v = g[k].to_numpy()
        ax.barh(y, v, left=left, color=COLS[k], height=0.7, label=k, edgecolor="white", lw=.4); left += v
    ax.set_yticks(y); ax.set_yticklabels([f"{f} ({int(nn)})" for (_, f), nn in zip(g.index, n)], fontsize=8)
    agents = [a for a, _ in g.index]
    for tick, ag in zip(ax.get_yticklabels(), agents):
        tick.set_color(AGENT[ag])
    for i in range(1, len(agents)):
        if agents[i] != agents[i - 1]:
            ax.axhline(i - 0.5, color="k", lw=0.7)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False, axis="y")
    clab = "closed-access" if cohort == "abstract_only" else "open-access"
    ax.set_xlabel(f"share of fields populated by both, {clab}")
    panel_title(ax, letter)


def main():
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw={"wspace": 0.9})
    agent_panel(axes[0], "abstract_only", "a"); agent_panel(axes[1], "with_methods", "b")
    save_composite(fig, FIG, "figure_source_choice_agents"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 12))
    source_choice_panel(ax)
    save_composite(fig, FIG, "figure_source_choice"); plt.close(fig)
    save_panels(plt, FIG, "figure_source_choice", {"a": source_choice_panel}, {"a": (9, 12)})


if __name__ == "__main__":
    main()
