"""Entity coverage and Lin-IC value agreement, manuscript vs manuscript + PRIDE.

Two questions on matched_prompt_comparison/job_966224/field_changes.csv
(per dataset and field: the value list from the manuscript-only run and from
the manuscript + PRIDE run):
  1. coverage: is the field populated in one, the other, or both?
  2. agreement: where both populate it, how close are the values, scored
     with Lin similarity on the field's ontology (same resolver and graphs
     as the staircase analysis), best match per manuscript value, mean over
     values. Fields without an ontology are scored by normalised string
     identity only.

Manuscript vs PRIDE-only per field cannot be computed here: the PRIDE-only
condition is committed only as per-dataset totals.

    python final_analyses/PRIDEContextGain1737/lin_agreement.py
Writes manuscript_figure4/lin_agreement_by_field.csv, lin_agreement_pairs.csv, figure4_lin.{png,svg}.
"""
from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "manuscript_figure4"
FC = HERE / "matched_prompt_comparison/job_966224/field_changes.csv"
FRAMEWORK = REPO / "framework"
sys.path.insert(0, str(FRAMEWORK)); sys.path.insert(0, str(REPO / "final_analyses")); sys.path.insert(0, str(REPO / "final_analyses"))
logging.disable(logging.WARNING)

FIELD_ONT = {"tissue": ["uberon"], "cell_type": ["cl", "clo"], "cell_line": ["clo", "cl"], "disease_state": ["mondo"],
             "species": ["species"], "instrument": ["psi-ms"], "cleavage_agent": ["psi-ms"], "labeling": ["psi-ms"]}
FLAT = {"species"}
AGENT_OF = {"tissue": "BiologicalAgent", "cell_type": "BiologicalAgent", "cell_line": "BiologicalAgent", "disease_state": "BiologicalAgent",
            "species": "BiologicalAgent", "instrument": "TechnicalAgent", "cleavage_agent": "TechnicalAgent", "labeling": "TechnicalAgent"}


def main():
    from icgraph import IcGraph
    from normalization.normalizer import TermNormalizer
    from normalization.config import NormalizationConfig
    from normalization.download import create_species_ontology

    onto = FRAMEWORK / "ontologies"; create_species_ontology(onto)
    cfg = NormalizationConfig(); cfg.ontology_dir, cfg.cache_dir = str(onto), str(FRAMEWORK / "ontology_cache")
    nz = TermNormalizer(cfg)
    for o, f in (("cl", "cl.obo"), ("uberon", "uberon.obo"), ("mondo", "mondo.obo"), ("psi-ms", "psi-ms.obo"), ("clo", "clo.owl"), ("species", "species.obo")):
        nz.load_ontology(o, str(onto / f))
    graphs = {"uberon": IcGraph(onto / "uberon.obo"), "cl": IcGraph(onto / "cl.obo"), "mondo": IcGraph(onto / "mondo.obo", relations=("is_a",)),
              "psi-ms": IcGraph(onto / "psi-ms.obo"), "clo": IcGraph(onto / "clo.owl"), "species": IcGraph(onto / "species.obo", relations=("is_a",))}

    @lru_cache(maxsize=None)
    def resolve(term, o):
        r = nz.normalize(term, ontology_id=o)
        return r.ontology_id if r.is_normalized else None

    def in_graph(g, t, o):
        return bool(t) and (o in FLAT or t in g.terms or t in g.parents)

    def lin_sets(a_vals, b_vals, field):
        """mean over manuscript values of the best Lin against the combined values.
        None when no manuscript value resolves."""
        out = []
        for a in a_vals:
            best, resolved = 0.0, False
            for o in FIELD_ONT[field]:
                g = graphs[o]; aid = resolve(a, o)
                if not in_graph(g, aid, o):
                    continue
                resolved = True
                for b in b_vals:
                    bid = resolve(b, o)
                    if in_graph(g, bid, o):
                        best = max(best, float(aid == bid) if o in FLAT else g.lin(bid, aid))
            if resolved:
                out.append(best)
        return float(np.mean(out)) if out else None

    fc = pd.read_csv(FC)
    fc["old"] = fc.old_values.map(json.loads); fc["new"] = fc.new_values.map(json.loads)
    fc["in_manuscript"] = fc.old.map(len) > 0; fc["in_combined"] = fc.new.map(len) > 0
    cov = fc.groupby("field").agg(agent=("agent", "first"), datasets=("pxd", "size"),
                                  manuscript=("in_manuscript", "mean"), combined=("in_combined", "mean"),
                                  both=("in_manuscript", lambda s: (s & fc.loc[s.index, "in_combined"]).mean()))
    both = fc[fc.in_manuscript & fc.in_combined & fc.field.isin(FIELD_ONT)].copy()
    print(f"scoring {len(both)} dataset-field pairs over {both.field.nunique()} ontology-backed fields", flush=True)
    both["lin"] = [lin_sets(a, b, f) for a, b, f in zip(both.old, both.new, both.field)]
    both["string_identical"] = [set(a) == set(b) for a, b in zip(both.old, both.new)]
    both[["pxd", "cohort", "agent", "field", "old_values", "new_values", "lin", "string_identical"]].to_csv(FIG / "lin_agreement_pairs.csv", index=False)
    res = both[both.lin.notna()]
    agr = res.groupby("field").agg(n_both=("lin", "size"), resolved_share=("lin", lambda s: len(s) / (both.field == s.name).sum() if False else np.nan),
                                   mean_lin=("lin", "mean"), same_term=("lin", lambda s: (s >= 0.99).mean()),
                                   related=("lin", lambda s: ((s >= 0.5) & (s < 0.99)).mean()), distant=("lin", lambda s: (s < 0.5).mean()),
                                   string_identical=("string_identical", "mean"))
    agr["resolved_share"] = res.groupby("field").size() / both.groupby("field").size()
    table = cov.join(agr, how="left").sort_values(["agent", "both"], ascending=[True, False])
    table.round(3).to_csv(FIG / "lin_agreement_by_field.csv")
    print(table.round(2).to_string())
    draw(table)


def draw(table):
    from figure_style import style, panel_title, legend_outside, save_composite, save_panels, LIN_KIND, SOURCE, NEUTRAL, AGENT, AGENT_LABEL
    plt = style()
    t = table[table.mean_lin.notna()].sort_values("same_term", ascending=False)

    def p_a(ax):
        y = np.arange(len(t)); left = np.zeros(len(t))
        for col, lab, c in [("same_term", "same term (Lin ≥ 0.99)", LIN_KIND["same term"]), ("related", "related (0.5–0.99)", LIN_KIND["related (Lin 0.5-0.99)"]),
                            ("distant", "distant (< 0.5)", LIN_KIND["unrelated (Lin <= 0.01)"])]:
            v = t[col].to_numpy(); ax.barh(y, v, left=left, color=c, height=0.65, edgecolor="white", lw=.5, label=lab); left += v
        ax.set_yticks(y); ax.set_yticklabels([f"{f}  ({int(n):,})" for f, n in zip(t.index, t.n_both)], fontsize=8); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_xlabel("share of datasets populating the field in both runs"); ax.grid(False)
        legend_outside(ax, "below", ncol=3); panel_title(ax, "a")

    def p_b(ax):
        c = table.sort_values(["agent", "both"], ascending=[True, False])
        y = np.arange(len(c))
        ax.barh(y, c.manuscript, color=SOURCE["manuscript"], height=0.4, label="manuscript")
        ax.barh(y + 0.4, c.combined, color=SOURCE["combined"], height=0.4, label="manuscript + PRIDE")
        ax.scatter(c.both, y + 0.2, marker="|", s=60, color="k", zorder=3, label="both")
        ax.set_yticks(y + 0.2); ax.set_yticklabels(c.index, fontsize=6.5); ax.invert_yaxis()
        agents = list(c.agent)
        for i in range(1, len(agents)):
            if agents[i] != agents[i - 1]:
                ax.axhline(i - 0.1, color="k", lw=0.8)
        ax.set_xlim(0, 1); ax.set_xlabel("share of 1,737 datasets with the field populated"); ax.grid(False)
        legend_outside(ax, "below", ncol=3); panel_title(ax, "b")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.1]})
    p_b(axes[0]); p_a(axes[1])
    save_composite(fig, FIG, "figure4_lin"); plt.close(fig)
    save_panels(plt, FIG, "figure4_lin", {"a": p_b, "b": p_a}, {"a": (6.5, 7), "b": (6.5, 4)})
    print(f"wrote {FIG / 'figure4_lin.png'}")


if __name__ == "__main__":
    main()
