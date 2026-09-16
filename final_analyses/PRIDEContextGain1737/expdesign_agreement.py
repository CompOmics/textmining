"""Field-appropriate agreement for the experimental-design fields, manuscript
vs manuscript + PRIDE, to sit next to Lin IC in Figure 4d.

  technology_type   Lin IC on EFO (fallback PSI-MS)
  factor_value      Lin IC on the best-resolving of EFO, Mondo, UBERON, CL, ChEBI
  experimental_design   controlled vocabulary of nine values: identical / not
  number_of_*, *_replicate   numbers: equal = agree, within one = partial, else disagree
Per dataset-field the best match over value pairs; free-text fields that
resolve nowhere are excluded (counted in `unscored`).

    python final_analyses/PRIDEContextGain1737/expdesign_agreement.py
Writes manuscript_figure4/expdesign_agreement_pairs.csv and _by_field.csv.
"""
from __future__ import annotations

import json
import logging
import re
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
sys.path.insert(0, str(FRAMEWORK)); sys.path.insert(0, str(REPO / "final_analyses")); sys.path.insert(0, str(HERE))
logging.disable(logging.WARNING)

ONT_FIELDS = {"technology_type": ["experimentalfactor", "psi-ms"],
              "factor_value": ["experimentalfactor", "mondo", "uberon", "cl", "chebi"]}
CATEGORICAL = ["experimental_design"]
NUMERIC = ["number_of_samples", "number_of_biological_replicates", "number_of_technical_replicates", "number_of_fractions",
           "biological_replicate", "technical_replicate"]


def num(x):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[a-z ]*", str(x).lower())
    return float(m.group(1)) if m else None


def main():
    from icgraph import IcGraph
    from normalization.normalizer import TermNormalizer
    from normalization.config import NormalizationConfig
    onto = FRAMEWORK / "ontologies"
    cfg = NormalizationConfig(); cfg.ontology_dir, cfg.cache_dir = str(onto), str(FRAMEWORK / "ontology_cache")
    nz = TermNormalizer(cfg)
    files = {"experimentalfactor": "experimentalfactor.obo", "psi-ms": "psi-ms.obo", "mondo": "mondo.obo", "uberon": "uberon.obo",
             "cl": "cl.obo", "chebi": "chebi.obo"}
    graphs = {}
    for o, f in files.items():
        if o == "chebi" and not (FRAMEWORK / "ontology_cache" / "chebi_index.pkl").exists():
            print("chebi index not built yet, skipped", flush=True)
            ONT_FIELDS["factor_value"].remove("chebi"); continue
        nz.load_ontology(o, str(onto / f))
        graphs[o] = IcGraph(onto / f, relations=("is_a", "part_of") if o in ("uberon", "cl", "psi-ms") else ("is_a",))
        print("loaded", o, graphs[o].N, flush=True)

    @lru_cache(maxsize=None)
    def resolve(term, o):
        r = nz.normalize(term, ontology_id=o)
        return r.ontology_id if r.is_normalized else None

    def in_graph(g, t):
        return bool(t) and (t in g.terms or t in g.parents)

    def lin_sets(a_vals, b_vals, onts):
        out = []
        for a in a_vals:
            best, resolved = 0.0, False
            for o in onts:
                g = graphs[o]; aid = resolve(a, o)
                if not in_graph(g, aid):
                    continue
                resolved = True
                for b in b_vals:
                    bid = resolve(b, o)
                    if in_graph(g, bid):
                        best = max(best, g.lin(bid, aid))
            if resolved:
                out.append(best)
        return float(np.mean(out)) if out else None

    fc = pd.read_csv(FC)
    fc = fc[fc.field.isin(list(ONT_FIELDS) + CATEGORICAL + NUMERIC)].copy()
    fc["old"] = fc.old_values.map(json.loads); fc["new"] = fc.new_values.map(json.loads)
    both = fc[(fc.old.map(len) > 0) & (fc.new.map(len) > 0)].copy()
    rows = []
    for r in both.itertuples():
        f, a, b = r.field, r.old, r.new
        if f in ONT_FIELDS:
            s = lin_sets(a, b, ONT_FIELDS[f]); kind = "lin"
            cat = None if s is None else ("agree" if s >= 0.99 else "partial" if s >= 0.5 else "disagree")
        elif f in CATEGORICAL:
            s = float(bool(set(x.lower() for x in a) & set(x.lower() for x in b))); kind = "exact"
            cat = "agree" if s else "disagree"
        else:
            na = [num(x) for x in a]; nb = [num(x) for x in b]; na = [x for x in na if x is not None]; nb = [x for x in nb if x is not None]
            kind = "numeric"
            if not na or not nb:
                s, cat = None, None
            else:
                d = min(abs(x - y) for x in na for y in nb); s = d
                cat = "agree" if d == 0 else "partial" if d <= 1 else "disagree"
        rows.append({"pxd": r.pxd, "cohort": r.cohort, "field": f, "kind": kind, "score": s, "category": cat,
                     "old_values": r.old_values, "new_values": r.new_values})
    pairs = pd.DataFrame(rows); pairs.to_csv(FIG / "expdesign_agreement_pairs.csv", index=False)
    by = pairs.groupby("field").agg(kind=("kind", "first"), n_both=("category", "size"), scored=("category", lambda s: s.notna().sum()),
                                    agree=("category", lambda s: (s == "agree").sum()), partial=("category", lambda s: (s == "partial").sum()),
                                    disagree=("category", lambda s: (s == "disagree").sum()))
    for c in ("agree", "partial", "disagree"):
        by[f"{c}_share"] = by[c] / by.scored
    by.round(3).to_csv(FIG / "expdesign_agreement_by_field.csv")
    sc = pairs[pairs.category.notna()]
    pooled = sc.category.value_counts(normalize=True).reindex(["agree", "partial", "disagree"]).fillna(0)
    print(by.round(2).to_string()); print("\npooled over scored experimental-design pairs:", pooled.round(3).to_dict(), "n =", len(sc), "unscored:", int(pairs.category.isna().sum()))


if __name__ == "__main__":
    main()
