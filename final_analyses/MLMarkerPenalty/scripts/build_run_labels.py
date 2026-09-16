"""Run-level labels with provenance, priority SDRF > run name > HAMLET.

Rebuilt from the former ExtendedTrainingSet/scripts/build_run_labels.py.
One row per (pxd, run) of the reprocessing run table. For organism, tissue,
disease and cell line a run takes the SDRF value if the SDRF has one, else
the run-name value, else HAMLET's project-level value; a multi-valued HAMLET
field is broadcast only when it has a single value, otherwise the run gets
"AMBIGUOUS:<a; b>" and, for tissue, tissue_ambiguous = True. material type,
sample source and cell type exist only at project level (HAMLET).

    python final_analyses/MLMarkerPenalty/scripts/build_run_labels.py
Writes results/run_labels.tsv.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
NAME = REPO / "MLMarker/Reprocessing_database/notebooks/run_meta_name.tsv"
SDRF = REPO / "MLMarker/Reprocessing_database/notebooks/run_metadata_sdrf.tsv"
HAMLET = REPO / "mlmarker_hamlet/output/HAMLET_normalized.tsv"
TRAINING = REPO / "MLMarker/Training_PXDs/SupplementaryTableS1.tsv"
OUT = ROOT / "results" / "run_labels.tsv"

HEALTHY = re.compile(r"^(normal|healthy|disease free|control|none|no disease|not diseased|wild type)$")
RUN_FIELDS = {"organism": "species", "tissue": "tissue", "disease": "disease_state", "cell_line": "cell_line"}
PROJECT_FIELDS = {"material_type": "material_type", "sample_source": "sample_source", "cell_type": "cell_type"}


def hamlet_values(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        o = ast.literal_eval(cell)
    except Exception:
        return [cell.strip().lower()]
    out = []
    for it in (o if isinstance(o, list) else [o]):
        v = (it.get("ontology_name") or it.get("value")) if isinstance(it, dict) else it
        if isinstance(v, str) and v.strip().lower() not in ("unknown", "false", ""):
            v = v.strip().lower()
            if v not in out:
                out.append(v)
    return out


def main():
    runs = pd.read_csv(NAME, sep="\t", low_memory=False).drop_duplicates(["pxd", "run"]).reset_index(drop=True)
    sdrf = pd.read_csv(SDRF, sep="\t", low_memory=False).drop_duplicates(["pxd", "run"])
    ham = pd.read_csv(HAMLET, sep="\t", low_memory=False).set_index("source_file")
    training = set(TRAINING.read_text().split()) - {"PXD_accession"}

    out = runs[["pxd", "run"]].copy()
    sd = out.merge(sdrf, on=["pxd", "run"], how="left")
    for col, hcol in RUN_FIELDS.items():
        s_val = sd[col].where(sd[col].notna() & ~sd[col].astype(str).str.strip().str.lower().isin(["", "-", "not available", "not applicable", "na", "n/a"]))
        n_val = runs[col].where(runs[col].notna() & runs[col].astype(str).str.strip().ne("")).reset_index(drop=True)
        h_list = out.pxd.map(ham[hcol].map(hamlet_values)) if hcol in ham else pd.Series([[]] * len(out))
        h_val = h_list.map(lambda v: v[0] if len(v) == 1 else (f"AMBIGUOUS:{'; '.join(v)}" if len(v) > 1 else None))
        value = s_val.str.lower().where(s_val.notna(), n_val.str.lower().where(n_val.notna(), h_val))
        source = pd.Series("hamlet", index=out.index).where(h_list.map(len) <= 1, "hamlet-multi")
        source = source.where(h_val.notna(), None)
        source = source.where(n_val.isna(), "name").where(s_val.isna(), "sdrf")
        out[col] = value; out[f"{col}_source"] = source
    for col, hcol in PROJECT_FIELDS.items():
        h_list = out.pxd.map(ham[hcol].map(hamlet_values))
        out[col] = h_list.map(lambda v: "; ".join(v) if v else None)
        out[f"{col}_source"] = out[col].map(lambda v: "hamlet" if v else None)

    org = out.organism.fillna("").str.lower()
    out["is_human"] = org.str.contains("homo sapiens|human")   # multi-species projects keep their human runs
    dis = out.disease.fillna("").str.lower()
    out["is_healthy"] = dis.map(lambda d: bool(HEALTHY.match(d.strip())))
    out["has_cell_line"] = out.cell_line.notna()
    out["in_old_atlas"] = out.pxd.isin(training)
    out["tissue_ambiguous"] = out.tissue.fillna("").str.startswith("AMBIGUOUS:")
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, sep="\t", index=False)
    print(f"{len(out):,} runs, {out.pxd.nunique():,} projects -> {OUT}")
    for col in RUN_FIELDS:
        print(f"  {col:10s} sources {out[f'{col}_source'].value_counts(dropna=False).to_dict()}")
    print(f"  human {int(out.is_human.sum()):,}, healthy {int(out.is_healthy.sum()):,}, cell line {int(out.has_cell_line.sum()):,}, "
          f"in training projects {int(out.in_old_atlas.sum()):,}, tissue ambiguous {int(out.tissue_ambiguous.sum()):,}")


if __name__ == "__main__":
    main()
