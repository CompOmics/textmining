"""Extract metadata from run filenames using regex patterns."""

import logging
import pandas as pd
from pathlib import Path

from .patterns import build_field_patterns, BODY_FLUIDS, MULTI_FIELDS

log = logging.getLogger(__name__)


def extract_from_run_name(run_name: str, field_patterns: dict) -> dict:
    """Match all field patterns against a run name.

    Returns first match per field, except MULTI_FIELDS which collect all matches.
    For tissue: body fluids take priority over solid tissues.
    """
    result = {}
    for field, patterns in field_patterns.items():
        if field in MULTI_FIELDS:
            matches = []
            for regex, canon in patterns:
                if regex.search(run_name) and canon not in matches:
                    matches.append(canon)
            result[field] = "; ".join(str(v) for v in matches) if matches else None
        elif field == "tissue":
            all_matches = []
            for regex, canon in patterns:
                if regex.search(run_name) and canon not in all_matches:
                    all_matches.append(canon)
            if not all_matches:
                result[field] = None
            else:
                fluid_matches = [m for m in all_matches if m in BODY_FLUIDS]
                result[field] = fluid_matches[0] if fluid_matches else all_matches[0]
        else:
            matched = None
            for regex, canon in patterns:
                if regex.search(run_name):
                    matched = canon
                    break
            result[field] = matched

    # Infer organism from cell line
    if result.get("cell_line"):
        if result["organism"] is None:
            result["organism"] = "Homo sapiens"
        elif result["organism"] != "Homo sapiens":
            result["organism"] = result["organism"] + "; Homo sapiens"

    return result


def run(quant_path: Path, output_path: Path, ontology_dir: Path = None):
    """Run the name extraction pipeline.

    Args:
        quant_path: Path to pride_quant.parquet (or any parquet with pxd, run columns)
        output_path: Where to write the output TSV
        ontology_dir: Path to ontologies/ directory (for cell line patterns)
    """
    log.info("Loading runs from %s", quant_path)
    all_runs = pd.read_parquet(quant_path, columns=["pxd", "run"])
    all_runs = all_runs.drop_duplicates(subset=["pxd", "run"]).reset_index(drop=True)
    log.info("Unique runs: %d", len(all_runs))

    field_patterns = build_field_patterns(ontology_dir)
    log.info("Patterns: %d fields", len(field_patterns))

    records = []
    for _, row in all_runs.iterrows():
        extracted = extract_from_run_name(row["run"], field_patterns)
        extracted["pxd"] = row["pxd"]
        extracted["run"] = row["run"]
        records.append(extracted)

    run_meta = pd.DataFrame(records)
    id_cols = ["pxd", "run"]
    field_cols = [c for c in run_meta.columns if c not in id_cols]
    run_meta = run_meta[id_cols + sorted(field_cols, key=str)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_meta.to_csv(output_path, sep="\t", index=False)

    # Coverage stats
    n = len(run_meta)
    for col in sorted(field_cols):
        filled = run_meta[col].notna().sum()
        if filled > 0:
            log.info("  %-20s %6d / %d (%.1f%%)", col, filled, n, filled / n * 100)

    log.info("Saved %d rows to %s", len(run_meta), output_path)
