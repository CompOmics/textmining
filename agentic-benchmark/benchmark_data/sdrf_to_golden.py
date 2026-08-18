#!/usr/bin/env python3
"""
SDRF → Golden Set Converter
============================
Converts SDRF .tsv files into the golden-set JSON format used by
compare_llm_golden.py / semantic_matcher.py for benchmark evaluation.

Each SDRF produces 3 JSONs:
  - <PXD>_BiologicalAgent_golden.json
  - <PXD>_TechnicalAgent_golden.json
  - <PXD>_ExperimentalDesignAgent_golden.json
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path for core.field_mappings import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.field_mappings import (
    SDRF_BIOLOGICAL as BIOLOGICAL_MAPPING,
    SDRF_TECHNICAL as TECHNICAL_MAPPING,
    SDRF_EXPERIMENTAL as EXPERIMENTAL_MAPPING,
)

# Columns that need special handling (multi-column or parsed)
SPECIAL_COLUMNS = {"modification", "biologicalreplicate", "fractionidentifier"}


def normalize_header(h: str) -> str:
    """Normalize an SDRF header for matching: lowercase, strip brackets/comments."""
    h = h.strip().lower()
    # Remove comment[...] wrapper but keep inner text
    h = re.sub(r"^comment\[(.+)\]$", r"\1", h)
    # Remove characteristics[...] wrapper
    h = re.sub(r"^characteristics\[(.+)\]$", r"\1", h)
    # Remove remaining non-alphanumeric except spaces
    h = re.sub(r"[^a-z0-9 ]", "", h)
    h = h.strip().replace(" ", "")
    return h


def parse_nt_value(entry: str) -> str:
    """Extract the NT= (name) value from an SDRF modification/instrument string.
    
    e.g. 'NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed' → 'Carbamidomethyl'
         'AC=MS:1001742;NT=LTQ Orbitrap Velos' → 'LTQ Orbitrap Velos'
    """
    m = re.search(r"NT=([^;]+)", entry)
    if m:
        return m.group(1).strip()
    return entry.strip()


def parse_ac_value(entry: str) -> str:
    """Extract the AC= (accession) value from an SDRF string.
    
    e.g. 'AC=MS:1001251;NT=Trypsin' → 'MS:1001251'
    """
    m = re.search(r"AC=([^;]+)", entry)
    if m:
        return m.group(1).strip()
    return None


def unique_values(values: list, parse_fn=None) -> str | None:
    """Deduplicate and join values, optionally applying a parse function."""
    parsed = []
    for v in values:
        if v is None or str(v).strip() == "" or str(v).strip().lower() in ("not available", "not applicable", "na", "n/a"):
            continue
        if parse_fn:
            p = parse_fn(str(v))
            if p and p not in parsed:
                parsed.append(p)
        else:
            s = str(v).strip()
            if s not in parsed:
                parsed.append(s)
    if not parsed:
        return None
    return "; ".join(parsed)


def convert_sdrf(sdrf_path: Path, pxd_id: str) -> dict:
    """Parse an SDRF TSV and return 3 golden-set dicts (one per agent)."""
    
    with open(sdrf_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return {}
    
    raw_headers = lines[0].strip().split("\t")
    norm_headers = [normalize_header(h) for h in raw_headers]
    
    # Parse data rows
    rows = []
    for line in lines[1:]:
        if line.strip():
            fields = line.strip().split("\t")
            rows.append(fields)
    
    if not rows:
        return {}
    
    # Build column index: norm_header → list of column indices (for multi-column like modification)
    col_index = defaultdict(list)
    for i, nh in enumerate(norm_headers):
        col_index[nh].append(i)
    
    def get_all_values(norm_name: str) -> list:
        """Get all values for a normalized header across all rows and columns."""
        vals = []
        for idx in col_index.get(norm_name, []):
            for row in rows:
                if idx < len(row):
                    vals.append(row[idx])
        return vals
    
    # ── Biological Agent ──
    bio_fields = {}
    for sdrf_col, golden_field in BIOLOGICAL_MAPPING.items():
        values = get_all_values(sdrf_col)
        if sdrf_col == "instrument":
            bio_fields[golden_field] = unique_values(values, parse_nt_value)
        else:
            bio_fields[golden_field] = unique_values(values)
    
    # Ensure all expected biological fields exist
    for field in ["species", "organ", "cell_type", "cell_line", "tissue", "disease",
                   "developmental_stage", "sex", "age", "strain", "ethnicity", "BMI", "material_type"]:
        if field not in bio_fields:
            bio_fields[field] = None
    
    bio_golden = {
        "pxd_id": pxd_id,
        "agent_type": "BiologicalAgent",
        "fields": bio_fields,
        "_source": "sdrf"
    }
    
    # ── Technical Agent ──
    tech_fields = {}
    # All technical SDRF columns use NT=...;AC=... format — parse the NT= value
    NT_PARSED_COLUMNS = {"instrument", "cleavageagent", "label",
                         "fragmentationmethod", "collisionenergy",
                         "ms2massanalyzer", "precursormasstolerance",
                         "fragmentmasstolerance"}
    for sdrf_col, golden_field in TECHNICAL_MAPPING.items():
        values = get_all_values(sdrf_col)
        if sdrf_col in NT_PARSED_COLUMNS:
            tech_fields[golden_field] = unique_values(values, parse_nt_value)
        else:
            tech_fields[golden_field] = unique_values(values)
    
    # Handle modifications (multiple columns, need NT= parsing + MT= categorization)
    mod_values = get_all_values("modification")
    all_mods = []
    for mv in mod_values:
        nt = parse_nt_value(mv)
        if nt and nt.lower() not in ("not available", "not applicable", "na", "n/a", ""):
            if nt not in all_mods:
                all_mods.append(nt)
    tech_fields["ptm"] = "; ".join(all_mods) if all_mods else None
    
    # Ensure all expected technical fields exist
    for field in ["instrument", "cleavage_agent", "ptm", "label", "fragmentation",
                   "precursor_tolerance", "fragment_tolerance", "collision_energy",
                   "mass_analyzer", "acquisition_method", "enrichment_method",
                   "fractionation", "reduction_reagent"]:
        if field not in tech_fields:
            tech_fields[field] = None
    
    tech_golden = {
        "pxd_id": pxd_id,
        "agent_type": "TechnicalAgent",
        "fields": tech_fields,
        "_source": "sdrf"
    }
    
    # ── Experimental Design Agent ──
    exp_fields = {}
    
    # Replicates: count unique biological replicate values
    rep_values = get_all_values("biologicalreplicate")
    unique_reps = set()
    for rv in rep_values:
        s = str(rv).strip()
        if s and s.lower() not in ("not available", "not applicable", "na", "n/a"):
            unique_reps.add(s)
    if unique_reps:
        exp_fields["replicates"] = str(len(unique_reps))
    else:
        exp_fields["replicates"] = None
    
    # Fractions: count unique fraction identifiers
    frac_values = get_all_values("fractionidentifier")
    unique_fracs = set()
    for fv in frac_values:
        s = str(fv).strip()
        if s and s.lower() not in ("not available", "not applicable", "na", "n/a"):
            unique_fracs.add(s)
    if unique_fracs:
        exp_fields["fractions"] = str(len(unique_fracs))
    else:
        exp_fields["fractions"] = None
    
    # Technology type
    for sdrf_col, golden_field in EXPERIMENTAL_MAPPING.items():
        # Need to handle "technology type" with space
        norm_key = sdrf_col.replace(" ", "")
        values = get_all_values(norm_key)
        exp_fields[golden_field] = unique_values(values)
    
    # Factor value — check for any factor value columns
    factor_vals = []
    for nh in col_index:
        if "factorvalue" in nh:
            factor_vals.extend(get_all_values(nh))
    exp_fields["factor_value"] = unique_values(factor_vals)
    
    for field in ["replicates", "missed_cleavages", "factor_value", "technology_type", "fractions"]:
        if field not in exp_fields:
            exp_fields[field] = None
    
    exp_golden = {
        "pxd_id": pxd_id,
        "agent_type": "ExperimentalDesignAgent",
        "fields": exp_fields,
        "_source": "sdrf"
    }
    
    return {
        "BiologicalAgent": bio_golden,
        "TechnicalAgent": tech_golden,
        "ExperimentalDesignAgent": exp_golden,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert SDRF files to golden-set JSON format")
    parser.add_argument("--input-dir", default="benchmark_data/matched",
                        help="Directory with PXD subfolders containing SDRF files")
    parser.add_argument("--output-dir", default="benchmark_data/sdrf_golden",
                        help="Output directory for golden-set JSON files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N PXDs (for testing)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pxd_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("PXD")])
    
    if args.limit:
        pxd_dirs = pxd_dirs[:args.limit]
    
    converted = 0
    skipped = 0
    
    for pxd_dir in pxd_dirs:
        pxd_id = pxd_dir.name
        sdrf_files = list(pxd_dir.glob("*.sdrf.tsv"))
        
        if not sdrf_files:
            print(f"  [SKIP] {pxd_id}: no SDRF file found")
            skipped += 1
            continue
        
        sdrf_path = sdrf_files[0]
        goldens = convert_sdrf(sdrf_path, pxd_id)
        
        if not goldens:
            print(f"  [SKIP] {pxd_id}: empty SDRF")
            skipped += 1
            continue
        
        for agent_name, golden_data in goldens.items():
            out_file = output_dir / f"{pxd_id}_{agent_name}_golden.json"
            with open(out_file, "w") as f:
                json.dump(golden_data, f, indent=2)
        
        converted += 1
        if converted % 20 == 0:
            print(f"  Converted {converted} PXDs...")
    
    print(f"\nDone: {converted} converted, {skipped} skipped")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
