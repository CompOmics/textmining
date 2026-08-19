"""
Export extracted metadata JSONs to a project-level TSV.

One row per sample group, with multi-value fields joined by "; ".
Uses ontology_name when available, falls back to raw value.

Usage:
    python export_tsv.py <input_dir> [output.tsv]
    python export_tsv.py results/extracted_metadata metadata.tsv
"""

import csv
import json
import sys
from pathlib import Path


# Fields to export — order determines column order
EVIDENCE_FIELDS = [
    "organism", "tissue", "disease", "cell_part", "cell_line",
    "instrument", "fragmentation", "enzymes", "modifications",
    "collision_energy", "gradient_time_min", "lc_column",
    "acquisition", "labeling", "ionization",
    "treatment_type", "treatment_name", "treatment_class",
]

SPECIAL_FIELDS = ["fractionation", "enrichment"]


# Programmatic mappings for known values that lack ontology matches
VALUE_OVERRIDES = {
    "disease free": "healthy",
    "normal": "healthy",
    "control": "healthy",
    "not applicable": "",
    "not available": "",
}


def extract_values(data) -> tuple[str, str]:
    """Extract only normalized values from a field.

    Returns (values_str, scores_str) both joined by '; '.
    Values without an ontology_name are checked against VALUE_OVERRIDES,
    and dropped if no override exists.
    """
    if isinstance(data, list):
        vals = []
        scores = []
        for item in data:
            if not isinstance(item, dict) or item.get("value") is None:
                continue
            if item.get("ontology_name"):
                vals.append(str(item["ontology_name"]))
                scores.append(str(round(item.get("similarity", 0), 3)))
            else:
                override = VALUE_OVERRIDES.get(str(item["value"]).lower())
                if override:
                    vals.append(override)
                    scores.append("1.0")
        return "; ".join(vals), "; ".join(scores)
    return "", ""


def extract_special(data) -> str:
    """Extract value and method from fractionation/enrichment fields."""
    if not isinstance(data, dict):
        return ""
    parts = []
    v = data.get("value")
    if v is not None and v is not False:
        parts.append(str(v))
    m = data.get("method_ontology_name") or data.get("method")
    if m:
        parts.append(str(m))
    return "; ".join(parts) if parts else ""


def export_tsv(input_dir: Path, output_path: Path):
    json_files = sorted(input_dir.glob("PXD*.json"))
    print(f"Exporting {len(json_files)} PXDs from {input_dir}")

    # Build column list: value + score for each evidence field, then special fields
    columns = ["pxd", "group_name"]
    for field in EVIDENCE_FIELDS:
        columns.append(field)
        columns.append(f"{field}_score")
    columns += SPECIAL_FIELDS
    rows = []

    for f in json_files:
        pxd_id = f.stem
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  Skipping {pxd_id}: {e}")
            continue

        for group in data.get("sample_groups", []):
            row = {
                "pxd": pxd_id,
                "group_name": group.get("name", ""),
            }
            for field in EVIDENCE_FIELDS:
                vals, scores = extract_values(group.get(field, []))
                row[field] = vals
                row[f"{field}_score"] = scores
            for field in SPECIAL_FIELDS:
                row[field] = extract_special(group.get(field, {}))

            # Skip empty groups (no normalized values in key fields)
            key_fields = ["organism", "tissue", "instrument"]
            if not any(row.get(f) for f in key_fields):
                continue

            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows ({len(json_files)} PXDs) to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_tsv.py <input_dir> [output.tsv]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("metadata.tsv")
    export_tsv(input_dir, output_path)
