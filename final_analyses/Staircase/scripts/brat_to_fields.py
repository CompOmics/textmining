#!/usr/bin/env python3
"""Validate S0 BRAT spans and map labels to the shared extraction fields."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LINE = re.compile(r"^(T\d+)(?:\|\|\||\t|\|)([A-Za-z][A-Za-z0-9_]*) (\d+) (\d+)(?:\|\|\||\t|\|)(.*)$")

# Explicit mappings only. Unclear S0 labels remain auditable/unmapped.
LABEL_TO_FIELD = {
    "AcquisitionMethod": "acquisition_method", "Age": "age",
    "AlkylationReagent": "alkylation_reagent", "AnatomicSiteTumor": "anatomic_site_tumor",
    "AncestryCategory": "ethnicity", "BiologicalReplicate": "biological_replicate",
    "BMI": "BMI", "CellLine": "cell_line", "CellType": "cell_type",
    "CleavageAgent": "cleavage_agent", "CollisionEnergy": "collision_energy",
    "DevelopmentalStage": "developmental_stage", "Disease": "disease_state",
    "EnrichmentMethod": "enrichment_method", "Experiment": "experimental_design",
    "FactorValue": "factor_value", "FractionationMethod": "fractionation_method",
    "FragmentationMethod": "fragmentation_method", "Instrument": "instrument",
    "IonizationType": "ionization_type", "Label": "labeling", "MaterialType": "material_type",
    "Modification": "ptm", "MS2MassAnalyzer": "mass_analyzer",
    "NumberOfFractions": "number_of_fractions",
    "NumberOfTechnicalReplicates": "number_of_technical_replicates",
    "NumberOfBiologicalReplicates": "number_of_biological_replicates",
    "NumberOfSamples": "number_of_samples", "Organism": "species",
    "OrganismPart": "tissue", "ReductionReagent": "reduction_reagent",
    "Sex": "sex", "SourceName": "sample_source", "Strain": "strain",
    "TechnicalReplicate": "technical_replicate",
}


def convert(text: str, ann: str) -> dict:
    fields: dict[str, list[dict]] = {}
    invalid, unmapped = [], []
    span_status_counts = {"exact": 0, "unique_realign": 0, "ambiguous_surface": 0, "surface_absent": 0}
    for raw in ann.splitlines():
        if not raw.strip():
            continue
        match = LINE.match(raw)
        if not match:
            invalid.append({"line": raw, "reason": "syntax"})
            continue
        tid, label, start_s, end_s, surface = match.groups()
        # A few S0 responses add a fourth BRAT-like column or repeat
        # the surface using a single pipe. Keep the first surface cell; when
        # two pipe cells are identical, collapse the duplicate deterministically.
        surface = surface.split("\t", 1)[0]
        pipe_parts = surface.split("|")
        if len(pipe_parts) == 2 and pipe_parts[0] == pipe_parts[1]:
            surface = pipe_parts[0]
        start, end = int(start_s), int(end_s)
        actual = text[start:end] if 0 <= start <= end <= len(text) else None
        if start >= 0 and end > start and end <= len(text) and actual == surface:
            span_status = "exact"
            resolved_start, resolved_end = start, end
        else:
            occurrences = [match.start() for match in re.finditer(re.escape(surface), text)] if surface else []
            if len(occurrences) == 1:
                span_status = "unique_realign"
                resolved_start, resolved_end = occurrences[0], occurrences[0] + len(surface)
            elif occurrences:
                span_status = "ambiguous_surface"
                resolved_start = resolved_end = None
            else:
                span_status = "surface_absent"
                resolved_start = resolved_end = None
            invalid.append({"line": raw, "reason": "offset_or_surface_mismatch", "actual": actual,
                            "span_status": span_status, "occurrence_count": len(occurrences)})
        span_status_counts[span_status] += 1
        field = LABEL_TO_FIELD.get(label)
        item = {"id": tid, "label": label, "value": surface,
                "reported_start": start, "reported_end": end,
                "start": resolved_start, "end": resolved_end, "span_status": span_status}
        if field is None:
            unmapped.append(item)
        else:
            fields.setdefault(field, []).append(item)
    return {"fields": fields, "invalid": invalid, "unmapped": unmapped,
            "mapped_count": sum(map(len, fields.values())), "invalid_count": len(invalid),
            "unmapped_count": len(unmapped), "span_status_counts": span_status_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--ann", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strict-offsets", action="store_true")
    args = parser.parse_args()
    result = convert(args.text.read_text(), args.ann.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.strict_offsets and result["invalid_count"]:
        raise SystemExit(f"invalid BRAT spans: {result['invalid_count']}")


if __name__ == "__main__":
    main()
