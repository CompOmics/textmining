#!/usr/bin/env python3
"""Build a TechnicalAgent record from explicit PRIDE project properties."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
FRAMEWORK = PROJECT / "framework"
sys.path.insert(0, str(FRAMEWORK))

from docetl_pipeline.run_docetl import (  # noqa: E402
    _add_confidence,
    _add_hallucination_flags,
    _ground_multivalue_evidence,
    _run_normalization,
)


FIELDS = (
    "acquisition_method",
    "alkylation_concentration",
    "alkylation_reagent",
    "cleavage_agent",
    "collision_energy",
    "enrichment_method",
    "fractionation_method",
    "fragmentation_method",
    "instrument",
    "ionization_type",
    "labeling",
    "mass_analyzer",
    "ptm",
    "reduction_concentration",
    "reduction_reagent",
)


def pride_properties(text: str) -> dict[str, tuple[str, list[str]]]:
    properties: dict[str, tuple[str, list[str]]] = {}
    in_pride = False
    for line in text.splitlines():
        if line.strip() == "=== PRIDE PROJECT PROPERTIES ===":
            in_pride = True
            continue
        if in_pride and line.startswith("=== "):
            break
        if not in_pride or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values = [item.strip() for item in value.split(";") if item.strip()]
        properties[key.strip().casefold()] = (line, values)
    return properties


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-tag", required=True)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8", errors="replace")
    properties = pride_properties(text)
    record = {field: [{"value": "unknown", "evidence": ""}] for field in FIELDS}

    if "experiment types" in properties:
        line, values = properties["experiment types"]
        record["acquisition_method"] = [
            {"value": value, "evidence": line} for value in values
        ]
    if "instruments" in properties:
        line, values = properties["instruments"]
        record["instrument"] = [
            {"value": value, "evidence": line} for value in values
        ]
        orbitraps = [value for value in values if "orbitrap" in value.casefold()]
        if orbitraps:
            record["mass_analyzer"] = [
                {"value": "Orbitrap", "evidence": f"inferred: {line}"}
            ]
    if "modifications" in properties:
        line, values = properties["modifications"]
        record["ptm"] = [{"value": value, "evidence": line} for value in values]
    if "quantification" in properties:
        line, values = properties["quantification"]
        record["labeling"] = [
            {"value": value, "evidence": line} for value in values
        ]

    _ground_multivalue_evidence(record, text)
    _add_confidence(record, text)
    _add_hallucination_flags(record)

    agent_dir = args.output / "TechnicalAgent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.input.stem}_technical_{args.model_tag}.json"
    (agent_dir / filename).write_text(json.dumps(record, indent=2) + "\n")

    _run_normalization(
        args.output,
        [("pipeline_technical.yaml", "TechnicalAgent", "_technical")],
    )
    print(f"ptm_count={len(record['ptm'])}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
