#!/usr/bin/env python3
"""Create legacy benchmark layouts for frozen staircase outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


STAIRCASE = Path(__file__).resolve().parents[1]
HAMLET = STAIRCASE.parents[2]
AGENT_LAYOUT = {
    "BiologicalAgent": ("Biological_annotations", "biological"),
    "TechnicalAgent": ("technical_metadata_output", "technical"),
    "ExperimentalDesignAgent": ("experimental_design_output", "experimental"),
}
AGENT_FIELDS = {
    "BiologicalAgent": {
        "species", "tissue", "cell_type", "cell_line", "disease_state",
        "sample_source", "age", "anatomic_site_tumor", "BMI", "sex",
        "strain", "material_type", "developmental_stage", "ethnicity",
    },
    "TechnicalAgent": {
        "instrument", "fragmentation_method", "ionization_type", "labeling",
        "cleavage_agent", "enrichment_method", "fractionation_method",
        "acquisition_method", "collision_energy", "reduction_reagent",
        "alkylation_reagent", "reduction_concentration",
        "alkylation_concentration", "mass_analyzer", "ptm", "modification",
    },
    "ExperimentalDesignAgent": {
        "experimental_design", "biological_replicate", "technical_replicate",
        "factor_value", "number_of_fractions",
        "number_of_technical_replicates", "number_of_biological_replicates",
        "number_of_samples", "technology_type", "experiment_type",
        "quantification_method",
    },
}


def relative_symlink(source: Path, destination: Path) -> None:
    if not source.exists():
        raise RuntimeError(f"missing source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    if destination.is_symlink():
        if os.readlink(destination) == target:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"refusing to replace non-symlink: {destination}")
    destination.symlink_to(target)


def s0_payload(adapter_path: Path, fields: set[str]) -> dict[str, str]:
    adapter = json.loads(adapter_path.read_text())
    output: dict[str, str] = {}
    for field, mentions in adapter.get("fields", {}).items():
        if field not in fields:
            continue
        values, seen = [], set()
        for mention in mentions:
            value = str(mention.get("value", "")).strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                values.append(value)
        if values:
            output[field] = "; ".join(values)
    return output


def prepare_run(model_tag: str, run: Path, destination: Path) -> dict:
    manifest = json.loads((STAIRCASE / "inputs/test_30/manifest.json").read_text())
    pxd_ids = [record["pxd"] for record in manifest["records"]]
    counts: dict[str, dict[str, int]] = {}
    for arm in (f"S{i}" for i in range(6)):
        arm_root = destination / model_tag / arm
        arm_counts = {agent: 0 for agent in AGENT_LAYOUT}
        for pxd in pxd_ids:
            for agent, (legacy_dir, suffix) in AGENT_LAYOUT.items():
                if arm == "S0":
                    source = run / arm / "field_adapter" / f"{pxd}.json"
                    if not source.exists():
                        raise RuntimeError(f"missing source: {source}")
                    output = arm_root / pxd / legacy_dir / "temp_0.0" / f"{pxd}_{suffix}_{model_tag}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    rendered = json.dumps(s0_payload(source, AGENT_FIELDS[agent]), indent=2, sort_keys=True) + "\n"
                    if not output.exists() or output.read_text() != rendered:
                        output.write_text(rendered)
                elif arm == "S5":
                    source = run / arm / "NormalizedAgent" / agent / f"{pxd}_{suffix}_{model_tag}.json"
                    output = arm_root / pxd / "normalized_output" / agent / "temp_0.0" / source.name
                    relative_symlink(source, output)
                else:
                    source = run / arm / agent / f"{pxd}_{suffix}_{model_tag}.json"
                    output = arm_root / pxd / legacy_dir / "temp_0.0" / source.name
                    relative_symlink(source, output)
                arm_counts[agent] += 1
        counts[arm] = arm_counts
    return {"model_tag": model_tag, "source_run": str(run.resolve()), "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="model_tag=/absolute/job/path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in args.run:
        model_tag, run_path = spec.split("=", 1)
        records.append(prepare_run(model_tag, Path(run_path), args.output))
    result = {
        "purpose": "Directory adapter for the SDRF benchmark",
        "gold_dir": str((HAMLET / "textmining/agentic-benchmark/benchmark_data/SDRFS_github").resolve()),
        "models": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
