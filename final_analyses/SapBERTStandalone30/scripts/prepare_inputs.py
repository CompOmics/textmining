#!/usr/bin/env python3
"""Adapt the frozen standalone-30 outputs to the historical benchmark layout."""

from __future__ import annotations

import json
import os
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
HAMLET = EXPERIMENT.parents[2]
SOURCE = HAMLET / "textmining/framework/smoke_outputs/qwen3_8_27b_qc30/test_set/NormalizedAgent"
DESTINATION = EXPERIMENT / "inputs/qwen3_8_27b"
AGENTS = {
    "BiologicalAgent": "Biological_annotations",
    "TechnicalAgent": "technical_metadata_output",
    "ExperimentalDesignAgent": "experimental_design_output",
}


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    if destination.is_symlink():
        if os.readlink(destination) == target:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"refusing to replace non-symlink: {destination}")
    destination.symlink_to(target)


def main() -> None:
    counts: dict[str, int] = {}
    pxd_ids: set[str] = set()
    for agent, _legacy_name in AGENTS.items():
        files = sorted((SOURCE / agent).glob("PXD*.json"))
        counts[agent] = len(files)
        for source in files:
            pxd = source.name.split("_", 1)[0]
            pxd_ids.add(pxd)
            destination = (
                DESTINATION / pxd / "normalized_output" / agent / "temp_0.0" / source.name
            )
            relative_symlink(source, destination)

    expected = {agent: 30 for agent in AGENTS}
    if counts != expected or len(pxd_ids) != 30:
        raise RuntimeError(f"incomplete standalone set: counts={counts}, PXDs={len(pxd_ids)}")

    manifest = {
        "purpose": "Legacy-layout adapter for the frozen standalone Qwen test-30 outputs",
        "source": str(SOURCE.resolve()),
        "destination": str(DESTINATION.resolve()),
        "pxd_count": len(pxd_ids),
        "agent_counts": counts,
        "pxd_ids": sorted(pxd_ids),
    }
    manifest_path = EXPERIMENT / "inputs/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

