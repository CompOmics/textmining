#!/usr/bin/env python3
"""Build minimal per-agent repair inputs for the incomplete production shards."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


HAMLET = Path(__file__).resolve().parents[2]
INPUT_ROOT = HAMLET / "textmining/production_inputs/abstract_methods_pride_protocols_1737_8shards"
OUTPUT_ROOT = HAMLET / "textmining/framework/production_outputs/qwen3_8_27b_abstract_methods_pride_protocols_multivalue_1737"
REPAIR_ROOT = HAMLET / "textmining/repair_inputs/qwen3_8_27b_protocols_multivalue_1737_missing"
STAGING_ROOT = HAMLET / "textmining/framework/repair_outputs/qwen3_8_27b_protocols_multivalue_1737"
AGENTS = {
    "biological": "BiologicalAgent",
    "technical": "TechnicalAgent",
    "experimental": "ExperimentalDesignAgent",
}


def present_pxds(directory: Path) -> set[str]:
    return {path.name.split("_", 1)[0] for path in directory.glob("PXD*.json")}


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
    tasks = []
    for shard_dir in sorted(INPUT_ROOT.glob("shard_*"), key=lambda p: int(p.name.split("_")[-1])):
        shard = int(shard_dir.name.split("_")[-1])
        sources = {path.stem: path for path in shard_dir.glob("PXD*.txt")}
        for agent_key, agent_dir in AGENTS.items():
            raw = present_pxds(OUTPUT_ROOT / shard_dir.name / agent_dir)
            normalized = present_pxds(OUTPUT_ROOT / shard_dir.name / "NormalizedAgent" / agent_dir)
            missing = sorted(set(sources) - (raw & normalized))
            if not missing:
                continue
            task_id = len(tasks)
            task_input = REPAIR_ROOT / f"task_{task_id}"
            for pxd in missing:
                relative_symlink(sources[pxd], task_input / f"{pxd}.txt")
            tasks.append({
                "task_id": task_id,
                "original_shard": shard,
                "agent": agent_key,
                "agent_dir": agent_dir,
                "expected": len(missing),
                "input_dir": str(task_input.resolve()),
                "staging_dir": str((STAGING_ROOT / f"task_{task_id}").resolve()),
                "destination_dir": str((OUTPUT_ROOT / shard_dir.name).resolve()),
                "pxd_ids": ",".join(missing),
            })

    if len(tasks) != 4 or sum(task["expected"] for task in tasks) != 41:
        raise RuntimeError(
            f"unexpected repair scope: tasks={len(tasks)}, agent-records={sum(t['expected'] for t in tasks)}"
        )
    REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = REPAIR_ROOT / "task_manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(tasks)
    summary = {
        "input_root": str(INPUT_ROOT.resolve()),
        "production_output": str(OUTPUT_ROOT.resolve()),
        "staging_root": str(STAGING_ROOT.resolve()),
        "task_count": len(tasks),
        "missing_agent_records": sum(task["expected"] for task in tasks),
        "distinct_pxds": len({pxd for task in tasks for pxd in task["pxd_ids"].split(",")}),
        "tasks": tasks,
    }
    (REPAIR_ROOT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

