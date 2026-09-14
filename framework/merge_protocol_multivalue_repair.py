#!/usr/bin/env python3
"""Validate one staged repair task and atomically add its missing outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


HAMLET = Path(__file__).resolve().parents[2]
REPAIR_ROOT = HAMLET / "textmining/repair_inputs/qwen3_8_27b_protocols_multivalue_1737_missing"


def load_task(task_id: int) -> dict[str, str]:
    with (REPAIR_ROOT / "task_manifest.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    matches = [row for row in rows if int(row["task_id"]) == task_id]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for task {task_id}, found {len(matches)}")
    return matches[0]


def validated_file(directory: Path, pxd: str) -> Path:
    matches = list(directory.glob(f"{pxd}_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one staged file for {pxd} under {directory}, found {len(matches)}")
    payload = json.loads(matches[0].read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid JSON object: {matches[0]}")
    return matches[0]


def install_missing(source: Path, destination: Path, task_id: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        json.loads(destination.read_text())
        return "already_present"
    temporary = destination.with_name(f".{destination.name}.repair-{task_id}.tmp")
    shutil.copy2(source, temporary)
    json.loads(temporary.read_text())
    os.replace(temporary, destination)
    return "copied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    task = load_task(args.task_id)
    staging = Path(task["staging_dir"])
    destination = Path(task["destination_dir"])
    agent_dir = task["agent_dir"]
    pxds = task["pxd_ids"].split(",")
    results = []
    for pxd in pxds:
        raw = validated_file(staging / agent_dir, pxd)
        normalized = validated_file(staging / "NormalizedAgent" / agent_dir, pxd)
        raw_state = install_missing(raw, destination / agent_dir / raw.name, args.task_id)
        normalized_state = install_missing(
            normalized, destination / "NormalizedAgent" / agent_dir / normalized.name, args.task_id
        )
        results.append({"pxd": pxd, "raw": raw_state, "normalized": normalized_state})

    report = {"task": task, "records": results, "status": "merged"}
    status_dir = destination.parent / "repair_status_protocols_multivalue"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / f"merge-task_{args.task_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

