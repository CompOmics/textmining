#!/usr/bin/env python3
"""Build agent-specific repair inputs for incomplete sharded production output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


AGENTS = {
    "biological": "BiologicalAgent",
    "technical": "TechnicalAgent",
    "experimental": "ExperimentalDesignAgent",
}


def output_ids(directory: Path) -> set[str]:
    return {path.name.split("_", 1)[0] for path in directory.glob("*.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    args = parser.parse_args()

    if args.repair_root.exists() and any(args.repair_root.iterdir()):
        raise SystemExit(f"Refusing non-empty repair root: {args.repair_root}")
    args.repair_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    task_id = 0
    for shard_dir in sorted(args.input_root.glob("shard_*")):
        shard = int(shard_dir.name.removeprefix("shard_"))
        expected = {
            path.stem: path.resolve()
            for path in shard_dir.glob("*.txt")
        }
        production_shard = args.output_root / shard_dir.name
        for agent, output_dir_name in AGENTS.items():
            present = output_ids(production_shard / output_dir_name)
            missing = sorted(set(expected) - present)
            if not missing:
                continue

            task_dir = args.repair_root / f"task_{task_id}"
            task_dir.mkdir()
            for pxd in missing:
                (task_dir / f"{pxd}.txt").symlink_to(expected[pxd])

            rows.append(
                {
                    "task_id": task_id,
                    "original_shard": shard,
                    "agent": agent,
                    "output_dir_name": output_dir_name,
                    "expected": len(missing),
                    "input_dir": str(task_dir.resolve()),
                    "output_dir": str(production_shard.resolve()),
                    "pxd_ids": ",".join(missing),
                }
            )
            task_id += 1

    if not rows:
        raise SystemExit("No missing agent outputs found")

    manifest = args.repair_root / "task_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} repair tasks for {sum(int(r['expected']) for r in rows)} outputs")
    for row in rows:
        print(
            f"task_{row['task_id']}: shard={row['original_shard']} "
            f"agent={row['agent']} papers={row['expected']}"
        )
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
