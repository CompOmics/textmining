#!/usr/bin/env python3
"""Create deterministic round-robin symlink shards for a text corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be positive")
    sources = sorted(args.input.glob("*.txt"))
    if not sources:
        raise SystemExit(f"No .txt files found in {args.input}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to mix with existing shard content: {args.output}")

    counts = [0] * args.shards
    for index in range(args.shards):
        (args.output / f"shard_{index}").mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources):
        shard = index % args.shards
        destination = args.output / f"shard_{shard}" / source.name
        destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))
        counts[shard] += 1

    manifest = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "total": len(sources),
        "shards": args.shards,
        "counts": counts,
        "assignment": "sorted filenames, round-robin",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
