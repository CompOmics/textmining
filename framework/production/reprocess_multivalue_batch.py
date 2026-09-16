#!/usr/bin/env python3
"""Replay deterministic multi-value grounding on completed batch outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
FRAMEWORK = PROJECT / "framework"
sys.path.insert(0, str(FRAMEWORK))

from docetl_pipeline.run_docetl import _ground_multivalue_evidence  # noqa: E402


AGENTS = ("BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    if args.destination.exists():
        raise SystemExit(f"Refusing to overwrite existing destination: {args.destination}")
    shutil.copytree(args.source_output, args.destination, symlinks=True)

    sources = {path.stem: path for path in args.input.glob("*.txt")}
    rewritten = 0
    for agent in AGENTS:
        for raw_path in sorted((args.destination / agent).glob("*.json")):
            pxd = raw_path.name.split("_", 1)[0]
            source = sources[pxd].read_text(encoding="utf-8", errors="replace")
            raw = json.loads(raw_path.read_text())
            _ground_multivalue_evidence(raw, source)
            raw_path.write_text(json.dumps(raw, indent=2) + "\n")

            normalized_path = args.destination / "NormalizedAgent" / agent / raw_path.name
            normalized = json.loads(normalized_path.read_text())
            for field, entries in raw.items():
                if field.startswith("_") or not isinstance(entries, list):
                    continue
                if normalized.get(field) == [] and entries == [
                    {"value": "unknown", "evidence": ""}
                ]:
                    normalized[field] = entries
            normalized_path.write_text(json.dumps(normalized, indent=2) + "\n")
            rewritten += 1

    print(f"reprocessed_files={rewritten}")


if __name__ == "__main__":
    main()
