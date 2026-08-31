#!/usr/bin/env python3
"""Audit multi-value DocETL outputs against their prepared source texts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


AGENT_DIRS = (
    "BiologicalAgent",
    "TechnicalAgent",
    "ExperimentalDesignAgent",
)
META_KEYS = {"_confidence", "_hallucination_flags", "_evidence_grounding_flags"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    sources = {path.stem: path for path in args.input.glob("*.txt")}
    problems: list[str] = []
    stats = Counter()
    multi_fields = Counter()

    for agent in AGENT_DIRS:
        raw_files = sorted((args.output / agent).glob("*.json"))
        normalized_dir = args.output / "NormalizedAgent" / agent
        stats[f"raw_{agent}"] = len(raw_files)
        stats[f"normalized_{agent}"] = len(list(normalized_dir.glob("*.json")))

        for raw_path in raw_files:
            pxd = raw_path.name.split("_", 1)[0]
            source_path = sources.get(pxd)
            if source_path is None:
                problems.append(f"{raw_path}: no source input")
                continue
            source = source_path.read_text(encoding="utf-8", errors="replace")
            source_lower = source.lower()
            raw = json.loads(raw_path.read_text())
            grounding_flags = raw.get("_evidence_grounding_flags", [])
            if isinstance(grounding_flags, list):
                stats["evidence_grounding_flags"] += len(grounding_flags)
            flagged_entries = {
                (item.get("field"), item.get("value"), item.get("evidence"))
                for item in grounding_flags
                if isinstance(item, dict)
            }
            normalized_path = normalized_dir / raw_path.name
            if not normalized_path.is_file():
                problems.append(f"{normalized_path}: missing normalized output")
                normalized = {}
            else:
                normalized = json.loads(normalized_path.read_text())

            for field, entries in raw.items():
                if field in META_KEYS:
                    continue
                stats["fields"] += 1
                if not isinstance(entries, list) or not entries:
                    problems.append(f"{raw_path}:{field}: expected non-empty list")
                    continue
                if len(entries) > 1:
                    multi_fields[f"{agent}.{field}"] += 1
                normalized_entries = normalized.get(field)
                if not isinstance(normalized_entries, list):
                    problems.append(f"{normalized_path}:{field}: not a normalized list")
                elif len(normalized_entries) != len(entries):
                    problems.append(
                        f"{normalized_path}:{field}: raw={len(entries)} normalized={len(normalized_entries)}"
                    )

                for index, entry in enumerate(entries):
                    stats["entries"] += 1
                    if not isinstance(entry, dict):
                        problems.append(f"{raw_path}:{field}[{index}]: not an object")
                        continue
                    if set(entry) != {"value", "evidence"}:
                        problems.append(
                            f"{raw_path}:{field}[{index}]: keys={sorted(entry)}"
                        )
                    value = entry.get("value")
                    evidence = entry.get("evidence")
                    if not isinstance(value, str) or not isinstance(evidence, str):
                        problems.append(f"{raw_path}:{field}[{index}]: non-string member")
                        continue
                    if value.lower() == "unknown":
                        stats["unknown_entries"] += 1
                        if evidence:
                            problems.append(
                                f"{raw_path}:{field}[{index}]: unknown has evidence"
                            )
                        continue
                    stats["known_entries"] += 1
                    if not evidence:
                        if (field, value, evidence) in flagged_entries:
                            stats["flagged_missing_evidence"] += 1
                        else:
                            problems.append(f"{raw_path}:{field}[{index}]: missing evidence")
                        continue
                    if evidence.lower().startswith("inferred: "):
                        quote = evidence[len("inferred: "):]
                        if quote.lower() not in source_lower:
                            if (field, value, evidence) in flagged_entries:
                                stats["flagged_unsupported_entries"] += 1
                            else:
                                problems.append(
                                    f"{raw_path}:{field}[{index}]: inferred quote absent"
                                )
                    else:
                        if evidence.lower() not in source_lower:
                            if (field, value, evidence) in flagged_entries:
                                stats["flagged_unsupported_entries"] += 1
                            else:
                                problems.append(
                                    f"{raw_path}:{field}[{index}]: evidence absent from input"
                                )
                        if value.lower() not in evidence.lower():
                            stats["value_not_in_evidence"] += 1

    report = {
        "sources": len(sources),
        "stats": dict(stats),
        "multi_valued_documents_by_field": dict(sorted(multi_fields.items())),
        "problems": problems,
        "problem_count": len(problems),
    }
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
