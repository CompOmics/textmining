#!/usr/bin/env python3
"""Build a DocETL corpus containing only abstract and methods text."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BOUNDARIES = {
    "PRIDE PROJECT PROPERTIES:",
    "ABSTRACT:",
    "PROJECT DESCRIPTION:",
    "SAMPLE PROCESSING:",
    "DATA PROCESSING:",
    "ABSTRACT (PubMed):",
    "METHODS:",
    "RESULTS:",
    "INTRODUCTION:",
    "SUPPLEMENTARY:",
    "SDRF SUMMARY:",
}

MISSING_TEXT_VALUES = {
    "",
    "n/a",
    "na",
    "none",
    "not available",
    "not provided",
    "unknown",
}


def available_text(value: str) -> str:
    """Return depositor text, or an empty string for explicit missing markers."""
    value = value.strip()
    if value.casefold().rstrip(".") in MISSING_TEXT_VALUES:
        return ""
    return value


def section(lines: list[str], heading: str) -> str:
    """Return text after an exact generated heading up to the next boundary."""
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return ""
    end = start
    while end < len(lines) and lines[end] not in BOUNDARIES:
        end += 1
    return "\n".join(lines[start:end]).strip()


def prepare(
    source: Path,
    destination: Path,
    *,
    include_pride_always: bool = False,
    include_pride_protocols: bool = False,
) -> dict[str, object]:
    raw = source.read_text(encoding="utf-8", errors="replace")
    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").split("\n")]

    pubmed_abstract = section(lines, "ABSTRACT (PubMed):")
    pride_abstract = section(lines, "ABSTRACT:")
    abstract = pubmed_abstract or pride_abstract
    abstract_source = "pubmed" if pubmed_abstract else "pride" if pride_abstract else "missing"
    methods = section(lines, "METHODS:")
    pride_properties = section(lines, "PRIDE PROJECT PROPERTIES:")
    pride_sample_processing_raw = section(lines, "SAMPLE PROCESSING:")
    pride_data_processing_raw = section(lines, "DATA PROCESSING:")
    pride_sample_processing = available_text(pride_sample_processing_raw)
    pride_data_processing = available_text(pride_data_processing_raw)
    use_pride_fallback = not methods and bool(pride_properties)
    include_pride = bool(pride_properties) and (include_pride_always or use_pride_fallback)

    parts = []
    if abstract:
        parts.append(f"=== ABSTRACT ===\n{abstract}")
    if methods:
        parts.append(f"=== MATERIALS AND METHODS ===\n{methods}")
    if include_pride:
        parts.append(f"=== PRIDE PROJECT PROPERTIES ===\n{pride_properties}")
    if include_pride_protocols and pride_sample_processing:
        parts.append(
            f"=== PRIDE SAMPLE PROCESSING PROTOCOL ===\n{pride_sample_processing}"
        )
    if include_pride_protocols and pride_data_processing:
        parts.append(
            f"=== PRIDE DATA PROCESSING PROTOCOL ===\n{pride_data_processing}"
        )
    destination.write_text("\n\n".join(parts) + "\n", encoding="utf-8")

    return {
        "pxd": source.stem,
        "source_file": str(source.resolve()),
        "output_file": str(destination.resolve()),
        "abstract_source": abstract_source,
        "has_abstract": bool(abstract),
        "has_methods": bool(methods),
        "has_pride_properties": bool(pride_properties),
        "includes_pride_properties": include_pride,
        "has_pride_properties_fallback": use_pride_fallback,
        "abstract_chars": len(abstract),
        "methods_chars": len(methods),
        "pride_properties_chars": len(pride_properties) if include_pride else 0,
        "has_pride_sample_processing": bool(pride_sample_processing),
        "pride_sample_processing_status": (
            "available" if pride_sample_processing else "unavailable"
        ),
        "includes_pride_sample_processing": bool(
            include_pride_protocols and pride_sample_processing
        ),
        "pride_sample_processing_chars": (
            len(pride_sample_processing) if include_pride_protocols else 0
        ),
        "has_pride_data_processing": bool(pride_data_processing),
        "pride_data_processing_status": (
            "available" if pride_data_processing else "unavailable"
        ),
        "includes_pride_data_processing": bool(
            include_pride_protocols and pride_data_processing
        ),
        "pride_data_processing_chars": (
            len(pride_data_processing) if include_pride_protocols else 0
        ),
        "output_chars": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-pride-always",
        action="store_true",
        help="Include PRIDE project properties even when manuscript Methods are present",
    )
    parser.add_argument(
        "--include-pride-protocols",
        action="store_true",
        help="Include PRIDE sample- and data-processing protocol text",
    )
    args = parser.parse_args()

    sources = sorted(args.input.glob("*.txt"))
    if not sources:
        raise SystemExit(f"No .txt files found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    existing = list(args.output.glob("*.txt"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"Refusing to mix with {len(existing)} existing .txt files in {args.output}"
        )

    rows = [
        prepare(
            src,
            args.output / src.name,
            include_pride_always=args.include_pride_always,
            include_pride_protocols=args.include_pride_protocols,
        )
        for src in sources
    ]
    manifest = args.output / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} manuscripts in {args.output}")
    print(f"Abstracts: {sum(bool(row['has_abstract']) for row in rows)}")
    print(f"Methods: {sum(bool(row['has_methods']) for row in rows)}")
    print(
        "PRIDE properties included: "
        f"{sum(bool(row['includes_pride_properties']) for row in rows)}"
    )
    print(
        "PRIDE properties fallback: "
        f"{sum(bool(row['has_pride_properties_fallback']) for row in rows)}"
    )
    print(
        "PRIDE sample-processing protocols included: "
        f"{sum(bool(row['includes_pride_sample_processing']) for row in rows)}"
    )
    print(
        "PRIDE data-processing protocols included: "
        f"{sum(bool(row['includes_pride_data_processing']) for row in rows)}"
    )
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
