#!/usr/bin/env python3
"""Create immutable Title+Abstract+Methods inputs and a checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EQ_HEADING = re.compile(r"^\s*===\s*([A-Z][A-Z &/\-]+?)\s*===\s*$")
COLON_HEADING = re.compile(r"^\s*([A-Z][A-Z &/\-]{2,}):\s*$")
KEEP = {"TITLE", "ABSTRACT", "METHODS", "MATERIALS AND METHODS", "MATERIALS & METHODS"}


def section_name(line: str) -> str | None:
    for pattern in (EQ_HEADING, COLON_HEADING):
        match = pattern.match(line)
        if match:
            return re.sub(r"\s+", " ", match.group(1).strip())
    return None


def restrict_text(text: str) -> tuple[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        heading = section_name(line)
        if heading is not None:
            current = heading
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)

    chosen: list[tuple[str, list[str]]] = []
    for canonical, aliases in (
        ("TITLE", ("TITLE",)),
        ("ABSTRACT", ("ABSTRACT",)),
        ("METHODS", ("METHODS", "MATERIALS AND METHODS", "MATERIALS & METHODS")),
    ):
        body: list[str] | None = None
        for alias in aliases:
            if alias in sections and any(x.strip() for x in sections[alias]):
                body = sections[alias]
                break
        if body is not None:
            chosen.append((canonical, body))

    if not sections:
        # Some benchmark records were supplied already restricted to abstract
        # and methods but without headings. Preserve those bytes as one frozen
        # pre-restricted block; do not attempt paragraph-level semantic guesses.
        stripped = text.strip()
        if not stripped:
            raise ValueError("empty unheaded manuscript")
        return stripped + "\n", ["PRE_RESTRICTED_UNHEADED"]
    if "ABSTRACT" not in {name for name, _ in chosen}:
        raise ValueError("headed manuscript is missing a non-empty ABSTRACT section")

    output = "\n\n".join(
        f"=== {name} ===\n" + "\n".join(lines).strip()
        for name, lines in chosen
    ).strip() + "\n"
    return output, [name for name, _ in chosen]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--pxd", action="append", default=[], help="Optional PXD allow-list")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manuscripts = sorted(args.source.glob("PXD*/manuscript.txt"))
    if args.pxd:
        wanted = set(args.pxd)
        manuscripts = [path for path in manuscripts if path.parent.name in wanted]
    if len(manuscripts) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} manuscripts, found {len(manuscripts)}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for source in manuscripts:
        pxd = source.parent.name
        restricted, sections = restrict_text(source.read_text(encoding="utf-8", errors="replace"))
        target_dir = args.output / pxd
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "manuscript.txt"
        target.write_text(restricted, encoding="utf-8")
        records.append({
            "pxd": pxd,
            "sections": sections,
            "source": str(source.resolve()),
            "source_sha256": sha256_bytes(source.read_bytes()),
            "restricted_sha256": sha256_bytes(restricted.encode()),
            "restricted_bytes": len(restricted.encode()),
        })

    aggregate = sha256_bytes("".join(f"{r['pxd']}\t{r['restricted_sha256']}\n" for r in records).encode())
    manifest = {
        "section_policy": "TITLE+ABSTRACT+METHODS; headings retained; original text otherwise unchanged",
        "source_root": str(args.source.resolve()),
        "count": len(records),
        "aggregate_sha256": aggregate,
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"count": len(records), "aggregate_sha256": aggregate, "output": str(args.output)}))


if __name__ == "__main__":
    main()
