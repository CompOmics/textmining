#!/usr/bin/env python3
"""
Setup Benchmark Data
====================
Copies CleanText manuscript files and their corresponding SDRF files from
the Intelligent-metadata-compilation repo into a local benchmark_data folder,
organized by PXD identifier.

Output structure:
    benchmark_data/
    ├── manuscripts/          # CleanText .txt files (one per PXD)
    ├── sdrfs/                # Corresponding SDRF .tsv files
    ├── matched/              # Only PXDs that have BOTH manuscript + SDRF
    │   ├── <PXD>/
    │   │   ├── manuscript.txt
    │   │   └── <PXD>.sdrf.tsv
    │   └── ...
    └── summary.json          # Mapping report
"""

import os
import re
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────── Paths ────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent  # benchmark_data/

# CleanText sources (relative to repo base)
CLEANTEXT_SUBDIRS = [
    "Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/CleanText/Training",
    "Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/CleanText/Unseen",
]

# SDRF sources (ordered by priority — first match wins)
SDRF_SUBDIRS = [
    "Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/GoldStandard_SDRFs",
    "NLP_metadata_extraction/NLP_Trainingset_annotation/data/SDRF",
    "NLP_metadata_extraction/NLP_Trainingset_annotation/data/origSDRF",
    "Hackathons_and_challenges/ISMB_collaboration_fest_2025/data/BenchmarkAnnotations/DummySDRFs",
]


def extract_pxd(filename: str) -> str | None:
    """Pull the PXD identifier out of a filename."""
    m = re.search(r"(PXD\d+)", filename)
    return m.group(1) if m else None


def collect_cleantext(repo_base: Path) -> dict[str, list[Path]]:
    """Return {PXD: [list of CleanText file paths]}."""
    pxd_texts: dict[str, list[Path]] = defaultdict(list)
    for subdir in CLEANTEXT_SUBDIRS:
        d = repo_base / subdir
        if not d.is_dir():
            print(f"  [WARN] CleanText dir not found: {d}")
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix == ".txt":
                pxd = extract_pxd(f.name)
                if pxd:
                    pxd_texts[pxd].append(f)
    return dict(pxd_texts)


def collect_sdrfs(repo_base: Path) -> dict[str, Path]:
    """Return {PXD: best SDRF path} — first directory match wins."""
    pxd_sdrfs: dict[str, Path] = {}
    for subdir in SDRF_SUBDIRS:
        d = repo_base / subdir
        if not d.is_dir():
            print(f"  [WARN] SDRF dir not found: {d}")
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and ".sdrf." in f.name:
                pxd = extract_pxd(f.name)
                if pxd and pxd not in pxd_sdrfs:
                    pxd_sdrfs[pxd] = f
    return pxd_sdrfs


def main():
    parser = argparse.ArgumentParser(description="Setup benchmark data from annotation repo")
    parser.add_argument("--repo-base", type=str, required=True,
                        help="Path to the Intelligent-metadata-compilation repository")
    args = parser.parse_args()
    repo_base = Path(args.repo_base)

    print("=" * 60)
    print("  Benchmark Data Setup")
    print("=" * 60)

    # Output sub-directories
    manuscripts_dir = OUTPUT_DIR / "manuscripts"
    sdrfs_dir = OUTPUT_DIR / "sdrfs"
    matched_dir = OUTPUT_DIR / "matched"
    for d in [manuscripts_dir, sdrfs_dir, matched_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Collect sources ──
    print("\n[1/4] Collecting CleanText manuscripts...")
    pxd_texts = collect_cleantext(repo_base)
    print(f"       Found {len(pxd_texts)} unique PXDs with CleanText")

    print("[2/4] Collecting SDRF files...")
    pxd_sdrfs = collect_sdrfs(repo_base)
    print(f"       Found {len(pxd_sdrfs)} unique PXDs with SDRF")

    # ── Copy manuscripts ──
    print("[3/4] Copying files...")
    copied_manuscripts = 0
    copied_sdrfs = 0
    matched_count = 0
    manuscript_only = 0
    sdrf_only = 0

    all_pxds = sorted(set(pxd_texts.keys()) | set(pxd_sdrfs.keys()))

    summary = {
        "total_unique_pxds": len(all_pxds),
        "matched": [],
        "manuscript_only": [],
        "sdrf_only": [],
    }

    for pxd in all_pxds:
        has_text = pxd in pxd_texts
        has_sdrf = pxd in pxd_sdrfs

        # Copy manuscript(s) — pick the first (longest) if multiple exist
        if has_text:
            # Sort by file size descending, take the largest (most complete)
            texts_sorted = sorted(pxd_texts[pxd], key=lambda p: p.stat().st_size, reverse=True)
            src_text = texts_sorted[0]
            dst_text = manuscripts_dir / f"{pxd}_manuscript.txt"
            shutil.copy2(src_text, dst_text)
            copied_manuscripts += 1

        # Copy SDRF
        if has_sdrf:
            src_sdrf = pxd_sdrfs[pxd]
            dst_sdrf = sdrfs_dir / f"{pxd}.sdrf.tsv"
            shutil.copy2(src_sdrf, dst_sdrf)
            copied_sdrfs += 1

        # Build matched directory (only PXDs with BOTH)
        if has_text and has_sdrf:
            pxd_dir = matched_dir / pxd
            pxd_dir.mkdir(exist_ok=True)
            shutil.copy2(src_text, pxd_dir / "manuscript.txt")
            shutil.copy2(src_sdrf, pxd_dir / f"{pxd}.sdrf.tsv")
            matched_count += 1
            summary["matched"].append(pxd)
        elif has_text:
            manuscript_only += 1
            summary["manuscript_only"].append(pxd)
        else:
            sdrf_only += 1
            summary["sdrf_only"].append(pxd)

    # ── Report ──
    print("[4/4] Writing summary...\n")

    summary["counts"] = {
        "matched": matched_count,
        "manuscript_only": manuscript_only,
        "sdrf_only": sdrf_only,
        "total_manuscripts_copied": copied_manuscripts,
        "total_sdrfs_copied": copied_sdrfs,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"  Manuscripts copied:  {copied_manuscripts}")
    print(f"  SDRFs copied:        {copied_sdrfs}")
    print(f"  Matched (both):      {matched_count}")
    print(f"  Manuscript only:     {manuscript_only}")
    print(f"  SDRF only:           {sdrf_only}")
    print(f"  Summary written to:  {summary_path}")
    print(f"\n  Output directory:    {OUTPUT_DIR}")
    print(f"    manuscripts/       — flat folder of all manuscript .txt")
    print(f"    sdrfs/             — flat folder of all .sdrf.tsv")
    print(f"    matched/           — per-PXD folders with both files")
    print("=" * 60)


if __name__ == "__main__":
    main()
