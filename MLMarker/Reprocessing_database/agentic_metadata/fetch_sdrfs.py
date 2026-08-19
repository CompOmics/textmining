"""
Standalone SDRF fetcher — run on any machine with Python + pandas + requests.

Downloads all SDRFs for projects marked has_sdrf=True in the metadata,
saves them to sdrf_cache/ as TSV files.

Usage:
    python fetch_sdrfs.py

Copy the resulting sdrf_cache/ folder to:
    run_level_data/sdrf_cache/
on your main machine.
"""

import json
import os
import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent
META_DIR = SCRIPT_DIR / "results" / "manuscripts" / "metadata"
CACHE_DIR = SCRIPT_DIR / "cache" / "sdrf"
SLEEP_BETWEEN = 1  # seconds between requests


def fetch_sdrf(pxd_id: str, timeout: int = 240) -> pd.DataFrame | None:
    """Fetch SDRF table for a PXD accession from PRIDE."""
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v3/projects/{pxd_id}/files"
    page = 0
    sdrf_file = None

    while page < 50:
        try:
            resp = requests.get(url, params={"pageSize": 100, "page": page}, timeout=30)
            resp.raise_for_status()
            files = resp.json()
            if not files:
                break
            for f in files:
                name = f.get("fileName", "")
                if name.lower() == "sdrf.tsv" or name.lower().endswith(".sdrf.tsv"):
                    sdrf_file = f
                    break
            if sdrf_file or len(files) < 100:
                break
            page += 1
        except Exception as e:
            print(f"  {pxd_id}: file list error: {e}")
            break

    if not sdrf_file:
        return None

    download_url = None
    for loc in sdrf_file.get("publicFileLocations", []):
        u = loc.get("value", "")
        if u.startswith("ftp://"):
            download_url = u.replace(
                "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/",
                "https://ftp.pride.ebi.ac.uk/pride/data/archive/"
            )
            break
        elif u.startswith("http"):
            download_url = u
            break

    if not download_url:
        return None

    try:
        resp = requests.get(download_url, timeout=timeout)
        resp.raise_for_status()
        return pd.read_csv(StringIO(resp.text), sep="\t")
    except Exception as e:
        print(f"  {pxd_id}: SDRF download failed: {e}")
        return None


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Find projects with has_sdrf=True
    sdrf_pxds = []
    for fname in sorted(os.listdir(META_DIR)):
        if not fname.endswith(".meta.json"):
            continue
        with open(META_DIR / fname) as fh:
            data = json.load(fh)
        if data.get("has_sdrf"):
            sdrf_pxds.append(fname.replace(".meta.json", ""))

    # Check which are already cached
    already_cached = {p.stem.replace(".sdrf", "") for p in CACHE_DIR.glob("*.sdrf.tsv")}
    to_fetch = [p for p in sdrf_pxds if p not in already_cached]

    print(f"Total projects with SDRF: {len(sdrf_pxds)}")
    print(f"Already cached: {len(already_cached)}")
    print(f"To fetch: {len(to_fetch)}")
    print()

    fetched = 0
    failed = 0
    for i, pxd_id in enumerate(to_fetch):
        sdrf_df = fetch_sdrf(pxd_id)
        if sdrf_df is not None:
            cache_path = CACHE_DIR / f"{pxd_id}.sdrf.tsv"
            sdrf_df.to_csv(cache_path, sep="\t", index=False)
            fetched += 1
            print(f"  [{i+1}/{len(to_fetch)}] {pxd_id}: {len(sdrf_df)} rows -> cached")
        else:
            failed += 1
            print(f"  [{i+1}/{len(to_fetch)}] {pxd_id}: failed")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone: {fetched} fetched, {failed} failed, {len(already_cached)} already cached")


if __name__ == "__main__":
    main()
