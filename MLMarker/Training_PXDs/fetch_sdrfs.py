"""Fetch SDRF files for the training PXDs, checking two sources:
  1. PRIDE itself (a project's submitted files sometimes include a .sdrf.tsv)
  2. bigbio/sdrf-annotated-datasets on GitHub (community-curated SDRFs)

PRIDE is checked first since it's the primary/authoritative source; bigbio is
used as a fallback (and noted separately if both have one, in case they
differ). Saves to SDRFs/{PXD}.sdrf.tsv and a per-PXD source manifest.

Usage:
    python fetch_sdrfs.py
"""
import csv
import json
import os
import re
import time

import requests

PRIDE_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"
BIGBIO_TREE_API = "https://api.github.com/repos/bigbio/sdrf-annotated-datasets/git/trees/main?recursive=1"
BIGBIO_RAW = "https://raw.githubusercontent.com/bigbio/sdrf-annotated-datasets/main/{path}"

HERE = os.path.dirname(os.path.abspath(__file__))
LICENSE_REPORT = os.path.join(HERE, "license_report.csv")
SDRF_DIR = os.path.join(HERE, "SDRFs")
MANIFEST_OUT = os.path.join(HERE, "SDRFs_manifest.csv")
SLEEP = 0.2


def read_pxds():
    with open(LICENSE_REPORT, newline="") as handle:
        return [r["accession"] for r in csv.DictReader(handle)]


def index_bigbio_sdrfs(session):
    resp = session.get(BIGBIO_TREE_API, timeout=60)
    resp.raise_for_status()
    tree = resp.json()["tree"]
    pattern = re.compile(r"^datasets/(PXD\d+)/(.*\.sdrf\.tsv)$")
    index = {}
    for entry in tree:
        if entry["type"] != "blob":
            continue
        m = pattern.match(entry["path"])
        if m:
            index.setdefault(m.group(1), []).append(f"datasets/{m.group(1)}/{m.group(2)}")
    return index


def find_pride_sdrf_url(accession, session, timeout=20, retries=3):
    resp = None
    for attempt in range(retries):
        try:
            resp = session.get(f"{PRIDE_API}/{accession}/files", params={"page": 0, "pageSize": 200}, timeout=timeout)
            break
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(1.0)
    if resp is None or resp.status_code != 200:
        return None
    for entry in resp.json():
        filename = entry.get("fileName", "")
        if "sdrf" not in filename.lower():
            continue
        for loc in entry.get("publicFileLocations", []) or []:
            if loc.get("name") == "FTP Protocol":
                return loc.get("value")
    return None


def main():
    pxds = read_pxds()
    os.makedirs(SDRF_DIR, exist_ok=True)
    session = requests.Session()

    print("Indexing bigbio/sdrf-annotated-datasets...")
    bigbio_index = index_bigbio_sdrfs(session)
    print(f"{len(bigbio_index)} PXDs have an SDRF in bigbio\n")

    manifest = []
    for i, accession in enumerate(pxds, start=1):
        print(f"[{i}/{len(pxds)}] {accession}", end=" ")
        row = {"accession": accession, "source": "", "path": "", "note": ""}

        pride_url = find_pride_sdrf_url(accession, session)
        if pride_url:
            try:
                r = session.get(pride_url, timeout=30)
                if r.status_code == 200 and r.content:
                    outpath = os.path.join(SDRF_DIR, f"{accession}.sdrf.tsv")
                    with open(outpath, "wb") as handle:
                        handle.write(r.content)
                    row["source"] = "PRIDE"
                    row["path"] = os.path.relpath(outpath, HERE)
                    if accession in bigbio_index:
                        row["note"] = "also present in bigbio"
                    print("-> PRIDE")
                    manifest.append(row)
                    time.sleep(SLEEP)
                    continue
            except requests.RequestException:
                pass

        if accession in bigbio_index:
            path = bigbio_index[accession][0]
            r = session.get(BIGBIO_RAW.format(path=path), timeout=30)
            if r.status_code == 200 and r.content:
                outpath = os.path.join(SDRF_DIR, f"{accession}.sdrf.tsv")
                with open(outpath, "wb") as handle:
                    handle.write(r.content)
                row["source"] = "bigbio"
                row["path"] = os.path.relpath(outpath, HERE)
                print("-> bigbio")
                manifest.append(row)
                time.sleep(SLEEP)
                continue

        row["note"] = "no SDRF found in PRIDE or bigbio"
        print("-> none found")
        manifest.append(row)
        time.sleep(SLEEP)

    with open(MANIFEST_OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "source", "path", "note"])
        writer.writeheader()
        writer.writerows(manifest)

    n_pride = sum(1 for r in manifest if r["source"] == "PRIDE")
    n_bigbio = sum(1 for r in manifest if r["source"] == "bigbio")
    n_none = sum(1 for r in manifest if not r["source"])
    print(f"\n{n_pride} from PRIDE, {n_bigbio} from bigbio, {n_none} not found (of {len(pxds)})")
    print(f"Manifest saved to {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
