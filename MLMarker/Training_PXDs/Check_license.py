"""Check the open-access license of the publication associated with each PXD.

For every PXD accession: look up its linked PubMed ID via the PRIDE Archive
API, then look up that PubMed ID's open-access status and license via Europe
PMC (which reports the license itself, e.g. "cc by", rather than just a PMC
open-access flag). A dataset is flagged text-mining-permissive when its paper
carries a CC-BY, CC-BY-SA, or CC0 license.

Usage:
    python Check_license.py --pxd-file SupplementaryTableS1.tsv \
        --outfile license_report.csv
"""
import argparse
import csv
import re
import time

import requests

PRIDE_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"
EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

TEXT_MINING_PERMISSIVE = {"cc by", "cc by-sa", "cc0", "cc-by", "cc-by-sa"}

PXD_PATTERN = re.compile(r"PXD\d+")


def read_pxds(path):
    """Extract PXD accessions from a file, tolerant of whitespace-separated
    tables (like SupplementaryTableS1.tsv) as well as one-per-line lists."""
    with open(path) as handle:
        text = handle.read()
    pxds = PXD_PATTERN.findall(text)
    seen = set()
    unique = []
    for pxd in pxds:
        if pxd not in seen:
            seen.add(pxd)
            unique.append(pxd)
    return unique


def fetch_pubmed_id(accession, session, timeout=20):
    resp = session.get(f"{PRIDE_API}/{accession}", timeout=timeout)
    if resp.status_code != 200:
        return None, f"PRIDE lookup failed (HTTP {resp.status_code})"
    project = resp.json()
    references = project.get("references", []) or []
    if not references:
        return None, "no linked publication in PRIDE record"
    pubmed_id = references[0].get("pubmedID", "")
    if not pubmed_id:
        return None, "PRIDE record has no PubMed ID"
    return str(pubmed_id), None


def fetch_license(pubmed_id, session, timeout=20):
    params = {
        "query": f"EXT_ID:{pubmed_id} AND SRC:MED",
        "format": "json",
        "resultType": "core",
    }
    resp = session.get(EUROPEPMC_API, params=params, timeout=timeout)
    if resp.status_code != 200:
        return None
    results = resp.json().get("resultList", {}).get("result", [])
    if not results:
        return None
    return results[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pxd-file", default="SupplementaryTableS1.tsv")
    parser.add_argument("--outfile", default="license_report.csv")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    pxds = read_pxds(args.pxd_file)
    print(f"Loaded {len(pxds)} unique PXD accessions from {args.pxd_file}")

    session = requests.Session()
    rows = []

    for i, accession in enumerate(pxds, start=1):
        print(f"[{i}/{len(pxds)}] {accession}", end=" ")
        row = {
            "accession": accession,
            "pubmed_id": "",
            "pmcid": "",
            "doi": "",
            "is_open_access": "",
            "license": "",
            "text_mining_permissive": "",
            "note": "",
        }

        pubmed_id, err = fetch_pubmed_id(accession, session)
        if err:
            row["note"] = err
            rows.append(row)
            print(f"-> {err}")
            time.sleep(args.sleep)
            continue
        row["pubmed_id"] = pubmed_id

        result = fetch_license(pubmed_id, session)
        if result is None:
            row["note"] = "not found in Europe PMC"
            rows.append(row)
            print("-> not found in Europe PMC")
            time.sleep(args.sleep)
            continue

        license_name = (result.get("license") or "").strip()
        is_open_access = result.get("isOpenAccess", "")
        row["pmcid"] = result.get("pmcid", "")
        row["doi"] = result.get("doi", "")
        row["is_open_access"] = is_open_access
        row["license"] = license_name
        row["text_mining_permissive"] = license_name.lower() in TEXT_MINING_PERMISSIVE
        if not license_name:
            row["note"] = (
                "open access but no license string recorded"
                if is_open_access == "Y"
                else "closed access (no license applies)"
            )
        rows.append(row)
        print(f"-> {license_name or '(no license recorded)'}")
        time.sleep(args.sleep)

    with open(args.outfile, "w", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    permissive = sum(1 for r in rows if r["text_mining_permissive"] is True)
    print(f"\nSaved {len(rows)} rows to {args.outfile}")
    print(f"Text-mining-permissive (CC-BY/CC-BY-SA/CC0): {permissive}/{len(rows)}")


if __name__ == "__main__":
    main()
