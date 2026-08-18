"""Fetch PRIDE Archive v2 project metadata for a fixed list of PXD accessions.

Unlike pride_census/pride_survey.py (a full-repository crawl over all ~39,000
PRIDE projects, and dependent on pride_client/pmc_client/llm_client modules
that are not present in this working directory), this script queries the
public PRIDE Archive REST API directly, one accession at a time, for exactly
the PXDs given on the command line.

Usage:
    python scrape_pride_metadata.py --pxd-list agentic_and_eubic_pxds.txt \
        --outfile pride_metadata_agentic_and_eubic.csv
"""
import argparse
import csv
import time

import requests

API_ROOT = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"


def read_pxd_list(path):
    pxds = []
    with open(path, newline="") as handle:
        for line in handle:
            line = line.strip().strip(",")
            if not line or line.upper() in {"PXD", "PXDS", "ACCESSION"}:
                continue
            pxds.append(line.split(",")[0])
    # de-duplicate, keep order
    seen = set()
    unique = []
    for pxd in pxds:
        if pxd not in seen:
            seen.add(pxd)
            unique.append(pxd)
    return unique


def fetch_project(accession, session, timeout=20):
    resp = session.get(f"{API_ROOT}/{accession}", timeout=timeout)
    if resp.status_code != 200:
        return None
    return resp.json()


def fetch_raw_file_count(accession, session, page_size=100, timeout=20, max_pages=20):
    raw_count = 0
    total = None
    page = 0
    while True:
        resp = session.get(
            f"{API_ROOT}/{accession}/files",
            params={"page": page, "pageSize": page_size},
            timeout=timeout,
        )
        if resp.status_code != 200:
            break
        if total is None:
            total = int(resp.headers.get("total_records", 0))
        files = resp.json()
        if not files:
            break
        for entry in files:
            filename = entry.get("fileName", "")
            if filename.lower().endswith(".raw"):
                raw_count += 1
        page += 1
        if total is not None and page * page_size >= total:
            break
        if page >= max_pages:
            break
    return raw_count, (total or 0)


def row_for_project(accession, project, raw_files, total_files):
    references = project.get("references", []) or []
    pubmed_id = references[0].get("pubmedID", "") if references else ""
    doi = project.get("doi", "")
    organisms = "; ".join(o.get("name", "") for o in project.get("organisms", []) or [])
    organism_parts = "; ".join(o.get("name", "") for o in project.get("organismParts", []) or [])
    diseases = "; ".join(d.get("name", "") for d in project.get("diseases", []) or [])
    instruments = "; ".join(i.get("name", "") for i in project.get("instruments", []) or [])
    experiment_types = "; ".join(e.get("name", "") for e in project.get("experimentTypes", []) or [])
    quant_methods = "; ".join(q.get("name", "") for q in project.get("quantificationMethods", []) or [])
    ptm_entries = project.get("identifiedPTMStrings", []) or []
    ptms = "; ".join(p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in ptm_entries)
    license_raw = project.get("license")
    license_name = license_raw.get("name", "") if isinstance(license_raw, dict) else (license_raw or "")

    return {
        "accession": accession,
        "title": project.get("title", ""),
        "pubmed_id": pubmed_id,
        "doi": doi,
        "license": license_name,
        "submission_date": project.get("submissionDate", ""),
        "publication_date": project.get("publicationDate", ""),
        "organisms": organisms,
        "organism_parts": organism_parts,
        "diseases": diseases,
        "instruments": instruments,
        "experiment_types": experiment_types,
        "quantification_methods": quant_methods,
        "identified_ptms": ptms,
        "raw_file_count": raw_files,
        "total_file_count": total_files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pxd-list", required=True, help="Text or CSV file with one PXD accession per line")
    parser.add_argument("--outfile", default="pride_metadata_scraped.csv")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between accessions, seconds")
    parser.add_argument("--skip-file-counts", action="store_true", help="Skip the per-project /files pagination (faster, no raw-file count)")
    args = parser.parse_args()

    pxds = read_pxd_list(args.pxd_list)
    print(f"Loaded {len(pxds)} unique accessions from {args.pxd_list}")

    session = requests.Session()
    rows = []
    not_found = []

    for i, accession in enumerate(pxds, start=1):
        print(f"[{i}/{len(pxds)}] {accession}", end=" ")
        project = fetch_project(accession, session)
        if project is None:
            print("-> not found / error")
            not_found.append(accession)
            time.sleep(args.sleep)
            continue

        if args.skip_file_counts:
            raw_files, total_files = "", ""
        else:
            raw_files, total_files = fetch_raw_file_count(accession, session)

        rows.append(row_for_project(accession, project, raw_files, total_files))
        print("-> ok")
        time.sleep(args.sleep)

    with open(args.outfile, "w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.outfile}")
    if not_found:
        print(f"{len(not_found)} accessions returned no project (private, withdrawn, or typo): {not_found}")


if __name__ == "__main__":
    main()
