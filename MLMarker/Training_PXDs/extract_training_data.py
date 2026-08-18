"""Build the training-data package for MLMarker/Training_PXDs.

Two things, for every PXD in SupplementaryTableS1.tsv:
  1. PRIDE project metadata (title, organism, instrument, disease, etc.) for
     ALL PXDs -> pride_metadata.csv
  2. Full manuscript text, extracted from the Europe PMC JATS full-text XML,
     for OPEN-ACCESS PXDs only (is_open_access == 'Y' in license_report.csv)
     -> manuscripts/{PXD}_manuscript.txt

Requires license_report.csv (from Check_license.py) to already exist in this
directory, since that's where the open-access flag and PMCID come from.

Usage:
    python extract_training_data.py
"""
import csv
import html
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

PRIDE_API = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"
EUROPEPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

HERE = os.path.dirname(os.path.abspath(__file__))
LICENSE_REPORT = os.path.join(HERE, "license_report.csv")
PRIDE_METADATA_OUT = os.path.join(HERE, "pride_metadata.csv")
MANUSCRIPTS_DIR = os.path.join(HERE, "manuscripts")
ABSTRACTS_DIR = os.path.join(HERE, "abstracts")
SLEEP = 0.2


def read_license_report():
    with open(LICENSE_REPORT, newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# 1. PRIDE project metadata, for every PXD
# --------------------------------------------------------------------------
def fetch_project(accession, session, timeout=20):
    resp = session.get(f"{PRIDE_API}/{accession}", timeout=timeout)
    if resp.status_code != 200:
        return None
    return resp.json()


def row_for_project(accession, project):
    references = project.get("references", []) or []
    pubmed_id = references[0].get("pubmedID", "") if references else ""
    organisms = "; ".join(o.get("name", "") for o in project.get("organisms", []) or [])
    organism_parts = "; ".join(o.get("name", "") for o in project.get("organismParts", []) or [])
    diseases = "; ".join(d.get("name", "") for d in project.get("diseases", []) or [])
    instruments = "; ".join(i.get("name", "") for i in project.get("instruments", []) or [])
    experiment_types = "; ".join(e.get("name", "") for e in project.get("experimentTypes", []) or [])
    quant_methods = "; ".join(q.get("name", "") for q in project.get("quantificationMethods", []) or [])
    ptm_entries = project.get("identifiedPTMStrings", []) or []
    ptms = "; ".join(p.get("name", str(p)) if isinstance(p, dict) else str(p) for p in ptm_entries)
    license_raw = project.get("license")
    pride_license = license_raw.get("name", "") if isinstance(license_raw, dict) else (license_raw or "")

    return {
        "accession": accession,
        "title": project.get("title", ""),
        "pubmed_id": pubmed_id,
        "doi": project.get("doi", ""),
        "pride_license": pride_license,
        "submission_date": project.get("submissionDate", ""),
        "publication_date": project.get("publicationDate", ""),
        "organisms": organisms,
        "organism_parts": organism_parts,
        "diseases": diseases,
        "instruments": instruments,
        "experiment_types": experiment_types,
        "quantification_methods": quant_methods,
        "identified_ptms": ptms,
    }


def build_pride_metadata(accessions, session):
    rows = []
    for i, accession in enumerate(accessions, start=1):
        print(f"[metadata {i}/{len(accessions)}] {accession}", end=" ")
        project = fetch_project(accession, session)
        if project is None:
            print("-> not found")
            rows.append({"accession": accession, "title": "", "pubmed_id": "", "doi": "",
                         "pride_license": "", "submission_date": "", "publication_date": "",
                         "organisms": "", "organism_parts": "", "diseases": "", "instruments": "",
                         "experiment_types": "", "quantification_methods": "", "identified_ptms": ""})
            time.sleep(SLEEP)
            continue
        rows.append(row_for_project(accession, project))
        print("-> ok")
        time.sleep(SLEEP)

    with open(PRIDE_METADATA_OUT, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {PRIDE_METADATA_OUT}")


# --------------------------------------------------------------------------
# 2. Manuscript full text, for open-access PXDs only
# --------------------------------------------------------------------------
def get_text(elem):
    if elem is None:
        return ""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def fetch_manuscript_text(pmcid, session, timeout=30):
    resp = session.get(EUROPEPMC_FULLTEXT.format(pmcid=pmcid), timeout=timeout)
    if resp.status_code != 200 or not resp.content:
        return None
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return None

    title = get_text(root.find(".//article-title"))
    abstract = get_text(root.find(".//abstract"))
    body = get_text(root.find(".//body"))
    if not (abstract or body):
        return None

    parts = []
    if title:
        parts.append(f"TITLE\n{title}\n")
    if abstract:
        parts.append(f"ABSTRACT\n{abstract}\n")
    if body:
        parts.append(f"BODY\n{body}\n")
    return "\n".join(parts)


def build_manuscripts(oa_rows, session):
    os.makedirs(MANUSCRIPTS_DIR, exist_ok=True)
    ok, failed = 0, []
    for i, row in enumerate(oa_rows, start=1):
        accession, pmcid = row["accession"], row["pmcid"]
        print(f"[manuscript {i}/{len(oa_rows)}] {accession} ({pmcid})", end=" ")
        if not pmcid:
            print("-> no PMCID recorded")
            failed.append(accession)
            continue
        text = fetch_manuscript_text(pmcid, session)
        if text is None:
            print("-> full text unavailable")
            failed.append(accession)
            time.sleep(SLEEP)
            continue
        outpath = os.path.join(MANUSCRIPTS_DIR, f"{accession}_manuscript.txt")
        with open(outpath, "w") as handle:
            handle.write(text)
        ok += 1
        print(f"-> saved ({len(text)} chars)")
        time.sleep(SLEEP)

    print(f"\nSaved {ok}/{len(oa_rows)} manuscripts to {MANUSCRIPTS_DIR}")
    if failed:
        print(f"{len(failed)} failed: {failed}")


# --------------------------------------------------------------------------
# 3. Abstract only, for closed-access PXDs (PubMed abstracts are public even
#    when the full text isn't)
# --------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]+>")


def fetch_abstract(pubmed_id, session, timeout=20):
    params = {"query": f"EXT_ID:{pubmed_id} AND SRC:MED", "format": "json", "resultType": "core"}
    resp = session.get(EUROPEPMC_SEARCH, params=params, timeout=timeout)
    if resp.status_code != 200:
        return None
    results = resp.json().get("resultList", {}).get("result", [])
    if not results:
        return None
    result = results[0]
    abstract = result.get("abstractText", "")
    if not abstract:
        return None
    abstract = html.unescape(TAG_RE.sub(" ", abstract))
    abstract = re.sub(r"\s+", " ", abstract).strip()
    title = result.get("title", "")
    return f"TITLE\n{title}\n\nABSTRACT\n{abstract}\n"


def build_abstracts(closed_rows, session):
    os.makedirs(ABSTRACTS_DIR, exist_ok=True)
    ok, failed = 0, []
    for i, row in enumerate(closed_rows, start=1):
        accession, pubmed_id = row["accession"], row["pubmed_id"]
        print(f"[abstract {i}/{len(closed_rows)}] {accession}", end=" ")
        if not pubmed_id:
            print("-> no PubMed ID recorded")
            failed.append(accession)
            continue
        text = fetch_abstract(pubmed_id, session)
        if text is None:
            print("-> abstract unavailable")
            failed.append(accession)
            time.sleep(SLEEP)
            continue
        outpath = os.path.join(ABSTRACTS_DIR, f"{accession}_Abstractonly.txt")
        with open(outpath, "w") as handle:
            handle.write(text)
        ok += 1
        print(f"-> saved ({len(text)} chars)")
        time.sleep(SLEEP)

    print(f"\nSaved {ok}/{len(closed_rows)} abstracts to {ABSTRACTS_DIR}")
    if failed:
        print(f"{len(failed)} failed: {failed}")


def main():
    rows = read_license_report()
    accessions = [r["accession"] for r in rows]
    # "Open access" = has a recorded CC license (any flavor), or Europe PMC's
    # own is_open_access flag is Y (covers the few OA-but-unlicensed cases).
    # Not gated on is_open_access alone: it tracks EuropePMC's own hosting,
    # not the paper's actual license, and undercounts CC-BY papers.
    oa_rows = [r for r in rows if r.get("license") or r.get("is_open_access") == "Y"]
    closed_rows = [r for r in rows if r not in oa_rows]

    print(f"{len(accessions)} total PXDs, {len(oa_rows)} open access, {len(closed_rows)} closed access\n")

    session = requests.Session()
    build_pride_metadata(accessions, session)
    print()
    build_manuscripts(oa_rows, session)
    print()
    build_abstracts(closed_rows, session)


if __name__ == "__main__":
    main()
