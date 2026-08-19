"""
Fetch manuscript text and metadata for a list of PXD accessions.

For each PXD:
  1. Query PRIDE API v3 for project metadata
  2. Get PMCID from references via NCBI ID converter
  3. Fetch open access full text via NCBI BioC API
  4. Fetch SDRF file if available from PRIDE
  5. Combine all information into one .txt file

Usage:
    python manuscripts/fetch_manuscripts.py
    python manuscripts/fetch_manuscripts.py --input manuscripts/pxd.txt --output manuscripts/output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# Rate limiting
PRIDE_DELAY = 0.5  # seconds between PRIDE API calls
PMC_DELAY = 0.5    # seconds between PMC API calls


# 1. PRIDE API
def fetch_pride_project(pxd_id: str) -> dict:
    """Fetch project metadata from PRIDE API v3."""
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v3/projects/{pxd_id}"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        print(f"  {pxd_id}: not found in PRIDE")
        return {}
    resp.raise_for_status()
    return resp.json()


def fetch_pride_files(pxd_id: str) -> list:
    """Fetch file list from PRIDE, paging through until we find SDRF or exhaust pages."""
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v3/projects/{pxd_id}/files"
    page = 0
    while page < 50:  # safety limit
        params = {"pageSize": 100, "page": page}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            files = resp.json()
            if not files:
                break
            # Check this page for SDRF before fetching more
            for f in files:
                name = f.get("fileName", "")
                if name.lower() == "sdrf.tsv" or name.lower().endswith(".sdrf.tsv"):
                    return [f]  # only need the SDRF file
            if len(files) < 100:
                break
            page += 1
        except Exception:
            break
    return []


def fetch_sdrf(pxd_id: str, files: list) -> str:
    """Download SDRF file content if available."""
    for f in files:
        name = f.get("fileName", "")
        if name.lower() == "sdrf.tsv" or name.lower().endswith(".sdrf.tsv"):
            # PRIDE v3 stores URLs in publicFileLocations array
            download_url = None
            for loc in f.get("publicFileLocations", []):
                url = loc.get("value", "")
                if url.startswith("ftp://"):
                    # Convert FTP to HTTP for easier downloading
                    download_url = url.replace("ftp://ftp.pride.ebi.ac.uk/pride/data/archive/",
                                               "https://ftp.pride.ebi.ac.uk/pride/data/archive/")
                    break
                elif url.startswith("http"):
                    download_url = url
                    break

            if download_url:
                try:
                    resp = requests.get(download_url, timeout=60)
                    resp.raise_for_status()
                    return resp.text
                except Exception as e:
                    print(f"  {pxd_id}: SDRF download failed: {e}")
    return ""



# 2. PMCID Lookup
def get_pmcid_from_pubmed(pubmed_id: str) -> str:
    """Convert PubMed ID to PMCID via NCBI ID converter."""
    if not pubmed_id or pubmed_id == "No pubmedID":
        return ""

    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {
        "tool": "agentic-metadata",
        "email": "agentic-metadata@example.com",
        "ids": str(pubmed_id),
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for record in data.get("records", []):
            pmcid = record.get("pmcid", "")
            if pmcid:
                return pmcid
    except Exception:
        pass
    return ""


# 3. Full Text Fetching (BioC API)
def fetch_bioc_fulltext(pmcid: str) -> dict:
    """Fetch full text as structured JSON from NCBI BioC API."""
    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
    try:
        resp = requests.get(url, timeout=30)
        if resp.ok:
            resp.encoding = "utf-8"
            return resp.json()
    except Exception:
        pass
    return []


def extract_sections(bioc_json: list) -> dict:
    """Extract text sections from BioC JSON response."""
    sections = {}

    if not bioc_json:
        return sections

    try:
        documents = bioc_json[0].get("documents", [])
        if isinstance(documents, list) and documents:
            doc = documents[0]
        else:
            doc = documents

        for passage in doc.get("passages", []):
            text = passage.get("text", "")
            if not text:
                continue

            infons = passage.get("infons", {})
            section_type = infons.get("section_type", "OTHER")

            if section_type in ("ABSTRACT", "METHODS", "RESULTS", "INTRO", "DISCUSS", "CONCL", "SUPPL"):
                if section_type not in sections:
                    sections[section_type] = []
                sections[section_type].append(text)
    except (KeyError, IndexError, TypeError):
        pass

    return sections


# 4. Fetch abstract from EuropePMC (fallback)
def fetch_europepmc_abstract(pubmed_id: str) -> str:
    """Fallback: fetch abstract from EuropePMC if BioC fails."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=ext_id:{pubmed_id}&format=json&resultType=core"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
        if results:
            return results[0].get("abstractText", "")
    except Exception:
        pass
    return ""


# 5. SDRF Summarization
# Columns we care about from SDRF — extract unique values from these
SDRF_COLUMNS = [
    "characteristics[organism]",
    "characteristics[organism part]",
    "characteristics[disease]",
    "characteristics[cell type]",
    "characteristics[cell line]",
    "characteristics[sex]",
    "characteristics[age]",
    "characteristics[developmental stage]",
    "characteristics[ancestry category]",
    "comment[instrument]",
    "comment[cleavage agent details]",
    "comment[modification parameters]",
    "comment[label]",
    "comment[fractionation method]",
    "comment[dissociation method]",
    "comment[collision energy]",
    "comment[precursor mass tolerance]",
    "comment[fragment mass tolerance]",
    "comment[MS2 mass analyzer]",
]


def parse_sdrf_value(raw: str) -> str:
    """Extract human-readable value from SDRF cell like 'NT=Trypsin;AC=MS:1001251'."""
    raw = raw.strip()
    if not raw or raw.lower() in ("not available", "not applicable", "not provided"):
        return ""
    # If it contains NT= (name term), extract that
    if "NT=" in raw:
        for part in raw.split(";"):
            part = part.strip()
            if part.startswith("NT="):
                return part[3:]
    # If it contains AC= only (like AC=MS:1002038), try to get NT from elsewhere
    if "AC=" in raw and "NT=" not in raw:
        return raw  # keep as-is, still informative
    return raw


def summarize_sdrf(sdrf_text: str) -> str:
    """Summarize SDRF by extracting unique values per relevant column."""
    lines = sdrf_text.strip().split("\n")
    if len(lines) < 2:
        return ""

    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]

    summary_parts = []
    summary_parts.append(f"Total runs: {len(rows)}")

    for col_name in SDRF_COLUMNS:
        # Find column index (case-insensitive, may appear multiple times)
        indices = [i for i, h in enumerate(header) if h.strip().lower() == col_name.lower()]
        if not indices:
            continue

        # Collect unique values across all matching columns
        unique_values = set()
        for row in rows:
            for idx in indices:
                if idx < len(row):
                    parsed = parse_sdrf_value(row[idx])
                    if parsed:
                        unique_values.add(parsed)

        if unique_values:
            # Clean column name for display
            display_name = col_name.replace("characteristics[", "").replace("comment[", "").rstrip("]")
            values_str = "; ".join(sorted(unique_values))
            summary_parts.append(f"{display_name}: {values_str}")

    return "\n".join(summary_parts)


def summarize_pride_properties(pride_data: dict) -> str:
    """Extract curated properties from PRIDE project API response."""
    lines = []

    # Organisms
    organisms = [o.get("name", "") for o in pride_data.get("organisms", [])]
    if organisms:
        lines.append(f"organisms: {'; '.join(organisms)}")

    # Organism parts
    org_parts = [o.get("name", "") for o in pride_data.get("organismParts", [])]
    if org_parts:
        lines.append(f"organism parts: {'; '.join(org_parts)}")

    # Diseases
    diseases = [d.get("name", "") for d in pride_data.get("diseases", [])]
    if diseases:
        lines.append(f"diseases: {'; '.join(diseases)}")

    # Instruments
    instruments = [i.get("name", "") for i in pride_data.get("instruments", [])]
    if instruments:
        lines.append(f"instruments: {'; '.join(instruments)}")

    # PTMs
    ptms = [p.get("name", "") for p in pride_data.get("identifiedPTMStrings", [])]
    if ptms:
        lines.append(f"modifications: {'; '.join(ptms)}")

    # Experiment types
    exp_types = [e.get("name", "") for e in pride_data.get("experimentTypes", [])]
    if exp_types:
        lines.append(f"experiment types: {'; '.join(exp_types)}")

    # Quantification methods
    quant = [q.get("name", "") for q in pride_data.get("quantificationMethods", [])]
    if quant:
        lines.append(f"quantification: {'; '.join(quant)}")

    # Keywords
    keywords = pride_data.get("keywords", [])
    if keywords:
        lines.append(f"keywords: {'; '.join(keywords)}")

    return "\n".join(lines)


# 6. Combine into .txt
def build_manuscript_text(
    pxd_id: str,
    pride_data: dict,
    sections: dict,
    sdrf_text: str,
    abstract_fallback: str = "",
) -> str:
    """Combine all fetched data into a single manuscript text."""
    parts = []

    # PRIDE project properties (curated metadata)
    pride_props = summarize_pride_properties(pride_data)
    if pride_props:
        parts.append(f"PRIDE PROJECT PROPERTIES:\n{pride_props}")

    # PRIDE metadata
    abstract = pride_data.get("projectDescription", "")
    sample_proc = pride_data.get("sampleProcessingProtocol", "")
    data_proc = pride_data.get("dataProcessingProtocol", "")

    if abstract:
        parts.append(f"ABSTRACT:\n{abstract}")

    if pride_data.get("title"):
        parts.append(f"PROJECT DESCRIPTION:\n{pride_data['title']}")

    if sample_proc:
        parts.append(f"SAMPLE PROCESSING:\n{sample_proc}")

    if data_proc:
        parts.append(f"DATA PROCESSING:\n{data_proc}")

    # Full text sections from BioC
    if sections.get("METHODS"):
        parts.append(f"METHODS:\n" + "\n".join(sections["METHODS"]))

    if sections.get("RESULTS"):
        parts.append(f"RESULTS:\n" + "\n".join(sections["RESULTS"]))

    if sections.get("INTRO"):
        parts.append(f"INTRODUCTION:\n" + "\n".join(sections["INTRO"]))

    if sections.get("SUPPL"):
        parts.append(f"SUPPLEMENTARY:\n" + "\n".join(sections["SUPPL"]))

    # Fallback abstract if no BioC text
    if not sections and abstract_fallback:
        if abstract_fallback != abstract:
            parts.append(f"ABSTRACT (PubMed):\n{abstract_fallback}")

    # SDRF data — summarize unique values per column
    if sdrf_text:
        sdrf_summary = summarize_sdrf(sdrf_text)
        if sdrf_summary:
            parts.append(f"SDRF SUMMARY:\n{sdrf_summary}")

    return "\n\n".join(parts)


# 6. Main Pipeline
def fetch_single_pxd(pxd_id: str, output_dir: Path, overwrite: bool = False) -> dict:
    """Fetch all data for a single PXD and save as .txt."""
    output_file = output_dir / f"{pxd_id}.txt"
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / f"{pxd_id}.meta.json"

    if output_file.exists() and not overwrite:
        print(f"  {pxd_id}: already exists, skipping")
        return {"status": "skipped"}

    meta = {"pxd_id": pxd_id, "has_fulltext": False, "has_sdrf": False}

    # 1. PRIDE API
    pride_data = fetch_pride_project(pxd_id)
    if not pride_data:
        return {"status": "not_found"}
    time.sleep(PRIDE_DELAY)

    # Extract references for PMCID lookup
    references = pride_data.get("references", [])
    pubmed_id = ""
    pmcid = ""
    for ref in references:
        pid = ref.get("pubmedID", "")
        if pid and str(pid) != "0":
            pubmed_id = str(pid)
            break

    # 2. Get PMCID
    if pubmed_id:
        pmcid = get_pmcid_from_pubmed(pubmed_id)
        meta["pubmed_id"] = pubmed_id
        meta["pmcid"] = pmcid
        time.sleep(PMC_DELAY)

    # 3. Fetch full text
    sections = {}
    abstract_fallback = ""
    if pmcid:
        bioc = fetch_bioc_fulltext(pmcid)
        sections = extract_sections(bioc)
        meta["has_fulltext"] = bool(sections)
        time.sleep(PMC_DELAY)

    # 4. Fallback abstract
    if not sections and pubmed_id:
        abstract_fallback = fetch_europepmc_abstract(pubmed_id)

    # 5. Fetch SDRF
    files = fetch_pride_files(pxd_id)
    sdrf_text = fetch_sdrf(pxd_id, files)
    meta["has_sdrf"] = bool(sdrf_text)
    time.sleep(PRIDE_DELAY)

    # 6. Combine and save
    manuscript = build_manuscript_text(pxd_id, pride_data, sections, sdrf_text, abstract_fallback)

    if manuscript.strip():
        output_file.write_text(manuscript, encoding="utf-8")
        meta["status"] = "success"
        meta["chars"] = len(manuscript)
        meta["sections"] = list(sections.keys())
    else:
        meta["status"] = "empty"

    # Save metadata
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    status = "+" if meta["has_fulltext"] else "~"
    sdrf_flag = " +SDRF" if meta["has_sdrf"] else ""
    print(f"  {pxd_id}: {status} {meta.get('chars', 0):,} chars{sdrf_flag}")

    return meta

def fetch_all_manuscripts(
    input_file: Path = None,
    output_dir: Path = None,
    benchmark_dir: Path = None,
    overwrite: bool = False,
    pxd_ids: list = None,
) -> dict:
    """Fetch manuscript text for all PXD accessions.

    Provide either pxd_ids (list of PXD strings) or input_file (text file, one PXD per line).

    Returns:
        dict with counts: success, skipped, not_found, empty, fulltext, sdrf.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if pxd_ids is None:
        pxd_ids = [line.strip() for line in input_file.read_text().splitlines() if line.strip() and line.strip().startswith("PXD")]
    print(f"Fetching {len(pxd_ids)} PXDs -> {output_dir}/")

    stats = {"success": 0, "skipped": 0, "not_found": 0, "empty": 0, "fulltext": 0, "sdrf": 0}

    for i, pxd_id in enumerate(pxd_ids):
        print(f"[{i+1}/{len(pxd_ids)}]", end="")
        meta = fetch_single_pxd(pxd_id, output_dir, overwrite=overwrite)

        if benchmark_dir and meta.get("status") == "success":
            bench_pxd_dir = benchmark_dir / pxd_id
            if bench_pxd_dir.exists():
                src = output_dir / f"{pxd_id}.txt"
                dst = bench_pxd_dir / f"{pxd_id}.txt"
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

        status = meta.get("status", "error")
        stats[status] = stats.get(status, 0) + 1
        if meta.get("has_fulltext"):
            stats["fulltext"] += 1
        if meta.get("has_sdrf"):
            stats["sdrf"] += 1

    print(f"\nDone: {stats['success']} fetched, {stats['skipped']} skipped, "
          f"{stats['not_found']} not found, {stats['empty']} empty")
    print(f"Full text: {stats['fulltext']}, SDRF: {stats['sdrf']}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Fetch manuscript text for PXD accessions")
    parser.add_argument("--input", default="manuscripts/pxd.txt", help="File with PXD IDs, one per line")
    parser.add_argument("--output", default="manuscripts/output", help="Output directory for .txt files")
    parser.add_argument("--benchmark", default=None, help="Also copy .txt into benchmark test_PXD dirs (e.g. benchmark/test_PXD)")
    parser.add_argument("--overwrite", action="store_true", help="Re-fetch existing files")
    args = parser.parse_args()

    fetch_all_manuscripts(
        input_file=Path(args.input),
        output_dir=Path(args.output),
        benchmark_dir=Path(args.benchmark) if args.benchmark else None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
