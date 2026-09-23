#!/usr/bin/env python3
"""Build a standalone test set from pre-2026 bigbio SDRF-Proteomics annotations.

Source: bigbio/proteomics-sample-metadata, annotated-projects/, at the last
commit before 2026-01-01 (c1873bd, 2025-10-16). This predates the migration to
bigbio/sdrf-annotated-datasets and the LLM-assisted annotations of 2026.

Selection: PXDs in annotated-projects/ at that commit that are not in the
107-PXD training set and whose originating publication is open access (see
SELECTION below for the evidence per PXD).

For each PXD this script writes to <PXD>/:
  sources/*.sdrf.tsv       the original bigbio SDRF file(s), unmodified
  <PXD>.sdrf.tsv           all SDRF files of the PXD merged into one table
  manuscript.txt           abstract + methods, no headings
  manuscript_fulltext.txt  title, abstract and all body sections with headings
and to gold/ the three <PXD>_<Agent>_golden.json files, produced with the
unmodified convert_sdrf() from agentic-benchmark/benchmark_data/sdrf_to_golden.py.

Nothing outside this directory is written.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "framework"))
sys.path.insert(0, str(REPO / "agentic-benchmark" / "benchmark_data"))
from sdrf_to_golden import convert_sdrf, normalize_header, parse_nt_value, unique_values  # noqa: E402

COMMIT = "c1873bd63af9087c9045161dec24398026527c9c"
RAW = "https://raw.githubusercontent.com/bigbio/proteomics-sample-metadata/{commit}/{path}"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&query=PMCID:{pmcid}"
EPMC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
NCBI_XML = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={num}&rettype=xml"
AGENTS = ["BiologicalAgent", "TechnicalAgent", "ExperimentalDesignAgent"]
MISSING = "not available"

# pxd: (sdrf paths at COMMIT, pmid, pmcid, text source, licence, how the paper was matched)
# text source: epmc = Europe PMC OA full text; ncbi = NCBI PMC (author manuscript,
# text-mining permitted); manual = open access (CC BY) but no machine-readable
# full text could be downloaded, text has to be added by hand.
SELECTION = {
    "PXD000288": (["annotated-projects/PXD000288/PXD000288.sdrf.tsv"], "", "PMC4390264", "epmc", "cc by", "EPMC full-text hit, same study as PRIDE title"),
    "PXD001224": (["annotated-projects/PXD001224/PXD001224.sdrf.tsv"], "", "PMC4510444", "epmc", "", "EPMC full-text hit, title match 0.71 (data article)"),
    "PXD004452": (["annotated-projects/PXD004452/PXD004452-celllines.sdrf.tsv", "annotated-projects/PXD004452/PXD004452-tissues.sdrf.tsv"], "28601559", "PMC5493283", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD006430": (["annotated-projects/PXD006430/PXD006430-silac.sdrf.tsv", "annotated-projects/PXD006430/PXD006430-tmt.sdrf.tsv"], "28915803", "PMC5602878", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD008841": (["annotated-projects/PXD008841/PXD008841.sdrf.tsv"], "", "PMC6453966", "epmc", "cc by", "EPMC full-text hit; jPOST JPST000265 (Johansson et al.), not in PRIDE"),
    "PXD009909": (["annotated-projects/PXD009909/PXD009909.sdrf.tsv"], "", "PMC6334985", "epmc", "", "EPMC full-text hit, same study as PRIDE title"),
    "PXD014458": (["annotated-projects/PXD014458/PXD14458.sdrf.tsv"], "31316139", "PMC6637242", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD014502": (["annotated-projects/PXD014502/PXD014502.sdrf.tsv"], "", "PMC11563575", "epmc", "", "EPMC full-text hit, same study as PRIDE title"),
    "PXD014525": (["annotated-projects/PXD014525/PXD014525-dda.sdrf.tsv", "annotated-projects/PXD014525/PXD014525-dia.sdrf.tsv"], "32034161", "PMC7005859", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD015093": (["annotated-projects/PXD015093/PXD015093-LFQ.sdrf.tsv", "annotated-projects/PXD015093/PXD015093-TMT.sdrf.tsv"], "32817103", "PMC7439480", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD015270": (["annotated-projects/PXD015270/PXD015270.sdrf.tsv"], "", "PMC7183755", "epmc", "", "EPMC full-text hit, title match 0.60"),
    "PXD015578": (["annotated-projects/PXD015578/PXD015578.sdrf.tsv"], "", "PMC7210691", "epmc", "", "EPMC full-text hit, same study as PRIDE title"),
    "PXD015833": ([f"annotated-projects/PXD015833/PXD015833-Exp{i}.sdrf.tsv" for i in range(1, 7)], "32284562", "PMC7116245", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD016772": (["annotated-projects/PXD016772/PXD016772.sdrf.tsv"], "", "PMC7054685", "epmc", "", "EPMC full-text hit, title match 1.00"),
    "PXD017291": (["annotated-projects/PXD017291/PXD017291-mixed-label.sdrf.tsv", "annotated-projects/PXD017291/PXD017291-tmt.sdrf.tsv"], "33022891", "PMC7538195", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD018241": (["annotated-projects/PXD018241/PXD018241.sdrf.tsv", "annotated-projects/PXD018241/PXD018241-phosphoproteomics.sdrf.tsv"], "", "PMC7386171", "epmc", "", "EPMC full-text hit, same study as PRIDE title"),
    "PXD018678": (["annotated-projects/PXD018678/PXD018678-dda.sdrf.tsv", "annotated-projects/PXD018678/PXD018678-dia.sdrf.tsv"], "32487995", "PMC7266817", "epmc", "cc by", "PRIDE pubmedID"),
    "PXD018830": (["annotated-projects/PXD018830/PXD018830-DIA.sdrf.tsv"], "33855848", "PMC8155562", "epmc", "cc by", "PRIDE pubmedID"),
    # one SDRF and one paper cover PXD019185 and PXD018883; kept as one entry
    "PXD018883": (["annotated-projects/PXD019185_PXD018883/PXD019185_PXD018883.sdrf.tsv"], "35121989", "PMC8818089", "epmc", "", "PRIDE pubmedID (also covers PXD019185)"),
    "PXD020381": (["annotated-projects/PXD020381/PXD020381.sdrf.tsv"], "", "PMC7990342", "epmc", "", "EPMC full-text hit, same study as PRIDE title"),
    "PXD020394": (["annotated-projects/PXD020394/PXD020394.sdrf.tsv"], "", "PMC7405904", "epmc", "", "EPMC full-text hit, title match 1.00 (data article)"),
    "PXD051889": (["annotated-projects/PXD051889/PXD051889.sdrf.tsv"], "", "PMC11161513", "epmc", "", "EPMC full-text hit, title match 0.65"),
    "PXD017710": (["annotated-projects/PXD017710/PXD017710-silac.sdrf.tsv", "annotated-projects/PXD017710/PXD017710-tmt.sdrf.tsv"], "32408336", "PMC7616921", "ncbi", "author manuscript (text mining permitted)", "PRIDE pubmedID"),
    "PXD001487": (["annotated-projects/PXD001487/PXD001487.sdrf.tsv"], "26801919", "PMC4824855", "manual", "cc by", "PRIDE pubmedID"),
    "PXD001736": (["annotated-projects/PXD001736/PXD001736.sdrf.tsv"], "25755297", "PMC4424410", "manual", "cc by", "PRIDE pubmedID"),
    "PXD002192": (["annotated-projects/PXD002192/PXD002192.sdrf.tsv"], "26081835", "PMC4528248", "manual", "cc by", "PRIDE pubmedID"),
    "PXD004436": (["annotated-projects/PXD004436/PXD004436.sdrf.tsv"], "28373295", "PMC5546195", "manual", "cc by", "PRIDE pubmedID"),
    "PXD004624": (["annotated-projects/PXD004624/PXD004624.sdrf.tsv"], "27879288", "PMC5217785", "manual", "cc by", "PRIDE pubmedID"),
    "PXD006542": (["annotated-projects/PXD006542/PXD006542.sdrf.tsv"], "29358339", "PMC5880108", "manual", "cc by", "PRIDE pubmedID"),
    "PXD012243": (["annotated-projects/PXD012243/PXD012243.sdrf.tsv"], "30798302", "PMC6495254", "manual", "cc by", "PRIDE pubmedID"),
}

METHODS_RE = re.compile(r"method|material|experimental (procedure|section|design)|procedures|star\W*methods", re.I)


def fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


# ── SDRF merging ──

def read_sdrf(text: str) -> tuple[list[str], list[list[str]]]:
    rows = [r for r in csv.reader(io.StringIO(text), delimiter="\t") if any(c.strip() for c in r)]
    header = [h.strip().lower() or "unnamed" for h in rows[0]]
    return header, rows[1:]


def merge_sdrfs(tables: list[tuple[list[str], list[list[str]]]]) -> tuple[list[str], list[list[str]]]:
    """Union of columns over all files. Repeated headers (e.g. several
    comment[modification parameters]) are aligned by occurrence number.
    Empty cells become 'not available' so no row starts with an empty cell."""
    keys: list[tuple[str, int]] = []
    per_file = []
    for header, rows in tables:
        seen: dict[str, int] = {}
        fkeys = []
        for h in header:
            k = (h, seen.get(h, 0))
            seen[h] = k[1] + 1
            fkeys.append(k)
            if k not in keys:
                keys.append(k)
        per_file.append((fkeys, rows))
    merged = []
    for fkeys, rows in per_file:
        for row in rows:
            cells = dict(zip(fkeys, row))
            merged.append([(cells.get(k) or "").strip() or MISSING for k in keys])
    return [k[0] for k in keys], merged


# ── gold ──

# The shared SDRF_TECHNICAL / SDRF_BIOLOGICAL mappings use the Kaggle-style
# names of the training SDRFs (CleavageAgent, ...). These standard
# SDRF-Proteomics headers are not mapped there, so convert_sdrf() returns null
# for them. Fill them here only when convert_sdrf() left the field null.
STANDARD_ALIASES = {
    "cleavageagentdetails": ("TechnicalAgent", "cleavage_agent"),
    "dissociationmethod": ("TechnicalAgent", "fragmentation"),
    "reductionreagent": ("TechnicalAgent", "reduction_reagent"),
    "strain": ("BiologicalAgent", "strain"),
}


def add_standard_fields(golden: dict, header: list[str], rows: list[list[str]]) -> list[str]:
    filled = []
    norm = [normalize_header(h) for h in header]
    for col, (agent, field) in STANDARD_ALIASES.items():
        if golden[agent]["fields"].get(field) is not None:
            continue
        values = [row[i] for i, h in enumerate(norm) if h == col for row in rows if i < len(row)]
        value = unique_values(values, parse_nt_value)
        if value is not None:
            golden[agent]["fields"][field] = value
            filled.append(field)
    # ontology-annotated biological values ("NT=head and neck;AC=MA:0000006") -> name
    for field, value in golden["BiologicalAgent"]["fields"].items():
        if isinstance(value, str) and "NT=" in value:
            golden["BiologicalAgent"]["fields"][field] = unique_values(value.split("; "), parse_nt_value)
    return filled


# ── JATS full text ──

def text_of(el) -> str:
    el = etree.fromstring(etree.tostring(el))
    # drop citation markers, tables, figures and formulas but keep the text after them
    etree.strip_elements(el, "xref", "table-wrap", "fig", "table", "disp-formula",
                         "supplementary-material", with_tail=False)
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def paragraphs(sec) -> list[str]:
    return [t for t in (text_of(p) for p in sec.xpath(".//p[not(ancestor::table-wrap) and not(ancestor::fig)]")) if t]


def extract(xml: bytes) -> tuple[str, str]:
    root = etree.fromstring(xml)
    art = root.xpath("//article")[0] if root.tag != "article" else root
    title = text_of(art.xpath(".//front//article-title")[0]) if art.xpath(".//front//article-title") else ""
    abstracts = [a for a in art.xpath(".//front//abstract") if a.get("abstract-type") not in ("graphical", "teaser", "toc")]
    abstract = [p for a in abstracts for p in paragraphs(a)]
    sections = []
    for sec in art.xpath("./body/sec|./back/sec[not(ancestor::ref-list)]"):
        head = text_of(sec.xpath("./title")[0]) if sec.xpath("./title") else ""
        kind = (sec.get("sec-type") or "") + " " + head
        sections.append((head, kind, paragraphs(sec)))
    methods = [p for head, kind, ps in sections if METHODS_RE.search(kind) for p in ps]
    short = "\n\n".join(abstract + methods)
    parts = [f"=== TITLE ===\n{title}", "=== ABSTRACT ===\n" + "\n\n".join(abstract)]
    parts += [f"=== {head.upper() or 'SECTION'} ===\n" + "\n\n".join(ps) for head, _, ps in sections if ps]
    return short, "\n\n".join(parts)


def main() -> None:
    gold_dir = HERE / "gold"
    gold_dir.mkdir(exist_ok=True)
    manifest = []
    for pxd, (paths, pmid, pmcid, source, licence, evidence) in SELECTION.items():
        hit = json.loads(fetch(EPMC_SEARCH.format(pmcid=pmcid)))["resultList"]["result"][0]
        if pmid and hit.get("pmid") != pmid:
            raise ValueError(f"{pxd}: PRIDE pmid {pmid} != Europe PMC pmid {hit.get('pmid')} for {pmcid}")
        pmid, title, year = hit.get("pmid", ""), hit.get("title", ""), hit.get("pubYear", "")
        out = HERE / pxd
        (out / "sources").mkdir(parents=True, exist_ok=True)
        tables = []
        for path in paths:
            text = fetch(RAW.format(commit=COMMIT, path=path)).decode("utf-8", errors="replace")
            (out / "sources" / Path(path).name).write_text(text)
            tables.append(read_sdrf(text))
        header, rows = merge_sdrfs(tables)
        with open(out / f"{pxd}.sdrf.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

        golden = convert_sdrf(out / f"{pxd}.sdrf.tsv", pxd)
        filled = add_standard_fields(golden, header, rows)
        for agent in AGENTS:
            golden[agent]["_source"] = f"sdrf:bigbio/proteomics-sample-metadata@{COMMIT[:7]}"
            (gold_dir / f"{pxd}_{agent}_golden.json").write_text(json.dumps(golden[agent], indent=2, ensure_ascii=False))
        n_gold = sum(v is not None for a in AGENTS for v in golden[a]["fields"].values())

        text_status = "pending: add manuscript text by hand"
        if source in ("epmc", "ncbi"):
            url = EPMC_XML.format(pmcid=pmcid) if source == "epmc" else NCBI_XML.format(num=pmcid[3:])
            short, full = extract(fetch(url))
            (out / "manuscript.txt").write_text(short + "\n")
            (out / "manuscript_fulltext.txt").write_text(full + "\n")
            text_status = f"ok ({len(short)} chars abstract+methods, {len(full)} chars full text)"
            if len(short) < 2000:
                text_status = "check: " + text_status
            time.sleep(0.4)

        manifest.append({
            "pxd": pxd, "pmid": pmid, "pmcid": pmcid, "year": year, "title": title, "licence": licence, "text_source": source,
            "paper_match": evidence, "n_sdrf_files": len(paths), "n_rows": len(rows),
            "n_nonnull_gold_fields": n_gold, "filled_from_standard_headers": ";".join(filled), "text_status": text_status,
            "sdrf_paths": ";".join(paths),
        })
        print(pxd, len(paths), "sdrf,", len(rows), "rows,", n_gold, "gold fields,", text_status)

    with open(HERE / "manifest.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)


if __name__ == "__main__":
    main()
