"""Regex patterns for extracting metadata from proteomics run filenames.

Each field has a list of (compiled_regex, canonical_term) tuples.
Patterns are tried in order; first match wins (except MULTI_FIELDS).
"""

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Boundary helpers for run name patterns
# ---------------------------------------------------------------------------

B = r"(?:^|(?<=[\s_\-\.]))"           # look-behind boundary
B_FLEX = r"(?:^|(?<=[\s_\-\.\d]))"    # also allows digit before (e.g. 10HeLa)
A = r"(?=$|[\s_\-\.\d])"              # look-ahead boundary
A_CAMEL = r"(?=$|[\s_\-\.\d]|(?=[A-Z]))"  # also allows camelCase transition


def _b(pattern: str, canon: str, *, case=False, after=None, before=None):
    """Compile a bounded pattern -> canonical term."""
    flags = re.IGNORECASE if not case else 0
    head = before if before is not None else B
    tail = after if after is not None else A
    return (re.compile(head + pattern + tail, flags), canon)


def load_ontology(ontology_dir: Path, name: str) -> list[str]:
    """Load ontology terms from a text file, skipping comments and blanks."""
    terms = []
    for line in (ontology_dir / f"{name}.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


# ---------------------------------------------------------------------------
# Pattern definitions per field
# ---------------------------------------------------------------------------

# ---------- INSTRUMENT ----------
instrument_patterns = [
    _b(r"(?:Orbitrap[\s_\-]?)?Exploris[\s_\-]?480", "Orbitrap Exploris 480"),
    _b(r"(?:Orbitrap[\s_\-]?)?Exploris[\s_\-]?240", "Orbitrap Exploris 240"),
    _b(r"(?:Orbitrap[\s_\-]?)?Exploris[\s_\-]?120", "Orbitrap Exploris 120"),
    _b(r"(?:Orbitrap[\s_\-]?)?Exploris", "Orbitrap Exploris 480"),
    _b(r"EXPL\d*", "Orbitrap Exploris 480"),
    _b(r"Q[\s_\-]?Exactive[\s_\-]?HF[\s_\-]?X", "Q Exactive HF-X"),
    _b(r"QEHFX\d*", "Q Exactive HF-X"),
    _b(r"HFX\d*", "Q Exactive HF-X"),
    _b(r"qExHF\d*", "Q Exactive HF"),
    _b(r"Q[\s_\-]?Exactive[\s_\-]?HF", "Q Exactive HF"),
    _b(r"QEHF\d*", "Q Exactive HF"),
    _b(r"qExPlus\d*", "Q Exactive Plus"),
    _b(r"Q[\s_\-]?Exactive[\s_\-]?Plus", "Q Exactive Plus"),
    _b(r"QEplus\d*", "Q Exactive Plus"),
    _b(r"QEp\d*", "Q Exactive Plus"),
    _b(r"qExac\d*", "Q Exactive"),
    _b(r"Q[\s_\-]?Exactive", "Q Exactive"),
    _b(r"QE\d+", "Q Exactive"),
    _b(r"Qe\d+", "Q Exactive", case=True),
    _b(r"(?:Orbitrap[\s_\-]?)?Eclipse", "Orbitrap Eclipse"),
    _b(r"(?:Orbitrap[\s_\-]?)?Fusion[\s_\-]?Lumos", "Orbitrap Fusion Lumos"),
    _b(r"LUMOS\d*", "Orbitrap Fusion Lumos"),
    _b(r"(?:Orbitrap[\s_\-]?)?Fusion", "Orbitrap Fusion"),
    _b(r"(?:LTQ[\s_\-]?)?Orbitrap[\s_\-]?Elite", "Orbitrap Elite"),
    _b(r"(?:LTQ[\s_\-]?)?Orbitrap[\s_\-]?Velos", "LTQ Orbitrap Velos"),
    _b(r"(?:LTQ[\s_\-]?)?Orbitrap[\s_\-]?XL", "LTQ Orbitrap XL"),
    _b(r"Orbitrap[\s_\-]?Astral", "Orbitrap Astral"),
    _b(r"Orbi\d+", "Orbitrap"),
    _b(r"LTQ[\s_\-]?Velos", "LTQ Velos"),
    _b(r"Velos\d*", "LTQ Velos"),
    _b(r"timsTOF[\s_\-]?Ultra", "timsTOF Ultra"),
    _b(r"timsTOF[\s_\-]?SCP", "timsTOF SCP"),
    _b(r"timsTOF[\s_\-]?HT", "timsTOF HT"),
    _b(r"timsTOF[\s_\-]?Pro", "timsTOF Pro"),
    _b(r"timsTOF", "timsTOF Pro"),
    _b(r"TripleTOF[\s_\-]?6600", "TripleTOF 6600"),
    _b(r"TripleTOF[\s_\-]?5600", "TripleTOF 5600"),
    _b(r"ZenoTOF[\s_\-]?7600", "ZenoTOF 7600"),
    _b(r"maXis", "maXis", case=True),
    _b(r"impact[\s_\-]?II", "impact II"),
    _b(r"Synapt", "Synapt G2-Si"),
]

# ---------- ACQUISITION ----------
_ACQ_AFTER = r"(?=$|[^A-Z])"
acquisition_patterns = [
    (re.compile(r"DDA" + _ACQ_AFTER), "DDA"),
    (re.compile(r"DIA" + _ACQ_AFTER), "DIA"),
]

# ---------- LABELING ----------
labeling_patterns = [
    (re.compile(r"TMT"), "TMT"),
    _b(r"iTRAQ\d*(?:plex)?", "iTRAQ"),
    _b(r"SILAC", "SILAC", case=True),
]

# ---------- FRAGMENTATION ----------
fragmentation_patterns = [
    _b(r"EThcD", "EThcD"),
    _b(r"ETD", "ETD", case=True),
    _b(r"HCD", "HCD", case=True),
    _b(r"CID", "CID", case=True),
    _b(r"UVPD", "UVPD", case=True),
]

# ---------- ORGANISM ----------
organism_patterns = [
    (re.compile(r"(?:Homo[\s_\-]?sapiens|[Hh]uman)", re.IGNORECASE), "Homo sapiens"),
    _b(r"(?:Mus[\s_\-]?musculus|[Mm]ouse)", "Mus musculus", after=A_CAMEL),
    _b(r"(?:Rattus[\s_\-]?norvegicus|[Rr]at)", "Rattus norvegicus", after=A_CAMEL),
    _b(r"(?:Danio[\s_\-]?rerio|[Zz]ebrafish)", "Danio rerio", after=A_CAMEL),
    _b(r"(?:Drosophila|[Ff]ly)", "Drosophila melanogaster", after=A_CAMEL),
    _b(r"(?:C[\.\s_]?elegans|[Ww]orm)", "Caenorhabditis elegans", after=A_CAMEL),
    _b(r"(?:S[\.\s_]?cerevisiae|[Yy]east)", "Saccharomyces cerevisiae", after=A_CAMEL),
    _b(r"(?:E[\.\s_]?coli)", "Escherichia coli", after=A_CAMEL),
    _b(r"(?:Arabidopsis)", "Arabidopsis thaliana", after=A_CAMEL),
]

# ---------- TISSUE ----------
_tissue_aliases = {
    # organs
    "brain": "brain", "heart": "heart", "liver": "liver", "lung": "lung",
    "kidney": "kidney", "pancreas": "pancreas", "spleen": "spleen",
    "stomach": "stomach", "colon": "colon", "rectum": "rectum",
    "esophagus": "esophagus", "duodenum": "duodenum",
    r"small[\s_\-]?intestine": "small intestine",
    "cecum": "cecum", "aorta": "aorta",
    "prostate": "prostate", "ovary": "ovary",
    r"testicul(?:ar)?": "testis", "testis": "testis",
    "uterus": "uterus", "cervix": "cervix",
    "retina": "retina", "skin": "skin", "breast": "breast",
    "gut": "gut", "tonsil": "tonsil", "placenta": "placenta",
    # body fluids
    "plasma": "blood plasma", "serum": "blood serum",
    "blood": "blood", "saliva": "saliva", "urine": "urine",
    "ascites": "ascites",
    "tear": "tear fluid", "milk": "milk", "sputum": "sputum",
    "feces": "feces", "stool": "feces", "sweat": "sweat",
    # tissue types
    "adipose": "adipose tissue", r"bone[\s_\-]?marrow": "bone marrow",
    r"lymph[\s_\-]?node": "lymph node", r"spinal[\s_\-]?cord": "spinal cord",
    r"frontal[\s_\-]?cortex": "frontal cortex", "cortex": "frontal cortex",
    # cell types
    "platelet": "platelets",
    "monocyte": "monocytes", "macrophage": "macrophage",
    "fibroblast": "fibroblast", "epithelial": "epithelial cell",
    "erythrocyte": "erythrocyte", "neutrophil": "neutrophil",
    "oocyte": "oocyte", "sperm": "sperm",
    # compound words in run names
    "MouseLiver": "liver", "MouseBrain": "brain", "MouseKidney": "kidney",
    "MouseHeart": "heart", "MouseLung": "lung",
    "SmallIntestine": "small intestine",
    "livernuc": "liver", "lowserum": "blood serum",
    "iPSCs?": "stem cells", "hiPSCs?": "stem cells", "hESCs?": "stem cells",
    r"Dentate[\s_\-]?Gyrus": "brain", r"Pyramidal[\s_\-]?Layer": "brain",
    r"Stratum[\s_\-]?Moleculare": "brain",
    "hippocampus": "brain", "cerebellum": "brain", "striatum": "brain",
    "hypothalamus": "brain", "amygdala": "brain", "thalamus": "brain",
    "midbrain": "brain",
}

tissue_patterns = []
for alias, canon in _tissue_aliases.items():
    if alias.isupper() and len(alias) <= 5:
        tissue_patterns.append(_b(alias, canon, case=True))
    else:
        tissue_patterns.append(_b(alias, canon))

# Additional tissue patterns needing special boundaries
tissue_patterns.append(_b(r"cardiac", "heart", after=A_CAMEL))
tissue_patterns.append((re.compile(r"PBMCs?"), "PBMCs"))
tissue_patterns.append(_b(r"[Ss]erum", "blood serum", after=A_CAMEL))
tissue_patterns.append(_b(r"[Mm]yofibre", "skeletal muscle"))
tissue_patterns.append(_b(r"[Mm]yofiber", "skeletal muscle"))
tissue_patterns.append((re.compile(r"[Mm]uscle", re.IGNORECASE), "skeletal muscle"))
tissue_patterns.append((re.compile(r"[Ss]perm", re.IGNORECASE), "sperm"))
tissue_patterns.append((re.compile(r"[Hh]epatocyte", re.IGNORECASE), "liver"))

# ---------- CELL PART ----------
cell_part_patterns = [
    _b(r"mitochondria(?:l)?", "mitochondria"),
    _b(r"[Mm]ito", "mitochondria"),
    _b(r"nucle(?:us|ar|i)", "nucleus"),
    _b(r"Nuc", "nucleus", case=True),
    _b(r"cytoplasm(?:ic)?", "cytoplasm"),
    _b(r"Cyt", "cytoplasm", case=True),
    _b(r"membrane", "membrane"),
    _b(r"MEMB", "membrane", case=True),
    _b(r"ribosom(?:e|al|es)", "ribosome"),
    _b(r"lysosom(?:e|al|es)", "lysosome"),
    _b(r"peroxisom(?:e|al|es)", "peroxisome"),
]

# ---------- DISEASE ----------
disease_patterns = [
    _b(r"[Hh]ealthy", "healthy"),
    _b(r"[Nn]on[\s_\-]?[Cc]ancerous", "healthy"),
    _b(r"[Bb]enign", "healthy"),
    _b(r"WT", "healthy", case=True),
    _b(r"BrCa", "breast cancer", case=True),
    (re.compile(r"[Cc]ancer"), "cancer"),
    _b(r"[Tt]umor", "cancer"),
    _b(r"[Tt]umour", "cancer"),
    _b(r"[Cc]arcinoma", "cancer"),
    _b(r"[Mm]elanoma", "melanoma"),
    _b(r"[Gg]lioblastoma", "glioblastoma"),
    _b(r"[Ll]eukemia", "leukemia"),
    _b(r"[Ll]ymphoma", "lymphoma"),
    _b(r"[Ii]nfluenza", "influenza"),
    _b(r"COVID", "COVID-19", case=True),
    _b(r"SARS", "COVID-19", case=True),
]

# ---------- ENRICHMENT ----------
enrichment_patterns = [
    (re.compile(r"[Pp]hosph", re.IGNORECASE), "phospho-enrichment"),
    _b(r"TiO\d", "phospho-enrichment", case=True),
    (re.compile(r"[Pp]hos(?=[\s_\-\.])", re.IGNORECASE), "phospho-enrichment"),
    _b(r"ubiquit(?:in|yl)", "ubiquitin-enrichment"),
    _b(r"acetyl", "acetyl-enrichment"),
    _b(r"glyco(?:syl)?", "glyco-enrichment"),
    _b(r"SUMO", "SUMO-enrichment", case=True),
]

# ---------- FRACTIONATION ----------
fractionation_patterns = [
    (re.compile(r"[Ff]racc?(?:tion(?:at(?:ed|ion))?)?"), "True"),
    (re.compile(r"[Ff]r(?=[\dA-Z_])"), "True"),
    (re.compile(r"[Ff]xn(?=\d)"), "True"),
    (re.compile(r"(?<=[\s_\-\.])[fdFD]\d+(?=$|[\s_\.\-])"), "True"),
]


# ---------------------------------------------------------------------------
# Build FIELD_PATTERNS (requires ontology_dir for cell lines)
# ---------------------------------------------------------------------------

def build_field_patterns(ontology_dir: Path = None) -> dict:
    """Build the master field -> patterns dict.

    Args:
        ontology_dir: Path to ontology directory (needed for cell line terms).
                      If None, cell_line patterns are empty.
    """
    # Cell line patterns (need ontology file)
    cell_line_pats = []
    if ontology_dir and (ontology_dir / "cell_lines.txt").exists():
        cell_line_terms = load_ontology(ontology_dir, "cell_lines")
        for term in cell_line_terms:
            escaped = re.escape(term)
            flexible = re.sub(r"\\[ \-]", r"[\\s_\\-]?", escaped)
            cell_line_pats.append((re.compile(B_FLEX + flexible + A, re.IGNORECASE), term))
        # Common short aliases
        cell_line_pats.append(_b(r"HEK", "HEK293", before=B_FLEX))
        cell_line_pats.append(_b(r"Jur", "Jurkat", case=True, before=B_FLEX))

    return {
        "instrument": instrument_patterns,
        "acquisition": acquisition_patterns,
        "labeling": labeling_patterns,
        "fragmentation": fragmentation_patterns,
        "organism": organism_patterns,
        "tissue": tissue_patterns,
        "cell_line": cell_line_pats,
        "cell_part": cell_part_patterns,
        "disease": disease_patterns,
        "enrichment": enrichment_patterns,
        "fractionation": fractionation_patterns,
    }


# Body fluids: when both a fluid and a solid tissue match, the fluid wins
BODY_FLUIDS = {
    "blood plasma", "blood serum", "blood", "saliva", "urine",
    "ascites", "tear fluid", "milk", "sputum", "feces", "sweat",
    "cerebrospinal fluid", "seminal plasma",
}

# Fields where all matches are collected (not just first)
MULTI_FIELDS = {"fragmentation"}
