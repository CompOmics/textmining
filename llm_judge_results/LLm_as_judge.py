from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import faulthandler
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openai

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
try:  
    from deepeval.test_case import SingleTurnParams as LLMTestCaseParams
except ImportError:
    from deepeval.test_case import LLMTestCaseParams


DATA_ROOT = "/Users/fateme/Desktop/ncems_Tine_judge/NLP_train_test"
SPLIT = "train_set"
SDRF_DIR = ""
RESULTS_ROOT = "/Users/fateme/Desktop/ncems_Tine_judge/deepeval_proteomics_results"
EVALUATION_MODEL = "google/gemma-4-31b-it"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_TEMPERATURE = 0
MODEL_ENABLE_THINKING = True
MAX_COMPLETION_TOKENS = 16000
REQUEST_TIMEOUT = 300
DEGENERATE_RETRY_TEMPERATURE = 0.3
USE_LLM_JUDGE = True
MAX_WORKERS = 4
SHOW_DEEPEVAL_BANNER = True
THREAD_STACK_MB = 256
MAX_RETRIES = 3
RETRY_DELAY = 2
CACHE_ENABLED = True

PROMPT_VERSION = "proteomics_sdrf_judge_v3"
RECOVER_MISSED = True

SDRF_ADD_MISSING_COLUMNS = False

SAMPLE_SHEET_UNVERIFIABLE = True
UNVERIFIABLE_MIN_DISTINCT = 5
FIX_ACCESSION_DRIFT = True

SKIP_COLUMNS: set[str] = set()
ONLY_PXDS: list[str] = []
LIMIT: int | None = None
DEDUP_MIN_CHARS = 200
MAX_EXAMPLE_RUNS = 8
_thread_local = threading.local()

#field definitions one field per kind of SDRF column

_FIELD_GROUPS = [
    ("SAMPLE CHARACTERISTICS", [
        ("species",
         'Organism whose proteins the experiment analysed, as a scientific or common name in any '
         'case ("homo sapiens", "Homo sapiens", "human", "plasmodium falciparum"). May be inferred '
         'from human subjects (patient, donor, clinical sample, biopsy of a person) or from a cell '
         'line of known origin. In a host pathogen study it is the organism the study analyses. A '
         'host listed as well is right only when the paper analyses the proteins of both '
         'organisms. An organism that appears only in the search database is wrong. A cell line '
         'name belongs to cell_line.'),
        ("tissue",
         'Organism part the sample came from ("liver", "colon mucosa", "blood plasma"). For a tumor '
         'the organ of origin. For a cell line the organ of origin of the line may be inferred '
         '(HeLa cervix, HEK293 kidney, MCF-7 breast, A549 lung, HepG2 liver, Jurkat blood). For a '
         'unicellular organism or a pathogen grown in culture, the host cells or medium it grew in '
         'are wrong (rule D). A broader term that is still correct ("immune system" for a B cell '
         'line, "colon" for colon mucosa) is right when the paper names no narrower part. A '
         'species word in front of the term ("Human colon mucosa") is acceptable. A cell type '
         'belongs to cell_type.'),
        ("cell_type",
         'Biological cell type ("T cell", "hepatocyte", "epithelial cell"). For a cell line the cell '
         'type of the line may be inferred (MCF-7 epithelial cell, Jurkat T cell). A cell line name '
         'belongs to cell_line. An organ belongs to tissue. A life cycle stage of a parasite '
         '(schizont, trophozoite) belongs to developmental_stage. A value for a tissue sample with '
         'no stated cell type is unsupported.'),
        ("disease_state",
         'Disease of the subjects or material of the runs carrying this value ("colorectal '
         'cancer", "type 2 diabetes"). The SDRF term for healthy subjects is "normal". Other '
         'wording for healthy ("Normal healthy", "healthy") is a format error. In a case versus '
         'control study cases carry the disease and controls "normal". For a cell line the disease '
         'of origin may be inferred (HeLa cervical adenocarcinoma, MCF-7 breast adenocarcinoma, '
         'A549 lung adenocarcinoma, K562 chronic myeloid leukemia, Jurkat acute T cell leukemia, '
         'HepG2 hepatoblastoma, HEK293 normal).'),
        ("age",
         'Age of the subjects of the runs carrying this value, with its unit as stated ("45Y", '
         '"45 years", "8 weeks", "E14.5"). Never inferred. A stage word belongs to '
         'developmental_stage.'),
        ("sex",
         'Biological sex of the subjects of the runs carrying this value ("male", "female"). Never '
         'inferred, also not from a cell line.'),
        ("cell_line",
         'Name of an immortalized cell line as written ("HeLa", "HEK293T", "MCF-7"). Primary '
         'cells, mouse embryonic fibroblasts isolated by the authors, tissues and organism strains '
         'are wrong. A cell line used only outside the MS experiment is wrong (rule D).'),
        ("strain",
         'Strain of the organism ("C57BL/6J", "BALB/c", "3D7", "BY4741"). A cell line name belongs '
         'to cell_line. A genotype or knockout alone is wrong.'),
        ("material_type",
         'Broad class of the analysed material: "cell line", "primary cells", "tissue", '
         '"biofluid", "organoid", "whole organism" or "cell culture". May be inferred from the '
         'sample description.'),
        ("developmental_stage",
         'Developmental stage of the source ("adult", "embryonic", "fetal", "neonatal"), the life '
         'cycle stage of a parasite ("schizont", "ring stage") or the growth phase of a microbe '
         '("mid-log phase"). "adult" may be inferred from adult patients or donors or from a '
         'stated adult age.'),
        ("ethnicity",
         'Ancestry or ethnic group of the subjects as stated ("Caucasian", "East Asian"). Never '
         'inferred. The country of the hospital or of the authors is not an ethnicity (rule A).'),
    ]),
    ("TECHNICAL MASS SPECTROMETRY", [
        ("labeling",
         'Label of the runs carrying this value. "label free sample" is the SDRF term for label '
         'free runs. "label-free", "label free" and "LFQ" are equivalent wording. A labelled run '
         'carries its channel ("TMT126", "SILAC heavy"). A label that was not used is wrong.'),
        ("instrument",
         'Model of the primary LC MS/MS mass spectrometer ("Q Exactive Plus", "LTQ Orbitrap '
         'Velos", "Orbitrap Fusion Lumos", "timsTOF Pro"). The NT term is judged, the AC accession '
         'is not. A vendor prefix is optional. A different model of the same family is a wrong '
         'detail ("Q Exactive" when the paper prints "Q Exactive Plus"). The LC system and an '
         'instrument used only for validation are wrong (rule D).'),
        ("cleavage_agent",
         'Protease or chemical cleavage agent used to digest the proteins ("Trypsin", "Lys-C"). '
         'Product names are equivalent ("Trypsin Gold", "sequencing grade trypsin" and "Trypsin/P" '
         'are trypsin). One cell names one agent. When two agents were used, a cell naming one of '
         'them is right.'),
        ("fragmentation_method",
         'Fragmentation method of the runs carrying this value ("HCD", "CID", "ETD", "EThcD"). '
         'Spelled out names are equivalent. ETD with supplemental activation is ETD. One cell names '
         'one method. When the listed runs used several methods (a decision tree choosing CID or '
         'ETD per precursor, ETD triggered after CID), a cell naming one method that those runs '
         'used is right (HIGH). It is never incomplete for that reason. A method those runs did '
         'not use is wrong. An instrument "equipped with ETD" only names hardware: it is not '
         'evidence that ETD was used for these runs.'),
        ("collision_energy",
         'Collision energy as stated ("NCE 27", "27%", "35% normalized collision energy", "28 '
         'eV"). Never inferred.'),
        ("enrichment_method",
         'Enrichment of a protein or peptide subset before MS. The SDRF controlled terms '
         '"enrichment of phosphorylated protein", "enrichment of glycosylated protein" and "no '
         'enrichment" are right when they match the paper, and so is the method name ("IMAC", '
         '"TiO2", "Phos-Select IMAC", "anti-phosphotyrosine antibody"). A controlled term is never '
         'replaced by the method name or the reverse. Desalting, depletion of abundant proteins '
         'and offline fractionation are not enrichment (rule D).'),
        ("fractionation_method",
         'Offline separation of each sample into fractions before LC MS ("high pH reversed '
         'phase", "SCX", "SDS-PAGE gel slices"). The online LC gradient, desalting and splitting a '
         'sample into aliquots for repeated runs are not fractionation (rule D).'),
        ("reduction_reagent",
         'Chemical that reduced disulfide bonds ("DTT", "TCEP"). DTT or TCEP in the lysis buffer '
         'counts when no separate reduction step is described.'),
        ("alkylation_reagent",
         'Chemical that alkylated cysteines ("iodoacetamide", "chloroacetamide", "NEM"). Cy dyes '
         'of 2D-DIGE are not alkylation reagents (rule D).'),
        ("ionization_type",
         'Ionization source of the primary instrument ("ESI", "nanoESI", "MALDI"). "nanospray" is '
         'nanoESI. May be inferred from a named ion source or emitter, never from the instrument '
         'model alone.'),
        ("mass_analyzer",
         'Mass analyzer type of the primary instrument ("Orbitrap", "linear ion trap", "QTOF", '
         '"TOF", "triple quadrupole"). May be inferred from the instrument model. For a hybrid '
         'instrument a cell naming one of its analyzers is right.'),
        ("acquisition_method",
         'MS acquisition scheme: "DDA", "DIA", "PRM", "SRM" or "MRM". May be inferred from a named '
         'strategy (data dependent, top N, decision tree or neutral loss triggered acquisition give '
         '"DDA". SWATH gives "DIA").'),
        ("technology_type",
         '"proteomic profiling by mass spectrometry" is the SDRF term for MS based proteomics. '
         '"bottom-up proteomics" and "shotgun proteomics" are equivalent for a digested DDA or DIA '
         'workflow. May be inferred from the workflow.'),
        ("modification",
         'ONE modification set in the database search, as an SDRF cell "NT=<name>;AC=<UNIMOD '
         'id>;TA=<residues>" or "PP=<position>" and "MT=<Fixed or Variable>". Compare with the '
         'search parameters the paper prints, including modifications added in an additional '
         'search ("also searched with the addition of Phospho Y" makes Phospho Y a searched '
         'modification). HALLUCINATED (rule A) ONLY when the named modification (NT) is not '
         'searched at all. When the named modification is searched but a detail differs, it is a '
         'wrong detail (d, MEDIUM) with the corrected cell, never hallucinated: residues the paper '
         'does not give (TA=M,W,H when the paper says Oxidation (M), TA=S,T,Y,A when the paper '
         'gives S, T and Y), a position where the paper gives residues or the reverse, or a fixed '
         'or variable status that contradicts the paper. The same name on another site that the '
         'paper does not search (Acetyl on K when only Acetyl (Protein N-term) is searched) is '
         'hallucinated. The AC accession is not judged.'),
        ("precursor_mass_tolerance",
         'Precursor (MS1) mass tolerance of the database search with its unit as printed ("10 '
         'ppm", "20 ppm"). When the paper prints a precursor tolerance and the cell holds another '
         'number, it is a wrong detail (d, MEDIUM) with the printed value. HALLUCINATED only when '
         'the paper prints no precursor tolerance at all.'),
        ("fragment_mass_tolerance",
         'Fragment (MS/MS) mass tolerance of the database search with its unit as printed ("0.5 '
         'Da", "20 ppm", "0.02 Da"). When the paper prints a fragment tolerance and the cell holds '
         'another number, it is a wrong detail (d, MEDIUM) with the printed value. HALLUCINATED '
         'only when the paper prints no fragment tolerance at all.'),
    ]),
    ("RUN STRUCTURE (one value per run)", [
        ("biological_replicate",
         'Biological replicate index of each run. Independent biological units of one condition '
         'get 1, 2, 3. Runs of one pooled or the same biological sample (aliquots, repeated '
         'injections, acquisition methods compared on one sample) all get the same index. '
         'Separate biopsies, animals, donors or cultures are independent units.'),
        ("technical_replicate",
         'Technical replicate index of each run. Repeated runs of the same material get 1, 2, 3. '
         'A material measured once gets 1. Separate elutions or enrichment steps of one sample '
         '(a first and a second TiO2 elution) are different material: they are fractions, not '
         'technical replicates.'),
        ("fraction_identifier",
         'Fraction index of each run. Every run gets 1 when no offline fractionation is '
         'described. Separate elutions, enrichment steps or gel bands of one sample that were run '
         'separately are fractions 1, 2, 3. Conditions, preservation methods, aliquots and '
         'repeated runs are never fractions.'),
        ("individual",
         'Identifier of the subject or donor each run comes from ("1", "A", "patient 3"). Runs '
         'from the same person share the identifier. Needs the paper to state the subjects.'),
        ("factor_value",
         'Level of the studied variable for each run. The column header names the variable '
         '("factor value[preservation method]"). The variable must be the one the study compares '
         '(rule D when it is not). Each run must carry the level of the group it belongs to.'),
    ]),
]

_FIELD_DEFINITIONS: dict[str, str] = {}
for _label, _items in _FIELD_GROUPS:
    for _name, _text in _items:
        _FIELD_DEFINITIONS[_name] = _text
_CANONICAL_FIELD = {n.lower(): n for n in _FIELD_DEFINITIONS}
_FIELD_ORDER = [n for _, items in _FIELD_GROUPS for n, _ in items]
_FIELD_GROUP_OF = {n: label for label, items in _FIELD_GROUPS for n, _ in items}
_RUN_FIELDS = {n for n, _ in _FIELD_GROUPS[2][1]}
_GENERIC_FIELD = "other_column"


def _render_definitions() -> str:
    blocks = []
    for label, items in _FIELD_GROUPS:
        lines = [f"{label}:"]
        for name, text in items:
            lines.append(f"{name} : {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _field_def_for(field: str, header: str = "") -> str:
    if field == _GENERIC_FIELD:
        return (f"{_GENERIC_FIELD} : SDRF column \"{header}\". Judge whether the value is what a "
                f"column with this name should hold for this study as the paper describes it. A "
                f"value without any meaning for that column is wrong (rule D).")
    return f"{field} : {_FIELD_DEFINITIONS.get(field, 'Metadata field ' + field)}"


#sdrf column classification

def _sdrf_header_key(h: str) -> str:
    h = h.lower().replace("characteristics", "").replace("comment", "")
    return re.sub(r"[^a-z]", "", h)


#normalised header -> field
_SDRF_COLUMN_FIELDS = {
    "organism": "species", "organismpart": "tissue", "celltype": "cell_type",
    "disease": "disease_state", "age": "age", "sex": "sex", "cellline": "cell_line",
    "materialtype": "material_type", "developmentalstage": "developmental_stage",
    "ancestrycategory": "ethnicity", "ethnicity": "ethnicity",
    "individual": "individual", "donor": "individual",
    "label": "labeling", "instrument": "instrument",
    "cleavageagent": "cleavage_agent", "cleavageagentdetails": "cleavage_agent",
    "fragmentationmethod": "fragmentation_method", "dissociationmethod": "fragmentation_method",
    "collisionenergy": "collision_energy",
    "enrichmentmethod": "enrichment_method", "enrichmentprocess": "enrichment_method",
    "fractionationmethod": "fractionation_method",
    "reductionreagent": "reduction_reagent", "alkylationreagent": "alkylation_reagent",
    "ionizationtype": "ionization_type",
    "massanalyzer": "mass_analyzer", "massanalyzertype": "mass_analyzer",
    "acquisitionmethod": "acquisition_method", "technologytype": "technology_type",
    "modification": "modification", "modificationparameters": "modification",
    "precursormasstolerance": "precursor_mass_tolerance",
    "fragmentmasstolerance": "fragment_mass_tolerance",
    "biologicalreplicate": "biological_replicate", "technicalreplicate": "technical_replicate",
    "fractionidentifier": "fraction_identifier",
}
#identifier columns that are copied unchanged and never judged
_IDENTIFIER_KEYS = {"sourcename", "assayname", "datafile", "fileuri", "fileurl",
                    "proteomexchangeaccessionnumber",
                    #metadata about the SDRF file itself never stated in a paper
                    "toolmetadata", "sdrfversion", "sdrftemplate", "sdrfannotationtool",
                    "sdrftool", "toolversion"}

#column added for a recovered field when the input SDRF has none 
_SDRF_STANDARD_HEADER = {
    "species": "characteristics[organism]",
    "tissue": "characteristics[organism part]",
    "cell_type": "characteristics[cell type]",
    "disease_state": "characteristics[disease]",
    "age": "characteristics[age]",
    "sex": "characteristics[sex]",
    "cell_line": "characteristics[cell line]",
    "strain": "characteristics[strain/breed]",
    "material_type": "characteristics[material type]",
    "developmental_stage": "characteristics[developmental stage]",
    "ethnicity": "characteristics[ancestry category]",
    "labeling": "comment[label]",
    "instrument": "comment[instrument]",
    "cleavage_agent": "comment[cleavage agent details]",
    "fragmentation_method": "comment[fragmentation method]",
    "collision_energy": "comment[collision energy]",
    "enrichment_method": "comment[enrichment method]",
    "fractionation_method": "comment[fractionation method]",
    "reduction_reagent": "comment[reduction reagent]",
    "alkylation_reagent": "comment[alkylation reagent]",
    "ionization_type": "comment[ionization type]",
    "mass_analyzer": "comment[mass analyzer type]",
    "acquisition_method": "comment[acquisition method]",
    "technology_type": "technology type",
    "modification": "comment[modification parameters]",
    "precursor_mass_tolerance": "comment[precursor mass tolerance]",
    "fragment_mass_tolerance": "comment[fragment mass tolerance]",
}
#fields written as NT= cells in a new column
_KV_FIELDS = {"instrument", "cleavage_agent", "modification", "labeling"}
#fields the recovery pass may fill (one value for every run)
_RECOVERABLE_FIELDS = [f for f in _FIELD_ORDER if f not in _RUN_FIELDS]


def classify_column(header: str) -> tuple[str, str]:
    """(kind field) for an SDRF column -> kind is value modification runs or skip"""
    key = _sdrf_header_key(header)
    if header.strip().lower() in {c.lower() for c in SKIP_COLUMNS}:
        return "skip", ""
    if (key in _IDENTIFIER_KEYS or "file" in key or "accession" in key
            or key.endswith("uri") or key.endswith("url")):
        return "skip", ""
    if key.startswith("factorvalue"):
        return "runs", "factor_value"
    if key.endswith("strain") or key.endswith("strainbreed"):
        return "value", "strain"
    field = _SDRF_COLUMN_FIELDS.get(key)
    if field == "modification":
        return "modification", field
    if field in _RUN_FIELDS:
        return "runs", field
    if field:
        return "value", field
    return "value", _GENERIC_FIELD


#judge prompts

_STATIC_SYSTEM_PROMPT = """\
You are an expert in mass spectrometry based proteomics and in SDRF-Proteomics metadata.
An extraction pipeline read a proteomics paper and wrote an SDRF file (one row per MS
run). Your task is to judge one value of that SDRF file against the paper text and the
field definitions given at the end of these instructions.

You receive two inputs:
  1. A PAPER TEXT message holding the abstract and methods of the paper. It is the only
     source of truth.
  2. An assessment request as a JSON object with a "task" key.

=================================================================
TASK TYPE: "verify"   (one value of one SDRF column)
=================================================================
JSON input fields:
  field_name              the metadata field the column holds
  sdrf_column             the column header in the SDRF file
  field_definition        the definition and format of that field
  candidate_value         the SDRF cell to judge, exactly as written
  readable_value          the same cell in plain words
  cell_style              "key-value" when the column writes NT=...;AC=... cells
  other_values_in_column  the other values of the same column (other runs)
  runs_with_value         how many runs carry this value, and some of them
  checks                  automatic checks of the value (may be absent)
  related_columns         values of neighbouring columns (may be absent)

One value describes ALL the runs listed with it. Judge whether it is right for those
runs. Other runs may carry other values of the same column. That is normal when the
study has groups (cases and controls, two fragmentation schemes).

=================================================================
TASK TYPE: "verify_runs"   (one per run column, judged as a whole)
=================================================================
JSON input fields:
  field_name, sdrf_column, field_definition as above
  runs           one line per run: "<run key> = <value>" plus the other columns of that
                 run that tell the runs apart (assay, source, data file)
  candidate_value the same mapping on one line: "<run key> = <value> | ..."

Judge the whole mapping. It is right only when every run carries the right value. The
index values are labels: what matters is which runs share a value and which differ.

=================================================================
SDRF CONVENTIONS
=================================================================
SDRF cells use controlled terms. Lowercase organism names ("homo sapiens") are normal.
Ontology style cells "NT=<term>;AC=<accession>" are judged on the NT term. "not
available" and "not applicable" cells are never sent to you.
ONE VALUE PER CELL. An SDRF cell holds one value for its run. When the runs used several
values of a field (CID and ETD in one decision tree run, two proteases, the two analyzers
of a hybrid instrument), a cell naming ONE value that those runs used is right and
complete. It is HIGH. Never mark it incomplete and never correct it to a list. A cell is
wrong only when its value was not used for those runs.

=================================================================
SCOPE
=================================================================
Every value describes the proteomics LC MS experiment of the paper. Papers also describe
western blots, imaging, immunohistochemistry validation cohorts, transcriptomics or a
MALDI instrument used only to check gel spots. A value that belongs only to such another
experiment is printed but wrong (rule D). Enriched runs and proteome profiling runs of
the same study both belong to the MS experiment.
The search database lists organisms for spectrum matching. It does not define the sample
species. Author affiliations, supplier addresses, cited studies and the introduction
never describe the analysed samples.

=================================================================
CORE EVALUATION QUESTIONS (answer in order)
=================================================================
1. TYPE CHECK     Does the value belong to field_name?
                  If it is the right kind of information for ANOTHER field of the
                  definitions list, set TYPE_CORRECT: no and name that field (rule B).
                  Examples: a cell line name in the organism column belongs to
                  cell_line. A parasite life cycle stage in the cell type column belongs
                  to developmental_stage.
                  If it fits no field at all, keep TYPE_CORRECT: yes and judge it under
                  rule D.
2. SOURCE CHECK   Is the content of the value stated in the paper text?
                  Absent or unsupported means HALLUCINATED: yes (rule A).
                  Inference is allowed ONLY where the field definition allows it and
                  only from a basis printed in the paper:
                    species from human subjects or a cell line of known origin.
                    tissue, cell_type and disease_state from a cell line of known origin
                    as listed in the definitions, tissue from a named cancer type.
                    disease_state "normal" when the subjects are healthy.
                    material_type from the sample description.
                    developmental_stage "adult" from adult subjects.
                    acquisition_method from a named strategy.
                    mass_analyzer from the instrument model.
                    ionization_type from a named ion source or emitter.
                    labeling "label free sample" when no label is used.
                    technology_type from the workflow.
                    run structure values from the counts and the design the paper prints.
                  Any other value supplied from background knowledge or a software
                  default is HALLUCINATED: yes. Examples: the sex of a cell line donor,
                  an ethnicity taken from the hospital country, a mass tolerance or a
                  modification that the paper never states.
3. TRUTH CHECK    Is the value right for the runs that carry it? A value that is printed
                  but belongs to another experiment, another step, another material or a
                  passing mention is VALUE_CORRECT: no with HALLUCINATED: no (rule D).
                  Examples: the host erythrocytes as organism part of a cultured
                  parasite, a MALDI instrument used for spot checks as instrument.
4. COMPLETENESS   Does the value follow the format of its definition? A cell is complete
                  when it holds everything its own definition asks for (for a
                  modification its residues or position and its MT status). A cell that
                  names one of several values used by those runs is complete (ONE VALUE
                  PER CELL).

=================================================================
VERDICT RULES
=================================================================
LOW     use when ANY of:
  (A) HALLUCINATED: the content is not stated in the paper and is not an allowed
      inference.
  (B) TYPE MISMATCH: the value belongs to another field of the definitions list.
  (D) WRONG ITEM: the value is printed but describes another experiment, another step,
      another material or a passing mention. Also a run mapping that cannot be fixed
      from the paper.

MEDIUM  use when the type is correct and the item is right and supported but the value
        has a CLEAR error that the field definition names and that the paper lets you
        fix exactly:
  (a) FORMAT: right content in the wrong form ("Normal healthy" for "normal", a
      description where a plain number is required).
  (b) INCOMPLETE: a part that the cell's own definition requires is missing (a
      modification without its MT status). Naming one of several values used by the
      runs is NOT incomplete (ONE VALUE PER CELL).
  (c) DRIFT: a different model or term of the same family than the paper prints ("Q
      Exactive" when the paper prints "Q Exactive Plus").
  (d) WRONG DETAIL: the right item with one wrong detail that the paper prints
      correctly. A wrong residue list, a fixed status for a variable modification, a
      tolerance number other than the printed one, a run mapping where runs share or
      split an index against the design the paper describes. A wrong detail is never
      HALLUCINATED: the item itself is in the paper.
  (e) WRONG CHOICE: a controlled value holds the wrong choice and the paper supports
      another allowed value.
  CORRECTED_VALUE gives the whole fixed value:
    verify        one SDRF cell in the cell_style of the column. For a key-value column
                  write "NT=<term>" and keep the other parts (TA, PP, MT) that apply.
    verify_runs   the whole mapping on ONE line: "<run key> = <value> | <run key> =
                  <value> | ..." with every run key exactly as given.
  It uses only words and numbers printed in the paper, the allowed values of the
  definition and the inferences the definition allows. If you cannot state the fix this
  way, choose LOW when the value is wrong as written and HIGH when it is acceptable.

HIGH    use when the type is correct, the content is supported, it is right for the runs
        that carry it, it follows the field format and it is complete.

=================================================================
KEEP CORRECT VALUES AS THEY ARE
=================================================================
The filter replaces every MEDIUM value with your CORRECTED_VALUE. A needless fix turns a
right value into a wrong one. So:
  A value that is a correct reading of the paper is HIGH, even when you would have
  worded it differently.
  When the paper can be read in more than one way, keep the SDRF value.
  Never use MEDIUM to swap a synonym, to change the case, to add a vendor name, to turn
  one value into a list, to replace an SDRF controlled term by a method name or to add
  details that the definition does not require.

=================================================================
NEVER A REASON TO DOWNGRADE
=================================================================
  Lowercase or uppercase terms ("homo sapiens" and "Homo sapiens").
  Common and scientific names ("human" and "Homo sapiens").
  Abbreviations and their expansions ("HCD", "IAA" and "iodoacetamide").
  A vendor prefix in instrument ("Thermo Q Exactive Plus" and "Q Exactive Plus").
  Product names of the same protease ("Trypsin Gold" and "Trypsin").
  "label free sample", "label-free" and "LFQ". "nanospray" and "nanoESI".
  SDRF controlled terms ("enrichment of phosphorylated protein", "proteomic profiling by
  mass spectrometry", "label free sample").
  The same quantity in another unit or prefix ("0.1 uM" and "100 nM", "u" and "micro"
  for micro, "mM" and "mmol/L"). Convert before you compare numbers.
  One of several values that the runs used (ONE VALUE PER CELL).
  A broader term that is still correct when the paper names nothing narrower.
  An AC accession, which is not judged.
  A value found in the abstract instead of the methods.
  Other values in the same column for other runs.

=================================================================
OUTPUT FORMAT
=================================================================
Plain text only. No markdown, no bold, no bullets. Write each label once, in this
order, each on its own line:

TYPE CHECK:        <one sentence. If the type is wrong, name the correct field>
SOURCE CHECK:      <one sentence. Where the paper states it or that it is absent>
TRUTH CHECK:       <one sentence. Is it right for the runs that carry it>
COMPLETENESS CHECK:<one sentence on format and completeness>
TYPE_CORRECT:      yes | no
CORRECT_TYPE_NAME: <the correct field name when TYPE_CORRECT is no, otherwise NONE>
VALUE_CORRECT:     yes | no
VALUE_COMPLETE:    yes | no
HALLUCINATED:      yes | no
VERDICT:           high | medium | low
CORRECTED_VALUE:   <for medium, the whole fixed value on one line. Otherwise NONE>

Field meanings:
  TYPE_CORRECT      no ONLY when the value belongs to another field (rule B).
  CORRECT_TYPE_NAME a field name from the definitions list, spelled exactly.
  VALUE_CORRECT     yes when the content as written is right for this field.
                    no for rules A, B, D and for a wrong detail (d) or choice (e).
  VALUE_COMPLETE    no when an item that the paper gives is missing (b).
  HALLUCINATED      yes ONLY when the content is absent from the paper and is not an
                    allowed inference.
  CORRECTED_VALUE   the value that replaces the candidate. Never an explanation.

Typical couplings:
  HIGH                 TYPE_CORRECT yes  VALUE_CORRECT yes  VALUE_COMPLETE yes  HALLUCINATED no
  MEDIUM format/drift  TYPE_CORRECT yes  VALUE_CORRECT yes  VALUE_COMPLETE yes  HALLUCINATED no
  MEDIUM incomplete    TYPE_CORRECT yes  VALUE_CORRECT yes  VALUE_COMPLETE no   HALLUCINATED no
  MEDIUM wrong detail  TYPE_CORRECT yes  VALUE_CORRECT no   VALUE_COMPLETE yes  HALLUCINATED no
  LOW hallucinated     TYPE_CORRECT yes  VALUE_CORRECT no   VALUE_COMPLETE no   HALLUCINATED yes
  LOW type mismatch    TYPE_CORRECT no   VALUE_CORRECT no   VALUE_COMPLETE no   HALLUCINATED no
  LOW wrong item       TYPE_CORRECT yes  VALUE_CORRECT no   VALUE_COMPLETE no   HALLUCINATED no

=================================================================
WORKED EXAMPLES
=================================================================

Example 1. HIGH: allowed inference in SDRF case
field_name=species, candidate_value="homo sapiens"
paper: "Colon mucosal biopsies were sampled from two healthy participants"
TYPE CHECK:        An organism name, the right field.
SOURCE CHECK:      Human participants are stated, an allowed inference.
TRUTH CHECK:       The analysed biopsies are human.
COMPLETENESS CHECK:Complete. Lowercase is the SDRF convention.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high
CORRECTED_VALUE:   NONE

Example 2. MEDIUM (c): a related model of the same family
field_name=instrument, candidate_value="AC=MS:1001911;NT=Q Exactive", cell_style=key-value
paper: "The system was coupled online to a Q Exactive Plus mass spectrometer."
TYPE CHECK:        A mass spectrometer model, the right field.
SOURCE CHECK:      The methods name the instrument.
TRUTH CHECK:       The paper prints the Plus model.
COMPLETENESS CHECK:Drift to the base model, fixed to the printed model.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   NT=Q Exactive Plus

Example 3. LOW (B): a life cycle stage in the cell type column
field_name=cell_type, candidate_value="schizont"
paper: "We analysed the Plasmodium falciparum schizont phosphoproteome"
TYPE CHECK:        Schizont is a parasite life cycle stage. The correct field is
                   developmental_stage.
SOURCE CHECK:      Present in the abstract.
TRUTH CHECK:       Not a cell type.
COMPLETENESS CHECK:Not applicable, wrong field.
TYPE_CORRECT:      no
CORRECT_TYPE_NAME: developmental_stage
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low
CORRECTED_VALUE:   NONE

Example 4. LOW (D): host material as organism part of a cultured parasite
field_name=tissue, candidate_value="human erythrocytes"
paper: "P. falciparum 3D7 was cultured in human erythrocytes and parasites were
collected by saponin lysis, which removes the vast majority of erythrocyte material."
TYPE CHECK:        A tissue or cell material, the right kind of value.
SOURCE CHECK:      Erythrocytes are printed.
TRUTH CHECK:       They are the culture host that was removed, not the analysed material.
COMPLETENESS CHECK:Not applicable, wrong material.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      no
VERDICT:           low
CORRECTED_VALUE:   NONE

Example 5. LOW (A): a modification that was never searched
field_name=modification, candidate_value="NT=Ammonia-loss;AC=UNIMOD:385;TA=Q,C;MT=Variable"
paper: "Acetyl (Protein N-term), Oxidation (M), Deamidated (NQ), Carbamidomethyl (C)
and Phospho (ST) set as variable modifications"
TYPE CHECK:        A modification, the right field.
SOURCE CHECK:      Ammonia loss is not among the searched modifications.
TRUTH CHECK:       Cannot be accepted.
COMPLETENESS CHECK:Not applicable, hallucinated.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    no
HALLUCINATED:      yes
VERDICT:           low
CORRECTED_VALUE:   NONE

Example 6. MEDIUM (d): a fixed status for a variable modification
field_name=modification, candidate_value="NT=Carbamidomethyl;AC=UNIMOD:4;TA=C;MT=Fixed"
paper: same search parameters as example 5
TYPE CHECK:        A modification, the right field.
SOURCE CHECK:      Carbamidomethyl (C) is searched.
TRUTH CHECK:       It was set as variable, not fixed.
COMPLETENESS CHECK:Fixed to the printed status.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   NT=Carbamidomethyl;TA=C;MT=Variable

Example 7. HIGH: one value per cell for runs that used two methods
field_name=fragmentation_method, candidate_value="ETD", runs_with_value="3 of 6 runs
(the decision tree runs)", other_values_in_column=["CID"]
paper: "the top 10 most abundant ions were subjected to either CID or ETD fragmentation
... Triplicate DT experiments were performed"
TYPE CHECK:        A fragmentation method, the right field.
SOURCE CHECK:      Stated in the MS section.
TRUTH CHECK:       The decision tree runs used ETD.
COMPLETENESS CHECK:One cell names one method. ETD was used in these runs, so it is right.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           high
CORRECTED_VALUE:   NONE

Example 7b. MEDIUM (d): a searched modification with a wrong residue, not hallucinated
field_name=modification, candidate_value="NT=Phospho;AC=UNIMOD:21;TA=S,T,Y,A;MT=Variable"
paper: "Phospho (ST) set as variable modifications ... also searched with the addition
of Phospho Y"
TYPE CHECK:        A modification, the right field.
SOURCE CHECK:      Phospho is searched on S and T, and on Y in the additional search.
TRUTH CHECK:       Alanine is not a searched residue. The modification itself is right.
COMPLETENESS CHECK:Fixed to the printed residues.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   NT=Phospho;TA=S,T,Y;MT=Variable

Example 7c. MEDIUM (d): a tolerance other than the printed one
field_name=fragment_mass_tolerance, candidate_value="1.0005 Da"
paper: "10 ppm for MS mass tolerance, 0.5 Da for MS/MS mass tolerance"
TYPE CHECK:        A fragment mass tolerance, the right field.
SOURCE CHECK:      The paper prints a fragment tolerance of 0.5 Da.
TRUTH CHECK:       The cell holds another number than the printed one.
COMPLETENESS CHECK:Fixed to the printed tolerance.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   0.5 Da

Example 8. MEDIUM (d): aliquots of one pooled sample as biological replicates
task=verify_runs, field_name=biological_replicate
candidate_value="DT_01.raw = 1 | DT_02.raw = 2 | DT_03.raw = 3"
paper: "Phosphopeptides from two IMAC purifications were pooled and split into 6
aliquots for LC MS/MS analysis. Triplicate DT experiments were performed."
TYPE CHECK:        Biological replicate indices, the right field.
SOURCE CHECK:      The paper describes one pooled sample.
TRUTH CHECK:       The runs are aliquots of one biological sample, not independent units.
COMPLETENESS CHECK:Every run gets the same biological replicate index.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   DT_01.raw = 1 | DT_02.raw = 1 | DT_03.raw = 1

Example 9. MEDIUM (d): conditions written as fractions
task=verify_runs, field_name=fraction_identifier
candidate_value="A_DF_1.raw = 1 | A_RNAlater_1.raw = 2 | A_FFPE_1.raw = 3"
paper describes four preservation methods and no offline fractionation.
TYPE CHECK:        Fraction indices, the right field.
SOURCE CHECK:      No fractionation is described.
TRUTH CHECK:       The indices encode the preservation method, not fractions.
COMPLETENESS CHECK:Every run gets fraction 1.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     no
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   A_DF_1.raw = 1 | A_RNAlater_1.raw = 1 | A_FFPE_1.raw = 1

Example 10. MEDIUM (a): SDRF term for healthy subjects
field_name=disease_state, candidate_value="Normal healthy"
paper: "two gastroenterologically healthy participants"
TYPE CHECK:        A disease state, the right field.
SOURCE CHECK:      Healthy participants are stated.
TRUTH CHECK:       The subjects are healthy.
COMPLETENESS CHECK:The SDRF term for healthy subjects is normal.
TYPE_CORRECT:      yes
CORRECT_TYPE_NAME: NONE
VALUE_CORRECT:     yes
VALUE_COMPLETE:    yes
HALLUCINATED:      no
VERDICT:           medium
CORRECTED_VALUE:   normal

=================================================================
FIELD DEFINITIONS
=================================================================
""" + _render_definitions()


_RECOVER_SYSTEM_PROMPT = """\
You are an expert in mass spectrometry based proteomics and SDRF-Proteomics metadata. An
extraction pipeline wrote an SDRF file for a proteomics paper and a reviewer has checked
its values. Your task is to find values that the paper states but that the SDRF file
does not hold.

You receive two inputs:
  1. A PAPER TEXT message holding the abstract and methods of the paper. It is the only
     source of truth.
  2. A JSON request with these keys:
     task           always "recover"
     field_group    the name of the group of fields
     open_fields    fields with no accepted value in the SDRF, each with:
       definition             the definition and format of the field
       rejected_values        values the reviewer removed (may be absent)
       moved_here_by_review   values found in another column that belong here (may be absent)
     modification   present in the technical group only:
       accepted_values        the modifications the SDRF holds after review
       rejected_values        modifications the reviewer removed
     filled_fields  the accepted values of the other fields of the group, for context

=================================================================
WHAT COUNTS AS A MISSED VALUE
=================================================================
A value that the paper states for the proteomics LC MS experiment, that holds for every
run of the study, that fits the field definition and that you can support with a quote.
  Most open fields are empty because the paper says nothing about them. Returning
  nothing is normal.
  An inference counts only where the field definition allows it and only from a printed
  statement that you quote.
  A value in moved_here_by_review is a strong candidate. Return it when the paper
  supports it for this field.
  For modification: every modification the search parameters list that accepted_values
  does not hold yet, including modifications added in an additional search ("also
  searched with the addition of Phospho Y"). One object per modification, as an SDRF cell
  "NT=<name>;TA=<residues>;MT=<Fixed or Variable>" or with PP=<position>.

=================================================================
WHAT NEVER COUNTS
=================================================================
  A rejected value or a rewording of one.
  A value that differs between groups of runs. Those are not filled in here (this does
  not apply to modifications).
  A value inferred from background knowledge or a software default.
  A value of another experiment (western blot, imaging, IHC validation, transcriptomics,
  a MALDI spot check) or of the search database.
  A supplier address, an author address, a cited study or the introduction.
  A value you are not sure about. A wrong addition costs more than a missed value.

=================================================================
VALUES AND EVIDENCE
=================================================================
One value per field (several for modification), written as an SDRF cell in the format of
its field definition. Every value needs evidence: one contiguous span copied character
for character from the paper, at most 40 words. Do not join sentences. Do not shorten
with dots. Do not fix typos. If you cannot quote such a span, do not return the value.

=================================================================
OUTPUT FORMAT
=================================================================
Return only one JSON object and no other text:
{"missed": [{"field": "<field name>", "value": "<value>", "evidence": "<verbatim quote>"}]}
Use the field names exactly as given. Return {"missed": []} when nothing is missing.

=================================================================
EXAMPLES
=================================================================
Example 1. A reduction reagent and a searched modification the SDRF left out
open_fields: reduction_reagent. modification accepted_values: Oxidation (M)
paper: "Parasite pellets were re-suspended in extraction buffer (4% SDS, 0.1 M DTT,
0.1 M Tris pH 8)" and "with Acetyl (Protein N-term), Oxidation (M) set as variable
modifications"
output: {"missed": [
 {"field": "reduction_reagent", "value": "DTT", "evidence": "Parasite pellets were re-suspended in extraction buffer (4% SDS, 0.1 M DTT, 0.1 M Tris pH 8"},
 {"field": "modification", "value": "NT=Acetyl;PP=Protein N-term;MT=Variable", "evidence": "with Acetyl (Protein N-term), Oxidation (M) set as variable modifications"}]}

Example 2. Nothing is missing
open_fields: ethnicity, age
paper describes two healthy participants and gives no ancestry or age.
output: {"missed": []}
"""


#geval criteria string used when judging whether an SDRF value is correct
_ANNOTATION_GEVAL_CRITERIA = (
    "Judge whether one value of an SDRF-Proteomics file is right for the runs that carry it, "
    "based only on the paper text and the field definitions. "
    "Evaluate in order. "
    "(1) TYPE: does the value belong to this field? If it belongs to another field of the list, "
    "TYPE_CORRECT is no (rule B). The wrong instance of the right kind of thing is rule D. "
    "(2) SOURCE: is its content stated in the paper? Inference is allowed only where the field "
    "definition allows it, from a printed basis. Anything else that is absent is HALLUCINATED "
    "(rule A). "
    "(3) TRUTH: is it right for the runs that carry it and for the proteomics LC MS experiment? "
    "Values from another experiment, another step, the search database or a passing mention are "
    "wrong (rule D). "
    "(4) COMPLETENESS: does it follow the field format and hold every part its own definition "
    "asks for? One cell holds one value: naming one of several values used by those runs is "
    "complete. "
    "HIGH means right field, supported, right runs, right format and complete. "
    "MEDIUM means a clear error the paper lets you fix exactly: format, incomplete, drift, a wrong "
    "detail or a wrong choice. CORRECTED_VALUE gives the whole fixed value in SDRF form. A value "
    "that is a correct reading of the paper stays HIGH and is never reworded. "
    "LOW means hallucinated, another field or the wrong item."
)

#step by step instructions for GEval when evaluating sdrf value
_ANNOTATION_GEVAL_STEPS = [
    "TYPE CHECK: decide whether the value is the kind of information this field holds. If it "
    "belongs to another field of the definitions list, name that field and set TYPE_CORRECT: no.",
    "SOURCE CHECK: find the statement in the paper that supports the value. Accept inference "
    "only where the field definition allows it. Otherwise set HALLUCINATED: yes.",
    "TRUTH CHECK: confirm that the value is right for the runs that carry it and describes the "
    "proteomics LC MS experiment, not another experiment, another step or the search database.",
    "COMPLETENESS CHECK: compare the value with the field format. One cell holds one value, so "
    "one of several values used by those runs is complete. Use MEDIUM with a CORRECTED_VALUE for "
    "a format error, a missing required part, drift, a wrong detail or a wrong choice that the "
    "paper corrects. A wrong detail of a searched item is never hallucinated.",
    "Output TYPE CHECK, SOURCE CHECK, TRUTH CHECK, COMPLETENESS CHECK, TYPE_CORRECT, "
    "CORRECT_TYPE_NAME, VALUE_CORRECT, VALUE_COMPLETE, HALLUCINATED, VERDICT and CORRECTED_VALUE.",
]

_EXPECTED_OUTPUT_TEXT = (
    "Assess the candidate_value against the paper text and the field definition: the right field "
    "(TYPE_CORRECT yes/no), stated in the paper or an allowed inference (HALLUCINATED yes/no), "
    "right for the runs that carry it and the proteomics LC MS experiment (VALUE_CORRECT yes/no), "
    "complete in the field format (VALUE_COMPLETE yes/no). "
    "Verdict: high (all pass) | medium (format, incomplete, drift, wrong detail or wrong choice "
    "with a CORRECTED_VALUE in SDRF form) | low (hallucinated, another field or the wrong item)."
)


#fingerprint of everything that changes the judge's answer it is part of every
#disk cache key so editing the prompt or switching model never reuses old verdicts
_PROMPT_FINGERPRINT = hashlib.sha256("|".join([
    PROMPT_VERSION, _STATIC_SYSTEM_PROMPT, _RECOVER_SYSTEM_PROMPT, EVALUATION_MODEL,
    str(MODEL_TEMPERATURE), str(MODEL_ENABLE_THINKING),
]).encode("utf-8")).hexdigest()[:16]

_CACHE_MODEL_SLUG = EVALUATION_MODEL.replace("/", "_").replace(" ", "_")

#returned by generate() when the model gave no usable answer parsed as an ERROR
#row so the value is kept in the filtered SDRF instead of being removed
_JUDGE_ERROR_SENTINEL = "JUDGE_ERROR: no usable model response after retries"


#field level constants

#normalised SDRF cells that mean no value
_NOT_EXTRACTED_VALUES = {"", "unknown", "n/a", "na", "none", "null", "not available",
                         "not applicable", "not stated", "not specified", "not reported"}
_SDRF_NA = "not available"
_TOLERANCE_FIELDS = {"precursor_mass_tolerance", "fragment_mass_tolerance"}
#fields where runs can use several values but one SDRF cell names one of them
_ONE_VALUE_PER_CELL_FIELDS = {"fragmentation_method", "cleavage_agent", "mass_analyzer",
                              "ionization_type", "enrichment_method", "fractionation_method"}

#neighbouring fields shown to the judge for consistency and rule B decisions
_RELATED_FIELDS = {
    "species": ["cell_line", "strain", "tissue"],
    "tissue": ["species", "cell_line", "disease_state", "material_type"],
    "cell_type": ["cell_line", "tissue", "developmental_stage"],
    "disease_state": ["cell_line", "tissue", "factor_value"],
    "age": ["developmental_stage"],
    "sex": ["individual"],
    "cell_line": ["species", "cell_type", "material_type"],
    "strain": ["species", "cell_line"],
    "material_type": ["cell_line", "tissue"],
    "developmental_stage": ["age", "cell_type"],
    "ethnicity": [],
    "labeling": ["acquisition_method"],
    "instrument": ["mass_analyzer", "ionization_type", "fragmentation_method"],
    "cleavage_agent": ["modification"],
    "fragmentation_method": ["instrument", "acquisition_method", "collision_energy"],
    "collision_energy": ["fragmentation_method", "instrument"],
    "enrichment_method": ["modification", "fractionation_method"],
    "fractionation_method": ["enrichment_method"],
    "reduction_reagent": ["alkylation_reagent"],
    "alkylation_reagent": ["reduction_reagent", "modification"],
    "ionization_type": ["instrument"],
    "mass_analyzer": ["instrument", "fragmentation_method"],
    "acquisition_method": ["fragmentation_method", "instrument"],
    "technology_type": ["labeling", "enrichment_method"],
    "modification": ["cleavage_agent", "alkylation_reagent", "enrichment_method"],
    "precursor_mass_tolerance": ["fragment_mass_tolerance", "instrument"],
    "fragment_mass_tolerance": ["precursor_mass_tolerance", "instrument", "mass_analyzer"],
}

#field pairs where one fact may legitimately fill both fields (no rule B between them)
_ALLOWED_OVERLAP_PAIRS = {frozenset(p) for p in [
    ("enrichment_method", "modification"),
    ("age", "developmental_stage"),
    ("individual", "factor_value"),
]}

_WS_RE = re.compile(r"\s+")
_INT_RE = re.compile(r"^\d+$")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_NUMBER_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve",
}


#text helpers

def _canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    for ch in ("\u2212", "\u2013", "\u2014", "\u2010", "\u2011"):
        s = s.replace(ch, "-")
    return _WS_RE.sub(" ", s).strip()


def _norm(s: str) -> str:
    """normalise a string for comparison: unicode dashes whitespace lowercase"""
    return _canon(str(s or "")).lower()


def _is_not_extracted(value) -> bool:
    """return True when an SDRF cell holds no value"""
    return _norm(value).strip(" .") in _NOT_EXTRACTED_VALUES


def _number_printed(value: str, paper_text: str) -> bool:
    """true when the number in value appears in the text as digits or as a word"""
    m = _NUM_RE.search(value or "")
    if not m or not paper_text:
        return False
    num = m.group(0)
    #not part of a longer number or an identifier ("3D7" "PXD000070" "C18")
    if re.search(rf"(?<![\w.]){re.escape(num)}(?![\d]|\.\d)", paper_text):
        return True
    w = _NUMBER_WORDS.get(num)
    return bool(w and re.search(rf"\b{w}\b", paper_text, re.IGNORECASE))


#quantities with SI prefixes: "0.1 uM" == "100 nM" "0.5 Da" == "500 mDa"
_SI_PREFIX = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "\u00b5": 1e-6, "\u03bc": 1e-6,
              "m": 1e-3, "k": 1e3}
_SI_UNIT = {"M": "M", "mol": "mol", "g": "g", "l": "L", "L": "L", "Da": "Da"}
_QTY_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(p|n|u|\u00b5|\u03bc|m|k)?(M|mol|g|l|L|Da)(?![A-Za-z])")
_ACCESSION_RE = re.compile(r"^[A-Za-z]+[:_]\d+$")
#run columns where a fix may rename levels but not move runs between levels on its own
_REGROUP_GUARD_FIELDS = {"factor_value", "individual"}


def _run_key_printed(key: str, paper_text: str) -> bool:
    """true when the paper names this run (file name with or without its extension)"""
    if not key or not paper_text:
        return False
    stem = re.sub(r"\.(raw|mzml|mgf|wiff|d)$", "", key.strip(), flags=re.IGNORECASE)
    return bool(stem) and stem.lower() in paper_text.lower()


_MMU_RE = re.compile(r"(\d)\s*mmu\b", re.IGNORECASE)


def _mmu_to_mda(text: str) -> str:
    """'15 mmu' (milli mass units) -> '15 mDa'"""
    return _MMU_RE.sub(r"\1 mDa", text or "")


def _tolerance_in_da(value: str) -> str:
    """'15 mmu' / '20 mDa' -> '0.015 Da' / '0.02 Da'; anything else is returned unchanged"""
    q = _quantity(value)
    if q and q[1] == "Da" and not re.search(r"(?<![a-z])Da\b", value.strip()[-3:]):
        return f"{q[0]:.6g} Da"
    return value


def _quantity(value: str):
    """(base value unit family) when the whole value is one quantity else None"""
    m = _QTY_RE.fullmatch(_mmu_to_mda(value).strip())
    if not m:
        return None
    return float(m.group(1)) * _SI_PREFIX[m.group(2) or ""], _SI_UNIT[m.group(3)]


def _same_quantity(a: str, b: str) -> bool:
    qa, qb = _quantity(a), _quantity(b)
    return bool(qa and qb and qa[1] == qb[1]
                and math.isclose(qa[0], qb[0], rel_tol=1e-6, abs_tol=0.0))


def _quantity_printed(value: str, paper_text: str) -> bool:
    """true when the paper prints the same quantity in any SI prefix of the unit"""
    q = _quantity(value)
    if not q or not paper_text:
        return False
    for m in _QTY_RE.finditer(_mmu_to_mda(paper_text)):
        v = float(m.group(1)) * _SI_PREFIX[m.group(2) or ""]
        if _SI_UNIT[m.group(3)] == q[1] and math.isclose(v, q[0], rel_tol=1e-6):
            return True
    return False


def _searched_mod_sites(name: str, paper_text: str) -> list[str]:
    """sites printed with a searched modification: 'Acetyl (Protein N-term)' -> ['Protein N-term']"""
    if not name or not paper_text:
        return []
    pat = rf"(?<![\w-]){re.escape(name)}\s*\(([^()]{{1,40}})\)"
    return [m.group(1).strip() for m in re.finditer(pat, paper_text, re.IGNORECASE)]


def _mod_from_sites(original: str, sites: list[str]) -> str | None:
    """modification cell rebuilt from the printed sites keeping NT AC and MT of the original"""
    old = _kv_parse(original)
    if not old.get("NT"):
        return None
    residues, position = [], None
    for site in sites:
        low = site.lower()
        if "term" in low:
            if position is None:
                position = ("Protein " if "protein" in low else "Any ") + \
                           ("N-term" if "n-term" in low or "n term" in low else "C-term")
        elif re.fullmatch(r"[A-Za-z ,/]+", site):
            for ch in re.sub(r"[^A-Za-z]", "", site).upper():
                if ch not in residues:
                    residues.append(ch)
        else:
            return None
    if not residues and not position:
        return None
    new = {"NT": old["NT"]}
    if old.get("AC"):
        new["AC"] = old["AC"]
    if residues:
        new["TA"] = ",".join(residues)
    if position:
        new["PP"] = position
    if old.get("MT"):
        new["MT"] = old["MT"]
    return _kv_build(new)


def _kv_parse(cell: str) -> dict[str, str]:
    """'NT=Trypsin;AC=MS:1001251' -> {'NT': 'Trypsin' 'AC': 'MS:1001251'} (ordered)"""
    parts = {}
    for p in (cell or "").split(";"):
        if "=" in p:
            k, v = p.split("=", 1)
            if k.strip():
                parts[k.strip().upper()] = v.strip()
    return parts


def _kv_build(parts: dict[str, str]) -> str:
    #residues as SDRF writes them: "NQ" -> "N,Q" "S T Y" -> "S,T,Y"
    ta = parts.get("TA", "")
    if ta and "," not in ta and re.fullmatch(r"[A-Za-z ]+", ta) and len(ta.replace(" ", "")) > 1:
        parts = {**parts, "TA": ",".join(ta.replace(" ", "").upper())}
    order = ["NT", "AC", "TA", "PP", "MT", "CF", "TS"]
    keys = [k for k in order if k in parts] + [k for k in parts if k not in order]
    return ";".join(f"{k}={parts[k]}" for k in keys if parts[k] != "")


def _is_kv(cell: str) -> bool:
    return bool(re.search(r"(?:^|;)\s*(?:NT|AC)\s*=", cell or ""))


def _readable(cell: str) -> str:
    """plain words for a cell: 'NT=Oxidation;AC=UNIMOD:35;TA=M;MT=Variable' -> 'Oxidation (M, Variable)'"""
    if not _is_kv(cell):
        return cell.strip()
    parts = _kv_parse(cell)
    extra = [parts[k] for k in ("TA", "PP", "MT") if parts.get(k)]
    name = parts.get("NT", cell.strip())
    return f"{name} ({', '.join(extra)})" if extra else name


def _split_top_level(s: str) -> list[str]:
    """split on commas and semicolons that are not inside parentheses"""
    items, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    items.append(cur.strip())
    return [i for i in items if i]


def _ptm_to_sdrf(value: str) -> list[str]:
    """'Carbamidomethyl (C) fixed, Phospho (ST), all variable' -> SDRF modification cells"""
    if _is_kv(value):
        return [value.strip()]
    mods, global_mt = [], None
    for item in _split_top_level(value):
        low = item.lower().strip()
        if re.fullmatch(r"(?:all\s+)?(?:fixed|static|variable)(?:\s+modifications?)?", low):
            global_mt = "Variable" if "variable" in low else "Fixed"
            continue
        mt = None
        if re.search(r"\b(?:fixed|static)\b", low):
            mt = "Fixed"
        elif re.search(r"\bvariable\b", low):
            mt = "Variable"
        name_part = re.sub(r"\b(?:fixed|static|variable)\b", "", item, flags=re.IGNORECASE)
        name_part = name_part.strip(" ,;:")
        m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", name_part)
        name, site = (m.group(1).strip(), m.group(2).strip()) if m else (name_part, "")
        cell = f"NT={name}"
        if site:
            if re.search(r"term", site, re.IGNORECASE):
                cell += f";PP={site}"
            else:
                residues = re.sub(r"[^A-Za-z]", "", site).upper()
                if residues:
                    cell += ";TA=" + ",".join(residues)
        mods.append((cell, mt))
    return [c + (f";MT={mt or global_mt}" if (mt or global_mt) else "") for c, mt in mods]


def _format_cell(original: str, corrected: str, key_value: bool, field: str) -> str:
    """a corrected value in the cell style of its column keeping AC when the term is unchanged"""
    corrected = corrected.strip()
    if field == "modification":
        cells = _ptm_to_sdrf(corrected)
        corrected = cells[0] if cells else corrected
        key_value = True
    if not key_value:
        if _ACCESSION_RE.match(original.strip()) and _is_kv(corrected):
            return _kv_build(_kv_parse(corrected))
        return _kv_parse(corrected).get("NT", corrected) if _is_kv(corrected) else corrected
    new = _kv_parse(corrected) if _is_kv(corrected) else {"NT": corrected}
    old = _kv_parse(original) if _is_kv(original) else {}
    if old.get("AC") and "AC" not in new and _norm(old.get("NT", "")) == _norm(new.get("NT", "")):
        new = {"NT": new["NT"], "AC": old["AC"], **{k: v for k, v in new.items() if k != "NT"}}
    return _kv_build(new)


#thread-safe disk response cache

class DiskResponseCache:
    """on disk JSON cache for LLM judge responses"""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self._dir = Path(cache_dir)
        self._enabled = enabled
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(paper_id: str, task: str, field_name: str, primary_value: str,
                  context_key: str = "") -> str:
        blob = (f"{_PROMPT_FINGERPRINT}|{paper_id}|{task}|{field_name.lower()}|"
                f"{primary_value}|{context_key}")
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, paper_id, task, field_name, primary_value, context_key=""):
        if not self._enabled:
            return None
        key = self._make_key(paper_id, task, field_name, primary_value, context_key)
        path = self._dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self._hits += 1
                return data.get("response")
            except (json.JSONDecodeError, OSError):
                return None
        with self._lock:
            self._misses += 1
        return None

    def put(self, paper_id, task, field_name, primary_value, response, context_key=""):
        if not self._enabled:
            return
        key = self._make_key(paper_id, task, field_name, primary_value, context_key)
        try:
            (self._dir / f"{key}.json").write_text(json.dumps({
                "key": key, "prompt_fingerprint": _PROMPT_FINGERPRINT, "paper_id": paper_id,
                "task": task, "field_name": field_name, "primary_value": primary_value,
                "response": response, "timestamp": time.time(),
            }, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            print(f"  [cache write error] {exc}")

    def stats(self) -> dict:
        n_files = sum(1 for _ in self._dir.glob("*.json")) if self._enabled else 0
        total = self._hits + self._misses
        return {"enabled": self._enabled, "cache_dir": str(self._dir),
                "cached_responses_on_disk": n_files, "session_hits": self._hits,
                "session_misses": self._misses,
                "session_hit_rate": f"{self._hits / total * 100:.1f}%" if total > 0 else "n/a"}


#prompt-cached message builder and OpenRouter client
#
#structure:
#[system]  static rules + definitions        cache_control: ephemeral
#[user]    paper text (same for all values)  cache_control: ephemeral
#[asst]    ack
#[user]    per value task JSON               dynamic

def _build_messages(paper_text: str, entity_context: dict,
                    system_prompt: str | None = None) -> list[dict]:
    """construct the four turn message list sent to the judge model"""
    return [
        {"role": "system",
         "content": [{"type": "text", "text": system_prompt or _STATIC_SYSTEM_PROMPT,
                      "cache_control": {"type": "ephemeral"}}]},
        {"role": "user",
         "content": [{"type": "text",
                      "text": ("PAPER TEXT (abstract and methods). Use it as the only source of "
                               "truth for every SOURCE CHECK and TRUTH CHECK in this "
                               "conversation:\n\n" + paper_text),
                      "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant",
         "content": ("Understood. I have read the paper text. I will judge each value against "
                     "it using the field definitions and rules in my instructions.")},
        {"role": "user", "content": json.dumps(entity_context, indent=2, ensure_ascii=False)},
    ]


def _build_openrouter_client() -> openai.OpenAI:
    """create and return an OpenAI client configured to use the OpenRouter endpoint"""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable is not set.")
    return openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _extract_text_content(raw_message) -> str:
    """extract all plain text from an OpenAI compatible response message object"""
    if raw_message is None:
        return ""
    content = getattr(raw_message, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text") or "")
            elif hasattr(block, "type") and hasattr(block, "text"):
                if block.type == "text":
                    parts.append(block.text or "")
        return " ".join(parts)
    if isinstance(content, str):
        return content
    return ""


#gemma sometimes returns garbled or looping text these responses are retried and
#never cached
_DEGENERATE_PATTERN = re.compile(r"(- a[- ]){10,}")
_DEGENERATE_UNICODE_PATTERN = re.compile(
    r"[\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff\u3000-\u9fff\uac00-\ud7af"
    r"\u1100-\u11ff\uf900-\ufaff]{3,}[- a-z0-9]{0,5}"
    r"[\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff\u3000-\u9fff\uac00-\ud7af"
    r"\u1100-\u11ff\uf900-\ufaff]{3,}"
)


def _is_degenerate_response(text: str, expect_verdict: bool = True) -> bool:
    if not text or len(text) < 100:
        return False
    if _DEGENERATE_PATTERN.search(text):
        return True
    if len(re.findall(r"((?:- a){5,})", text)) >= 3:
        return True
    if _DEGENERATE_UNICODE_PATTERN.search(text):
        tail = text[len(text) // 2:]
        if len(set(tail)) < 30 and len(tail) > 200:
            return True
    if len(text) > 500 and len(set(text[-500:])) / 500 < 0.05:
        return True
    #a usable answer always ends with the VERDICT block
    if expect_verdict and len(text) > 4000 and not re.search(r"VERDICT", text, re.IGNORECASE):
        return True
    return False


#thread-safe LLM wrapper used by GEval

class Gemma4OpenRouter(DeepEvalBaseLLM):
    """llm judge wrapper that routes GEval calls through OpenRouter to Gemma 4"""

    def __init__(self, cache_dir: str):
        self._model_name = EVALUATION_MODEL
        self._client: openai.OpenAI | None = None
        self._paper_text: str = ""
        self._current_paper_id: str = ""
        self._response_cache = DiskResponseCache(cache_dir, enabled=CACHE_ENABLED)
        self._call_count = 0
        self._api_call_count = 0
        self._disk_hit_count = 0
        self._failed_count = 0
        self._lock = threading.Lock()

    def set_paper_context(self, source_text: str, paper_id: str = "") -> None:
        """set per-paper context (called ONCE before the parallel eval loop)"""
        self._paper_text = source_text
        self._current_paper_id = paper_id

    def set_entity_context(self, context: dict) -> None:
        """store per-value context in thread-local storage (thread-safe)"""
        _thread_local.current_context = context

    def load_model(self) -> openai.OpenAI:
        if self._client is None:
            self._client = _build_openrouter_client()
        return self._client

    def _generate_single(self, paper_text: str, paper_id: str, context: dict,
                         system_prompt: str | None = None) -> str:
        """
        thread safe single judge call with disk cache + cache_control system_prompt defaults to the verify prompt; the recovery pass passes its own
        retries up to MAX_RETRIES times on empty failed garbled or verdict less responses returns "" when no usable response was obtained
        """
        with self._lock:
            self._call_count += 1
            call_num = self._call_count

        task = context.get("task", "verify")
        field_name = context.get("sdrf_column") or context.get("field_group") or ""
        primary_value = context.get("candidate_value", "")
        system_prompt = system_prompt or _STATIC_SYSTEM_PROMPT
        expect_verdict = task in ("verify", "verify_runs")
        context_key = hashlib.sha256(
            (system_prompt + "|" + paper_text + "|"
             + json.dumps(context, sort_keys=True, ensure_ascii=False)).encode("utf-8")
        ).hexdigest()[:24]

        #layer 1: disk cache lookup
        cached = self._response_cache.get(paper_id, task, field_name, primary_value, context_key)
        if cached is not None:
            with self._lock:
                self._disk_hit_count += 1
            print(f"  [call {call_num}] DISK CACHE HIT  paper={paper_id!r}  task={task}  "
                  f"column={field_name}  value={primary_value[:40]!r}")
            return cached

        #layer 2: live API call with cache_control + retries
        with self._lock:
            self._api_call_count += 1
        messages = _build_messages(paper_text, context, system_prompt)
        print(f"  [call {call_num}] API CALL  paper={paper_id!r}  task={task}  "
              f"column={field_name}  value={primary_value[:40]!r}  "
              f"[sys={len(system_prompt):,}  paper={len(paper_text):,} chars]")

        extra_body = {"thinking": {"type": "enabled"}} if MODEL_ENABLE_THINKING else {}
        content = ""
        temperature = MODEL_TEMPERATURE
        for attempt in range(1, MAX_RETRIES + 1):
            wait = RETRY_DELAY * attempt
            reason_txt = ""
            try:
                response = self.load_model().chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=MAX_COMPLETION_TOKENS,
                    extra_body=extra_body if extra_body else None,
                    timeout=REQUEST_TIMEOUT,
                )
                if response is None or not getattr(response, "choices", None):
                    content, reason_txt = "", "empty/null response"
                else:
                    content = _extract_text_content(response.choices[0].message).strip()
                    if hasattr(response, "usage") and response.usage:
                        usage = response.usage
                        details = getattr(usage, "prompt_tokens_details", None)
                        cached_tokens = (getattr(details, "cached_tokens", 0) or 0) if details else 0
                        print(f"  [tokens] prompt={usage.prompt_tokens:,}  "
                              f"completion={usage.completion_tokens:,}  "
                              f"cached_prompt={cached_tokens:,}")
                    if not content:
                        reason_txt = "empty text content"
                    elif _is_degenerate_response(content, expect_verdict):
                        content, reason_txt = "", "garbled/repetitive response"
                        temperature = DEGENERATE_RETRY_TEMPERATURE
                    elif expect_verdict and _parse_verdict(content) is None:
                        content, reason_txt = "", "no VERDICT line"
            except Exception as api_exc:
                content, reason_txt = "", f"{type(api_exc).__name__}: {api_exc}"

            if content:
                break
            if attempt < MAX_RETRIES:
                print(f"  [WARNING] {reason_txt} (attempt {attempt}/{MAX_RETRIES}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [WARNING] {reason_txt} (attempt {attempt}/{MAX_RETRIES}), giving up.")

        #only cache usable responses
        if content:
            self._response_cache.put(paper_id, task, field_name, primary_value, content,
                                     context_key)
        else:
            with self._lock:
                self._failed_count += 1
        return content

    def generate(self, prompt: str, schema=None):
        """
        called by GEval internally reads context from thread-local storage
        so concurrent GEval.measure() calls don't interfere
        """
        context = getattr(_thread_local, "current_context", None) or {}
        content = self._generate_single(self._paper_text, self._current_paper_id, context)

        #no usable answer: hand back a sentinel that is parsed as an ERROR row
        if not content:
            content = _JUDGE_ERROR_SENTINEL

        print(f"  [response] {content[:200]}...")

        if schema is not None:
            payload = {"score": 1, "reason": content}
            for attempt in (
                lambda: schema(**payload),
                lambda: schema.model_validate(payload),
                lambda: schema.model_validate_json(json.dumps(payload)),
            ):
                try:
                    return attempt()
                except Exception:
                    continue
            return json.dumps(payload)
        return content

    async def a_generate(self, prompt: str, schema=None):
        """async wrapper that delegates to the synchronous generate method"""
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        suffix = "-thinking" if MODEL_ENABLE_THINKING else ""
        return f"{self._model_name}{suffix}"

    def get_cache_stats(self) -> dict:
        stats = self._response_cache.stats()
        stats["prompt_fingerprint"] = _PROMPT_FINGERPRINT
        stats["total_generate_calls"] = self._call_count
        stats["actual_api_calls"] = self._api_call_count
        stats["disk_cache_hits"] = self._disk_hit_count
        stats["failed_calls"] = self._failed_count
        if self._call_count > 0:
            stats["overall_disk_hit_rate"] = (
                f"{self._disk_hit_count / self._call_count * 100:.1f}%")
        return stats


#singleton model + per-thread GEval metrics

#module level singleton holding the shared judge model instance
_judge_model: Gemma4OpenRouter | None = None
_JUDGE_CACHE_DIR = ""


def _get_judge_model() -> Gemma4OpenRouter:
    """return the shared judge singleton creating it on first call"""
    global _judge_model
    if _judge_model is None:
        _judge_model = Gemma4OpenRouter(_JUDGE_CACHE_DIR or
                                        os.path.join(RESULTS_ROOT,
                                                     f".prompt_cache_{_CACHE_MODEL_SLUG}"))
    return _judge_model


def _get_thread_annotation_metric() -> GEval:
    """
    return a GEval metric instance that is LOCAL to the calling thread
    geval.measure() mutates internal state (self.reason self.score etc)
    so each thread must have its own instance to avoid data races
    all instances share the same underlying LLM model (which IS thread-safe
    thanks to thread-local context storage)
    """
    if not hasattr(_thread_local, "annotation_metric"):
        _thread_local.annotation_metric = GEval(
            name="AnnotationJudge",
            criteria=_ANNOTATION_GEVAL_CRITERIA,
            evaluation_steps=_ANNOTATION_GEVAL_STEPS,
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=_get_judge_model(),
            threshold=0.5,
            async_mode=False,  #synchronous: no nested event loop (the source of RecursionError)
        )
    return _thread_local.annotation_metric


#response parsing

def _clean_for_parse(reason: str) -> str:
    """remove markdown decoration so "**VERDICT:** high" parses like "VERDICT: high" """
    if not reason:
        return ""
    text = reason.replace("\r", "\n").replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"(?m)^[ \t]*(?:[-*+]|#+|>)[ \t]+", "", text)


def _last_field_match(reason: str, label: str, value_pattern: str):
    """last 'LABEL: value' line anchored matches win over labels echoed inside prose"""
    text = _clean_for_parse(reason)
    anchored = list(re.finditer(rf"(?im)^[ \t]*{label}[ \t]*:[ \t]*{value_pattern}", text))
    if anchored:
        return anchored[-1]
    loose = list(re.finditer(rf"(?i)\b{label}[ \t]*:[ \t]*{value_pattern}", text))
    return loose[-1] if loose else None


def _parse_verdict(reason: str) -> str | None:
    m = _last_field_match(reason, "VERDICT", r"(high|medium|low)\b")
    return m.group(1).lower() if m else None


def _parse_yes_no(reason: str, label: str) -> bool | None:
    """parse a named yes or no flag from a structured judge response string"""
    m = _last_field_match(reason, label, r"(yes|no)\b")
    return (m.group(1).lower() == "yes") if m else None


def _parse_correct_type_name(reason: str, field_name: str = "") -> str | None:
    own = field_name.lower()

    def _accept(name: str, strict: bool = False) -> str | None:
        n = name.strip().strip("'\"`,.()[]:")
        canonical = _CANONICAL_FIELD.get(n.lower())
        if not canonical or canonical.lower() == own:
            return None
        if strict and not ("_" in n or any(c.isupper() for c in n)):
            return None
        return canonical

    m = _last_field_match(reason, "CORRECT_TYPE_NAME", r"(\S+)")
    if m and m.group(1).strip().upper() not in ("NONE", "N/A", "NULL", "SAME", "YES", "NO"):
        result = _accept(m.group(1))
        if result:
            return result
    text = _clean_for_parse(reason)
    for pat in (r"correct\s+(?:field|type)\s+is\s+([A-Za-z][A-Za-z_]+)",
                r"belongs\s+(?:in|to)\s+(?:the\s+)?([A-Za-z][A-Za-z_]+)"):
        for m2 in re.finditer(pat, text, re.IGNORECASE):
            result = _accept(m2.group(1), strict=True)
            if result:
                return result
    return None


def _clean_corrected_value(raw: str) -> str | None:
    val = (raw or "").strip().strip("*`").strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        val = val[1:-1].strip()
    if not val or val.upper() in ("NONE", "N/A", "NULL", "SAME", "NA"):
        return None
    if val.startswith("<") and val.endswith(">"):
        return None  #the model copied the template placeholder
    return val


def _parse_corrected_value(reason: str) -> str | None:
    """extract the CORRECTED_VALUE from a judge response with a prose fallback"""
    m = _last_field_match(reason, "CORRECTED_VALUE", r"(.+?)[ \t]*$")
    if m:
        val = _clean_corrected_value(m.group(1))
        if val:
            return val
    text = _clean_for_parse(reason)
    block = re.search(r"COMPLETENESS\s*CHECK\s*:\s*(.+?)(?=TYPE_CORRECT|$)", text,
                      re.IGNORECASE | re.DOTALL)
    if block:
        fm = re.search(r'(?:should\s+be|corrected\s+to|becomes)\s+"([^"]{1,200})"',
                       block.group(1), re.IGNORECASE)
        if fm:
            val = _clean_corrected_value(fm.group(1))
            if val:
                print(f"  [corrected value fallback] Extracted '{val}' from prose")
                return val
    return None


def _parse_run_mapping(text: str, run_keys: list[str]) -> dict[str, str] | None:
    """'k1 = v1 | k2 = v2' -> {k1: v1 k2: v2} when it covers every run key else None"""
    if not text:
        return None
    mapping = {}
    for part in text.split("|"):
        if "=" not in part:
            continue
        k, v = part.rsplit("=", 1)
        mapping[k.strip()] = v.strip().strip("'\"")
    lowered = {k.lower(): v for k, v in mapping.items()}
    out = {}
    for key in run_keys:
        if key in mapping:
            out[key] = mapping[key]
        elif key.lower() in lowered:
            out[key] = lowered[key.lower()]
        else:
            return None
    return out


def _parse_checks_to_dict(issue_summary: str) -> dict:
    """parse the four narrative check lines from an issue summary into a structured dict"""
    keys_out = {"TYPE CHECK": "TYPE_CHECK", "SOURCE CHECK": "SOURCE_CHECK",
                "TRUTH CHECK": "TRUTH_CHECK", "COMPLETENESS CHECK": "COMPLETENESS_CHECK"}
    result = {v: "" for v in keys_out.values()}
    text = _clean_for_parse(issue_summary or "")
    for label, key in keys_out.items():
        m = re.search(rf"(?im)^[ \t]*{label}[ \t]*:(.*?)(?=^[ \t]*[A-Z_ ]{{4,}}:|\Z)",
                      text, re.DOTALL)
        if m:
            result[key] = _WS_RE.sub(" ", m.group(1)).strip()
    return result


def _error_result(message: str) -> dict:
    return {"verdict": None, "type_mismatch": None, "correct_type": None,
            "value_correct": None, "value_complete": None, "hallucination": None,
            "corrected_value": None, "issue_summary": message, "match_type": "ERROR",
            "geval_score": None}


#sdrf reading and judging units

def find_sdrf(pxd_id: str) -> Path | None:
    if not SDRF_DIR:
        p = Path(DATA_ROOT) / SPLIT / pxd_id / f"{pxd_id}.sdrf.tsv"
        return p if p.exists() else None
    base = Path(SDRF_DIR)
    for p in (base / f"{pxd_id}.sdrf.tsv", base / pxd_id / f"{pxd_id}.sdrf.tsv"):
        if p.exists():
            return p
    hits = sorted(base.glob(f"{pxd_id}*.sdrf.tsv"))
    return hits[0] if len(hits) == 1 else None


def read_sdrf(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        rows = [r for r in csv.reader(fh, delimiter="\t") if any(c.strip() for c in r)]
    if not rows:
        return [], []
    header = list(rows[0])
    body = [list(r) + [""] * (len(header) - len(r)) for r in rows[1:]]
    return header, body


def write_sdrf(path: str, header: list[str], body: list[list[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(body)


def _run_keys(header: list[str], body: list[list[str]]) -> tuple[list[str], list[str]]:
    """a unique key per run (data file name else assay name else row number) and a label"""
    keys_h = [_sdrf_header_key(h) for h in header]

    def col(name):
        return next((i for i, k in enumerate(keys_h) if k == name), None)

    i_file, i_assay, i_src = col("datafile"), col("assayname"), col("sourcename")
    keys = None
    for idx in (i_file, i_assay):
        if idx is not None:
            cand = [r[idx].strip() for r in body]
            if all(cand) and len(set(cand)) == len(cand):
                keys = cand
                break
    if keys is None:
        keys = [f"row {i + 1}" for i in range(len(body))]
    labels = []
    for i, r in enumerate(body):
        bits = [keys[i]]
        if i_assay is not None and r[i_assay].strip() and r[i_assay].strip() != keys[i]:
            bits.append(f"assay={r[i_assay].strip()}")
        if i_src is not None and r[i_src].strip():
            bits.append(f"source={r[i_src].strip()}")
        labels.append(", ".join(bits))
    return keys, labels


def _cell_identity(cell: str) -> str:
    """a cell without its accession: cells that differ only in AC are one value"""
    cell = cell.strip()
    if not _is_kv(cell):
        return cell
    return ";".join(p.strip() for p in cell.split(";")
                    if p.strip() and not re.match(r"\s*AC\s*=", p, re.IGNORECASE))


def build_units(header: list[str], body: list[list[str]]) -> tuple[list[dict], list[dict]]:
    """
    judging units of one SDRF:
      value         one per distinct value of a column (rows that carry it)
      modification  one per distinct modification cell across all Modification columns
      runs          one per run structure column (the whole run-to-value mapping)
    returns (units columns)
    """
    run_keys, run_labels = _run_keys(header, body)
    columns = []
    for i, h in enumerate(header):
        kind, field = classify_column(h)
        cells = [r[i] for r in body]
        key_value = any(_is_kv(c) for c in cells if not _is_not_extracted(c))
        columns.append({"index": i, "header": h, "kind": kind, "field": field,
                        "key_value": key_value, "cells": cells})

    units = []
    mod_cols = [c for c in columns if c["kind"] == "modification"]
    if mod_cols:
        #cells that differ only in their accession are judged once
        seen: dict[str, list] = {}
        variants: dict[str, list[str]] = {}
        for c in mod_cols:
            for ri, cell in enumerate(c["cells"]):
                if _is_not_extracted(cell):
                    continue
                ident = _cell_identity(cell)
                seen.setdefault(ident, []).append((c["index"], ri))
                if cell.strip() not in variants.setdefault(ident, []):
                    variants[ident].append(cell.strip())
        all_mods = [_readable(variants[k][0]) for k in seen]
        for ident, places in seen.items():
            cell = variants[ident][0]
            rows = sorted({ri for _, ri in places})
            units.append({"kind": "modification", "field": "modification",
                          "header": mod_cols[0]["header"],
                          "columns": sorted({ci for ci, _ in places}), "cell": cell,
                          "readable": _readable(cell), "rows": rows,
                          "siblings": [m for m in all_mods if m != _readable(cell)],
                          "key_value": True, "variants": variants[ident]})

    seen_value_units: dict[tuple, dict] = {}
    for c in columns:
        if c["kind"] == "value":
            #cells that differ only in their accession are judged once
            distinct, variants = [], {}
            for cell in c["cells"]:
                if _is_not_extracted(cell):
                    continue
                ident = _cell_identity(cell)
                if ident not in variants:
                    variants[ident] = []
                    distinct.append(cell.strip())
                if cell.strip() not in variants[ident]:
                    variants[ident].append(cell.strip())
            for cell in distinct:
                ident = _cell_identity(cell)
                rows = [ri for ri, x in enumerate(c["cells"])
                        if not _is_not_extracted(x) and _cell_identity(x) == ident]
                #a duplicate column with the same header and content is judged once
                dup_key = (c["header"].strip().lower(), cell, tuple(rows))
                if dup_key in seen_value_units:
                    seen_value_units[dup_key]["columns"].append(c["index"])
                    continue
                unit = {"kind": "value", "field": c["field"], "header": c["header"],
                        "columns": [c["index"]], "cell": cell, "readable": _readable(cell),
                        "rows": rows,
                        "siblings": [_readable(d) for d in distinct if d != cell],
                        "key_value": c["key_value"], "variants": variants[ident]}
                seen_value_units[dup_key] = unit
                units.append(unit)
        elif c["kind"] == "runs":
            if all(_is_not_extracted(x) for x in c["cells"]):
                continue
            mapping = " | ".join(f"{k} = {_readable(v)}" for k, v in zip(run_keys, c["cells"]))
            units.append({"kind": "runs", "field": c["field"], "header": c["header"],
                          "columns": [c["index"]], "cell": mapping, "readable": mapping,
                          "rows": list(range(len(body))), "siblings": [],
                          "key_value": c["key_value"], "run_keys": run_keys,
                          "run_labels": run_labels})
    order = {f: i for i, f in enumerate(_FIELD_ORDER + [_GENERIC_FIELD])}
    units.sort(key=lambda u: (order.get(u["field"], 999), u["columns"][0]))
    return units, columns


def _column_values(columns: list[dict]) -> dict[str, list[str]]:
    """{field: distinct readable values} over all columns of that field"""
    out = defaultdict(list)
    for c in columns:
        if not c["field"]:
            continue
        for cell in c["cells"]:
            if not _is_not_extracted(cell):
                r = _readable(cell)
                if r not in out[c["field"]]:
                    out[c["field"]].append(r)
    return out


def unit_checks(unit: dict, paper_canon_lower: str) -> list[str]:
    """deterministic hints shown to the judge"""
    checks = []
    field, readable = unit["field"], unit["readable"]
    if unit["kind"] != "runs":
        name = _kv_parse(unit["cell"]).get("NT", readable) if _is_kv(unit["cell"]) else readable
        if _norm(name) and _norm(name) in paper_canon_lower:
            checks.append("text: the term appears in the paper text")
        else:
            checks.append("text: the term does not appear verbatim in the paper text "
                          "(it may still be a correct normalized term or an allowed inference)")
    if field in _TOLERANCE_FIELDS:
        if not re.search(r"\d", readable) or not re.search(r"ppm|da|mmu|th", readable, re.I):
            checks.append("format: expected a number with a unit (ppm or Da)")
        elif not all(_number_printed(n, paper_canon_lower) for n in _NUM_RE.findall(readable)):
            checks.append("numbers: the tolerance number is not printed in the paper")
    if field == "disease_state" and _norm(readable) not in ("normal",) and \
            re.search(r"\b(healthy|normal|control)\b", readable, re.I):
        checks.append("format: the SDRF term for healthy subjects is 'normal'")
    if unit["kind"] == "modification":
        parts = _kv_parse(unit["cell"])
        if not parts.get("MT"):
            checks.append("format: no MT=Fixed or MT=Variable")
        if not parts.get("TA") and not parts.get("PP"):
            checks.append("format: no TA residues or PP position")
    if unit["kind"] == "runs" and field in ("biological_replicate", "technical_replicate",
                                            "fraction_identifier"):
        vals = [v.split("=", 1)[1].strip() for v in unit["cell"].split("|") if "=" in v]
        if vals and not all(_INT_RE.match(v) for v in vals):
            checks.append("format: every run should carry a plain number")
    return checks


def _unit_context(unit: dict, related: dict, n_runs: int, checks: list[str]) -> dict:
    if unit["kind"] == "runs":
        return {
            "task": "verify_runs",
            "field_name": unit["field"],
            "sdrf_column": unit["header"],
            "field_definition": _field_def_for(unit["field"], unit["header"]),
            "runs": [f"{lab} : {unit['header']} = {val.split('=', 1)[1].strip()}"
                     for lab, val in zip(unit["run_labels"], unit["cell"].split(" | "))],
            "candidate_value": unit["cell"],
            "checks": checks,
        }
    examples = [f"row {ri + 1}" for ri in unit["rows"][:MAX_EXAMPLE_RUNS]]
    ctx = {
        "task": "verify",
        "field_name": unit["field"],
        "sdrf_column": unit["header"],
        "field_definition": _field_def_for(unit["field"], unit["header"]),
        "candidate_value": unit["cell"],
        "readable_value": unit["readable"],
        "cell_style": "key-value" if unit["key_value"] else "plain text",
        "other_values_in_column": unit["siblings"],
        "runs_with_value": f"{len(unit['rows'])} of {n_runs} runs",
        "example_runs": unit.get("example_labels") or examples,
        "checks": checks,
    }
    if related:
        ctx["related_columns"] = related
    return ctx


#per-value GEval evaluation (runs inside a worker thread)

def interpret_reason(field_name: str, extracted_value: str, reason: str, paper_text: str,
                     kind: str = "value", run_keys: list[str] | None = None) -> dict:
    """turn one judge answer into flags with the safety rules applied after parsing"""
    notes: list[str] = []

    def _note(msg: str) -> None:
        print(msg)
        notes.append(msg.strip())

    verdict = _parse_verdict(reason)
    is_hallucinated = bool(_parse_yes_no(reason, "HALLUCINATED"))
    type_correct = _parse_yes_no(reason, "TYPE_CORRECT")
    is_mismatch = (type_correct is False)
    correct_type = _parse_correct_type_name(reason, field_name) if is_mismatch else None

    #one fact may fill both fields of an allowed pair: never a type mismatch
    if (is_mismatch and correct_type
            and frozenset((field_name, correct_type)) in _ALLOWED_OVERLAP_PAIRS):
        _note(f"  [mismatch-veto] {field_name} vs {correct_type}: allowed overlap, "
              f"judged as wrong value")
        is_mismatch, correct_type = False, None

    #a hallucinated value or a value of another field can never be kept or fixed
    if verdict in ("high", "medium") and (is_hallucinated or is_mismatch):
        _note(f"  [verdict-reconcile] {field_name}: '{extracted_value[:40]}' "
              f"verdict={verdict} but hallucinated={is_hallucinated} "
              f"mismatch={is_mismatch} -- setting verdict=low")
        verdict = "low"

    vc = _parse_yes_no(reason, "VALUE_CORRECT")
    vcomp = _parse_yes_no(reason, "VALUE_COMPLETE")
    value_correct = vc if vc is not None else (verdict in ("high", "medium"))
    value_complete = vcomp if vcomp is not None else (verdict == "high")
    if is_hallucinated or is_mismatch:
        value_correct = False
    if is_hallucinated:
        value_complete = False

    #parse corrected value for MEDIUM verdicts; a "fix" equal to the value is no fix
    corrected = _parse_corrected_value(reason) if verdict == "medium" else None
    if corrected is not None and _norm(corrected) == _norm(extracted_value):
        corrected = None
    #one value per cell: a "fix" that only turns the value into a list holding it is no fix
    if (verdict == "medium" and corrected and kind == "value"
            and field_name in _ONE_VALUE_PER_CELL_FIELDS):
        original = _kv_parse(extracted_value).get("NT", extracted_value) \
            if _is_kv(extracted_value) else extracted_value
        items = [_norm(_kv_parse(x).get("NT", x) if _is_kv(x) else x)
                 for x in re.split(r"[,;/]|\band\b", corrected) if x.strip()]
        if len(items) > 1 and _norm(original) in items:
            _note(f"  [one-value-guard] {field_name}: '{original}' is one of the values the runs "
                  f"used -- setting verdict=high")
            verdict, corrected, value_correct, value_complete = "high", None, True, True
    if kind == "runs" and corrected is not None:
        mapping = _parse_run_mapping(corrected, run_keys or [])
        if mapping is None:
            _note(f"  [runs-guard] {field_name}: corrected mapping does not cover every run "
                  f"-- kept as medium without a fix")
            corrected = None
        else:
            #the same quantity in another unit is no change ("0.1 uM" == "100 nM")
            original_map = _parse_run_mapping(extracted_value, run_keys or []) or {}
            for k in run_keys:
                if k in original_map and _same_quantity(original_map[k], mapping[k]):
                    mapping[k] = original_map[k]
            if original_map and all(_norm(mapping[k]) == _norm(original_map.get(k, ""))
                                    for k in run_keys):
                _note(f"  [unit-guard] {field_name}: corrected mapping only changes units "
                      f"-- setting verdict=high")
                verdict, corrected, value_correct, value_complete = "high", None, True, True
            else:
                corrected = " | ".join(f"{k} = {mapping[k]}" for k in run_keys)

    #factor value / individual: moving runs between levels needs the paper to name those runs
    if (kind == "runs" and corrected and field_name in _REGROUP_GUARD_FIELDS):
        original_map = _parse_run_mapping(extracted_value, run_keys or []) or {}
        new_map = _parse_run_mapping(corrected, run_keys or []) or {}
        if original_map and new_map:
            targets = defaultdict(set)
            for k in run_keys:
                targets[_norm(original_map.get(k, ""))].add(_norm(new_map.get(k, "")))
            moved = [k for k in run_keys
                     if _norm(new_map.get(k, "")) != _norm(original_map.get(k, ""))]
            splits_group = any(len(v) > 1 for v in targets.values())
            named = all(_run_key_printed(k, paper_text) for k in moved)
            if splits_group and not named:
                _note(f"  [regroup-guard] {field_name}: the fix moves {len(moved)} runs to other "
                      f"levels but the paper does not name those runs -- kept as medium "
                      f"without a fix")
                corrected = None

    if kind == "value" and not is_mismatch:
        #a value the paper prints in another unit is right ("0.1 uM" for "100 nM")
        if (verdict == "low" and is_hallucinated
                and _quantity_printed(extracted_value, paper_text)):
            _note(f"  [unit-guard] {field_name}: '{extracted_value}' is printed in the paper "
                  f"in another unit -- setting verdict=high")
            verdict, corrected, is_hallucinated = "high", None, False
            value_correct, value_complete = True, True
        elif verdict == "medium" and corrected and _same_quantity(extracted_value, corrected):
            _note(f"  [unit-guard] {field_name}: '{extracted_value}' -> '{corrected}' only "
                  f"changes the unit -- setting verdict=high")
            verdict, corrected, value_correct, value_complete = "high", None, True, True
        #a bare ontology accession keeps its accession next to the corrected name
        if (verdict == "medium" and corrected and _ACCESSION_RE.match(extracted_value.strip())
                and not _is_kv(corrected) and not _ACCESSION_RE.match(corrected.strip())):
            corrected = _kv_build({"NT": corrected.strip(), "AC": extracted_value.strip()})

    #a searched modification with wrong residues or position is a wrong detail not a
    #hallucination: 'Acetyl TA=K' when the paper searched 'Acetyl (Protein N-term)'
    if (kind == "modification" and verdict == "low" and is_hallucinated and not is_mismatch
            and _is_kv(extracted_value)):
        name = _kv_parse(extracted_value).get("NT", "")
        sites = _searched_mod_sites(name, paper_text)
        fixed = _mod_from_sites(extracted_value, sites) if sites else None
        if fixed:
            is_hallucinated, value_correct, value_complete = False, False, True
            if _norm(fixed) == _norm(extracted_value):
                verdict, corrected, value_correct = "high", None, True
            else:
                verdict, corrected = "medium", fixed
            _note(f"  [mod-guard] {name}: searched in the paper as {sites} -- "
                  f"setting verdict={verdict}" + (f", fix '{fixed}'" if corrected else ""))
    #a corrected tolerance is written in Da or ppm ("15 mmu" -> "0.015 Da")
    if verdict == "medium" and corrected and field_name in _TOLERANCE_FIELDS:
        corrected = _tolerance_in_da(corrected)
        if _same_quantity(extracted_value, corrected):
            verdict, corrected, value_correct, value_complete = "high", None, True, True
    #a corrected tolerance must use numbers printed in the paper (in any unit)
    if verdict == "medium" and corrected and field_name in _TOLERANCE_FIELDS:
        if not (all(_number_printed(n, paper_text) for n in _NUM_RE.findall(corrected))
                or _quantity_printed(corrected, paper_text)):
            _note(f"  [number-guard] {field_name}: corrected '{corrected}' is not printed "
                  f"in the paper -- setting verdict=low")
            verdict, corrected = "low", None

    #derive a human readable match type label from the parsed flags
    if verdict is None:
        match_type = "ERROR"
    elif is_hallucinated:
        match_type = "HALLUCINATED"
    elif is_mismatch:
        match_type = "TYPE_MISMATCH"
    elif verdict == "high":
        match_type = "CORRECT"
    elif verdict == "medium":
        match_type = "PARTIAL"
    else:
        match_type = "NO_MATCH"

    return {"verdict": verdict, "type_mismatch": is_mismatch, "correct_type": correct_type,
            "value_correct": value_correct, "value_complete": value_complete,
            "hallucination": is_hallucinated, "corrected_value": corrected,
            "issue_summary": reason + "".join(f"\nGUARD: {n}" for n in notes),
            "match_type": match_type}


_BANNER_LOCK = threading.Lock()


def _print_deepeval_banner(metric) -> None:
    """print DeepEval's metric description line without the rich live display"""
    try:
        from deepeval.metrics.indicator import format_metric_description
        text = format_metric_description(metric, async_mode=False)
        text = re.sub(r"\[/?rgb\([^\]]*\)\]", "", text)
    except Exception:
        text = (f"You're running DeepEval's latest {getattr(metric, '__name__', 'GEval')} "
                f"Metric! (using {getattr(metric, 'evaluation_model', EVALUATION_MODEL)}, "
                f"async_mode=False)...")
    with _BANNER_LOCK:
        print(f" {text}", flush=True)


def _run_geval_semantic(context: dict, kind: str = "value",
                        run_keys: list[str] | None = None) -> dict:
    """
    judge one SDRF value using the GEval annotation metric
    uses a per-thread GEval metric instance for thread safety
    falls back to a direct API call if GEval raises
    """
    model = _get_judge_model()
    field_name, extracted_value = context["field_name"], context["candidate_value"]

    #store context in thread-local (read back by model.generate)
    model.set_entity_context(context)

    test_case = LLMTestCase(
        input=f"Assess field='{field_name}' column='{context.get('sdrf_column')}' "
              f"candidate_value='{extracted_value}'",
        actual_output=f"field_name: {field_name}\ncandidate_value: {extracted_value}",
        expected_output=_EXPECTED_OUTPUT_TEXT,
    )

    #try GEval first fall back to a direct API call on errors
    reason = ""
    score = None
    try:
        metric = _get_thread_annotation_metric()
        #deepeval's own "You're running DeepEval's latest AnnotationJudge" line printed as
        #plain text the animated rich spinner stays off: from several worker threads it wraps
        #sys.stdout inside itself (RecursionError "_FileProxy__buffer" segmentation fault)
        #scoring is the same either way
        if SHOW_DEEPEVAL_BANNER:
            _print_deepeval_banner(metric)
        try:
            metric.measure(test_case, _show_indicator=False)
        except TypeError:
            metric.measure(test_case)
        reason = metric.reason or ""
        score = metric.score
    except RecursionError:
        print(f"  [GEval recursion] {field_name}: '{extracted_value[:40]}' "
              f"-- falling back to direct API call")
        reason = model._generate_single(model._paper_text, model._current_paper_id,
                                        context) or _JUDGE_ERROR_SENTINEL
    except Exception as exc:
        print(f"  [GEval error] {field_name}: {type(exc).__name__}: {exc}")
        try:
            reason = model._generate_single(model._paper_text, model._current_paper_id,
                                            context)
            if not reason:
                raise ValueError("Empty response from direct fallback")
            print(f"  [GEval->direct fallback] Got response for {field_name}: "
                  f"'{extracted_value[:40]}'")
        except Exception as fallback_exc:
            print(f"  [direct fallback also failed] {fallback_exc}")
            return _error_result(f"GEval error: {exc}")

    if not reason.strip() or reason.strip().startswith(_JUDGE_ERROR_SENTINEL):
        return _error_result(_JUDGE_ERROR_SENTINEL)

    result = interpret_reason(field_name, extracted_value, reason, model._paper_text, kind,
                              run_keys)
    if result["verdict"] is None:
        print(f"  [parse] no VERDICT found for {field_name}: '{extracted_value[:40]}' -- kept")
    result["geval_score"] = score
    return result


#recovery pass: values the paper states but the SDRF does not hold

def _parse_json_object(text: str):
    """first JSON object in a model answer tolerant of code fences and prose around it"""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


def _evidence_status(quote: str, paper_raw: str, paper_canon_lower: str) -> str:
    q = (quote or "").strip().strip('"').strip()
    if not q:
        return "absent"
    if q in paper_raw:
        return "verbatim"
    if _canon(q).lower() in paper_canon_lower:
        return "normalized"
    return "absent"


def recover_missed_values(paper_id: str, accepted: dict, rejected: dict, moved: dict,
                          paper_canon_lower: str) -> tuple[dict, list[dict]]:
    """
    one recovery call per field group (sample technical) a field is open when the SDRF
    holds no accepted value for it modifications are always searched for missing ones
    a candidate is kept only when its quote really is in the paper and it then passes the
    normal GEval verify step
    returns ({field: [(value evidence) ]} accepted additions rows for the CSV)
    """
    model = _get_judge_model()
    paper_text = model._paper_text
    groups = []
    for label, items in _FIELD_GROUPS[:2]:
        open_fields, filled = {}, {}
        for name, _ in items:
            if name == "modification":
                continue
            if accepted.get(name):
                filled[name] = accepted[name]
                continue
            entry = {"definition": _FIELD_DEFINITIONS[name]}
            if rejected.get(name):
                entry["rejected_values"] = rejected[name]
            if moved.get(name):
                entry["moved_here_by_review"] = moved[name]
            open_fields[name] = entry
        ctx = {"task": "recover", "field_group": label, "open_fields": open_fields,
               "filled_fields": filled}
        if any(n == "modification" for n, _ in items):
            ctx["modification"] = {"definition": _FIELD_DEFINITIONS["modification"],
                                   "accepted_values": accepted.get("modification", []),
                                   "rejected_values": rejected.get("modification", [])}
        if open_fields or "modification" in ctx:
            groups.append((label, ctx))
    if not groups:
        return {}, []

    print(f"  Recovery pass: {len(groups)} field groups with {MAX_WORKERS} workers...")
    candidates: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(model._generate_single, paper_text, paper_id, ctx,
                                   _RECOVER_SYSTEM_PROMPT): label for label, ctx in groups}
        for future in as_completed(futures):
            label = futures[future]
            try:
                data = _parse_json_object(future.result())
            except Exception as exc:
                print(f"  [recover] {label}: {type(exc).__name__}: {exc}")
                continue
            if data is None:
                print(f"  [recover] {label}: answer was not valid JSON, skipped")
                continue
            candidates.extend(c for c in (data.get("missed") or []) if isinstance(c, dict))

    per_field = Counter(_CANONICAL_FIELD.get(str(c.get("field") or "").strip().lower(), "")
                        for c in candidates)
    rows, to_verify, seen = [], [], set()
    for c in candidates:
        raw_field = str(c.get("field") or "").strip()
        field = _CANONICAL_FIELD.get(raw_field.lower(), "")
        value = str(c.get("value") or "").strip()
        evidence = str(c.get("evidence") or "").strip()
        row = {"paper_id": paper_id, "annotation_type": field or raw_field,
               "sdrf_column": "", "recovered_value": value, "evidence": evidence,
               "evidence_check": "", "verdict": "", "final_value": "", "accepted": False,
               "reason": "", "issue_summary": ""}
        if field == "modification":
            value = _format_cell("", value, True, "modification")
            row["recovered_value"] = value
        if not field or field not in _RECOVERABLE_FIELDS:
            row["reason"] = "unknown or not recoverable field"
        elif field != "modification" and accepted.get(field):
            row["reason"] = "field is not open"
        elif field != "modification" and per_field[field] > 1:
            row["reason"] = "several candidates for one field"
        elif _is_not_extracted(value):
            row["reason"] = "empty value"
        elif _norm(_readable(value)) in {_norm(v) for v in rejected.get(field, [])}:
            row["reason"] = "rejected by review"
        elif field == "modification" and _norm(_readable(value)) in {
                _norm(v) for v in accepted.get("modification", [])}:
            row["reason"] = "already in the SDRF"
        elif (field, _norm(value)) in seen:
            row["reason"] = "duplicate candidate"
        elif len(evidence) < 8 or len(evidence.split()) > 60:
            row["reason"] = "evidence too short or too long"
        else:
            status = _evidence_status(evidence, paper_text, paper_canon_lower)
            row["evidence_check"] = status
            if status == "absent":
                row["reason"] = "evidence not found in paper text"
            else:
                seen.add((field, _norm(value)))
                to_verify.append((field, value, evidence, row))
        rows.append(row)

    added: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if to_verify:
        print(f"  Verifying {len(to_verify)} recovered values...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for field, value, evidence, row in to_verify:
                unit = {"kind": "value", "field": field, "cell": value,
                        "readable": _readable(value), "key_value": _is_kv(value),
                        "header": _SDRF_STANDARD_HEADER.get(field, field),
                        "rows": [], "siblings": accepted.get(field, [])}
                ctx = _unit_context(unit, {r: accepted.get(r, []) for r in
                                           _RELATED_FIELDS.get(field, [])}, 0,
                                    unit_checks(unit, paper_canon_lower))
                ctx["runs_with_value"] = "proposed for every run"
                ctx["example_runs"] = []
                ctx["recovered_evidence"] = evidence
                futures[executor.submit(_run_geval_semantic, ctx)] = (field, value, evidence, row)
            for future in as_completed(futures):
                field, value, evidence, row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    row["reason"] = f"verify error: {exc}"
                    continue
                row["verdict"] = result.get("verdict") or "ERROR"
                row["issue_summary"] = result.get("issue_summary", "")
                final = None
                if result.get("verdict") == "high":
                    final = value
                elif result.get("verdict") == "medium" and result.get("corrected_value"):
                    final = result["corrected_value"]
                if final:
                    row.update(final_value=final, accepted=True, reason="added")
                    added[field].append((final, evidence))
                    print(f"  [recover:add] {field}: '{final[:60]}'")
                else:
                    row["reason"] = f"failed verify ({row['verdict']})"

    print(f"  [recover] candidates={len(candidates)}  verified={len(to_verify)}  "
          f"added={sum(len(v) for v in added.values())}")
    return dict(added), rows


#applying verdicts to the SDRF

#fields that describe a sample or subject; values may differ per sample
_SAMPLE_LEVEL_FIELDS = {"age", "sex", "ethnicity", "individual", "cell_line", "tissue",
                        "disease_state", "cell_type", "developmental_stage", "strain",
                        "material_type"}
_UNVERIFIABLE = "unverifiable"
#donor attributes of a cell line (looked up in cell line databases not in the paper)
_DONOR_FIELDS = {"sex", "ethnicity", "age", "developmental_stage"}
#fields with few possible values: two different values already mean per-sample data
_MIN_DISTINCT = {"sex": 2}


#per-sample numbers the paper usually reports only as a cohort range ("range 22-48")
_RANGE_KEYWORDS = {
    "age": [r"\bage[sd]?\b", r"\baged\b", r"\byears? old\b", r"\byear-old\b"],
    "bmi": [r"\bbmi\b", r"\bbody mass index\b"],
}
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\u2013|\u2014|\u2212|-|to|and)\s*(\d+(?:\.\d+)?)")
_AGE_VALUE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:y|yr|yrs|year|years)?\s*$", re.IGNORECASE)
_NUM_VALUE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


def _range_kind(u: dict) -> str | None:
    if u["field"] == "age":
        return "age"
    h = _norm(u["header"])
    if "bmi" in h or "body mass index" in h:
        return "bmi"
    return None


def _printed_ranges(kind: str, paper_text: str) -> list[tuple[float, float]]:
    """ranges printed within 120 characters after a keyword: 'age ... (range 22-48)'"""
    text = _canon(paper_text)
    out = []
    for pat in _RANGE_KEYWORDS[kind]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            for r in _RANGE_RE.finditer(text[m.end():m.end() + 120]):
                lo, hi = float(r.group(1)), float(r.group(2))
                if lo < hi and (kind != "age" or hi <= 125) and (lo, hi) not in out:
                    out.append((lo, hi))
    return out


def _sample_sheet_guard(units: list[dict], results: list[dict], columns: list[dict],
                        paper_text: str = "") -> int:
    """mark per-sample values the text cannot confirm as unverifiable returns the count"""
    if not SAMPLE_SHEET_UNVERIFIABLE:
        return 0
    n_distinct = {}
    for c in columns:
        n_distinct[c["index"]] = len({_norm(x) for x in c["cells"] if not _is_not_extracted(x)})
    #rows that name a cell line: donor sex ancestry and age come from cell line
    #databases (Cellosaurus) not from the paper
    cell_line_rows = set()
    for c in columns:
        if c["field"] == "cell_line":
            cell_line_rows |= {ri for ri, x in enumerate(c["cells"]) if not _is_not_extracted(x)}
    n = 0
    for u, r in zip(units, results):
        if u["kind"] != "value" or r.get("type_mismatch") or r.get("verdict") is None:
            continue
        #age / BMI: one rule for every value of a per-sample column -- inside the range the
        #paper prints is unverifiable (kept) outside it contradicts the paper (low)
        kind = _range_kind(u)
        distinct = max((n_distinct.get(ci, 0) for ci in u["columns"]), default=0)
        if kind and distinct >= UNVERIFIABLE_MIN_DISTINCT:
            m = (_AGE_VALUE_RE if kind == "age" else _NUM_VALUE_RE).match(u["cell"])
            ranges = _printed_ranges(kind, paper_text) if m else []
            if m and ranges:
                v = float(m.group(1))
                inside = [f"{lo:g}-{hi:g}" for lo, hi in ranges if lo <= v <= hi]
                if inside:
                    note = (f"[range-guard] {u['header']}: '{u['cell']}' is inside the range the "
                            f"paper prints ({', '.join(inside)}); the per-sample value is not "
                            f"printed -- verdict=unverifiable, kept")
                    r.update(verdict=_UNVERIFIABLE, match_type="NOT_IN_TEXT",
                             hallucination=False, value_correct=None, value_complete=None,
                             corrected_value=None)
                else:
                    shown = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in ranges)
                    note = (f"[range-guard] {u['header']}: '{u['cell']}' is outside the range "
                            f"the paper prints ({shown}) -- verdict=low")
                    r.update(verdict="low", match_type="NO_MATCH", hallucination=False,
                             value_correct=False, value_complete=False, corrected_value=None)
                print(f"  {note}")
                r["issue_summary"] = (r.get("issue_summary") or "") + f"\nGUARD: {note}"
                n += 1
                continue
        if r.get("verdict") != "low" or not r.get("hallucination"):
            continue
        if u["field"] not in _SAMPLE_LEVEL_FIELDS and u["field"] != _GENERIC_FIELD:
            continue
        distinct = max((n_distinct.get(ci, 0) for ci in u["columns"]), default=0)
        if (u["field"] in _DONOR_FIELDS and u["rows"]
                and all(ri in cell_line_rows for ri in u["rows"])):
            note = (f"[cell-line-guard] {u['header']}: '{u['readable'][:40]}' describes the "
                    f"donor of the cell line in these runs, which papers rarely state -- "
                    f"verdict=unverifiable, kept")
        elif distinct >= _MIN_DISTINCT.get(u["field"], UNVERIFIABLE_MIN_DISTINCT):
            note = (f"[sample-sheet-guard] {u['header']}: '{u['readable'][:40]}' is one of "
                    f"{distinct} per-sample values the paper text does not list -- "
                    f"verdict=unverifiable, kept")
        else:
            continue
        print(f"  {note}")
        r.update(verdict=_UNVERIFIABLE, match_type="NOT_IN_TEXT", hallucination=False,
                 value_correct=None, value_complete=None, corrected_value=None,
                 issue_summary=(r.get("issue_summary") or "") + f"\nGUARD: {note}")
        n += 1
    return n


_AC_NUM_RE = re.compile(r"^(.*?)(\d+)$")


def _fix_accession_drift(header: list[str], body: list[list[str]]) -> int:
    """reset accessions that count up row by row to the first accession of the series"""
    if not FIX_ACCESSION_DRIFT or not body:
        return 0
    fixed = 0
    for ci in range(len(header)):
        cells = [r[ci] for r in body]
        groups: dict[tuple, list[int]] = defaultdict(list)
        for ri, cell in enumerate(cells):
            if _is_kv(cell) and _kv_parse(cell).get("AC"):
                parts = _kv_parse(cell)
                groups[tuple(sorted((k, _norm(v)) for k, v in parts.items() if k != "AC"))]\
                    .append(ri)
        for rows in groups.values():
            acs = [_kv_parse(cells[ri])["AC"] for ri in rows]
            parsed = [_AC_NUM_RE.match(a) for a in acs]
            if len(set(acs)) < 3 or not all(parsed):
                continue
            prefixes = {m.group(1) for m in parsed}
            nums = [int(m.group(2)) for m in parsed]
            distinct = sorted(set(nums))
            #a drag-fill: one prefix and consecutive numbers
            if len(prefixes) != 1 or distinct[-1] - distinct[0] + 1 != len(distinct):
                continue
            first = f"{prefixes.pop()}{distinct[0]}"
            for ri in rows:
                if _kv_parse(body[ri][ci]).get("AC") != first:
                    #replace only the accession; the cell keeps its own key order
                    body[ri][ci] = re.sub(r"(AC\s*=\s*)[^;]+", lambda m: m.group(1) + first,
                                          body[ri][ci], count=1)
                    fixed += 1
    if fixed:
        print(f"  [accession-drift] reset {fixed} cell(s) whose accession counted up row by row")
    return fixed


def _mod_key(cell: str) -> tuple:
    """identity of a modification cell without its accession and letter case"""
    if not _is_kv(cell):
        return ("", _norm(cell))
    p = _kv_parse(cell)
    ta = ",".join(sorted(x.strip().upper() for x in re.split(r"[,\s]+", p.get("TA", "")) if x.strip()))
    return (_norm(p.get("NT", "")), ta, _norm(p.get("PP", "")), _norm(p.get("MT", "")))


def apply_to_sdrf(header: list[str], body: list[list[str]], units: list[dict],
                  results: list[dict], recovered: dict) -> tuple[list[str], list[list[str]], dict, dict]:
    """
    filtered SDRF in the input layout returns (header body stats actions per unit index)
    low -> "not available" (value and modification cells) a run column with a low verdict
    is kept unchanged because SDRF needs it medium with a corrected value -> replaced
    recovered values fill a column that holds no value; recovered modifications go into
    modification columns
    """
    header = list(header)
    body = [list(r) for r in body]
    stats = {"total": len(units), "kept": 0, "removed": 0, "corrected": 0, "added": 0,
             "unchanged_low_run_columns": 0, "kept_unverifiable": 0, "accession_drift_fixed": 0}
    actions = {}
    mod_cols = [i for i, h in enumerate(header) if classify_column(h)[0] == "modification"]
    for ui, (u, r) in enumerate(zip(units, results)):
        verdict, corrected = r.get("verdict"), r.get("corrected_value")
        action = "kept"
        if u["kind"] == "runs":
            if verdict == "medium" and corrected:
                mapping = _parse_run_mapping(corrected, u["run_keys"]) or {}
                ci = u["columns"][0]
                for ri, key in enumerate(u["run_keys"]):
                    if key in mapping:
                        body[ri][ci] = _format_cell(body[ri][ci], mapping[key], u["key_value"],
                                                    u["field"])
                action = "corrected"
            elif verdict == "low":
                action = "kept_low_run_column"
                stats["unchanged_low_run_columns"] += 1
        else:
            places = [(ci, ri) for ci in u["columns"] for ri in u["rows"]
                      if body[ri][ci].strip() in u.get("variants", [u["cell"]])]
            if verdict == "low":
                for ci, ri in places:
                    body[ri][ci] = _SDRF_NA
                action = "removed"
            elif verdict == "medium" and corrected:
                n_dup = 0
                for ci, ri in places:
                    new = _format_cell(u["cell"], corrected, u["key_value"], u["field"])
                    #a corrected modification that another column of the row already holds
                    #is a duplicate: the wrong copy is removed instead
                    if u["kind"] == "modification" and any(
                            j != ci and _mod_key(body[ri][j]) == _mod_key(new)
                            for j in mod_cols):
                        body[ri][ci] = _SDRF_NA
                        n_dup += 1
                    else:
                        body[ri][ci] = new
                action = "removed_duplicate" if places and n_dup == len(places) else "corrected"
        if verdict == _UNVERIFIABLE and action == "kept":
            action = "kept_unverifiable"
            stats["kept_unverifiable"] += 1
        actions[ui] = action
        if action in ("removed", "removed_duplicate"):
            stats["removed"] += 1
        elif action == "corrected":
            stats["corrected"] += 1
        else:
            stats["kept"] += 1

    keys = [_sdrf_header_key(h) for h in header]
    kinds = [classify_column(h) for h in header]
    for field, items in recovered.items():
        if field == "modification":
            mod_idx = [i for i, (k, f) in enumerate(kinds) if k == "modification"]
            for value, _ in items:
                target = next((i for i in mod_idx
                               if all(_is_not_extracted(r[i]) for r in body)), None)
                if target is None:
                    pos = (max(mod_idx) + 1) if mod_idx else next(
                        (i for i, k in enumerate(keys) if k.startswith("factorvalue")), len(header))
                    header.insert(pos, header[mod_idx[0]] if mod_idx
                                  else _SDRF_STANDARD_HEADER["modification"])
                    for r in body:
                        r.insert(pos, _SDRF_NA)
                    keys = [_sdrf_header_key(h) for h in header]
                    kinds = [classify_column(h) for h in header]
                    mod_idx = [i for i, (k, f) in enumerate(kinds) if k == "modification"]
                    target = pos
                for r in body:
                    r[target] = _format_cell("", value, True, "modification")
                stats["added"] += 1
            continue
        value = items[0][0]
        cols = [i for i, (k, f) in enumerate(kinds) if f == field and k == "value"]
        if cols:
            ci = cols[0]
            if all(_is_not_extracted(r[ci]) for r in body):
                kv = field in _KV_FIELDS
                for r in body:
                    r[ci] = _format_cell("", value, kv, field)
                stats["added"] += 1
        elif SDRF_ADD_MISSING_COLUMNS and field in _SDRF_STANDARD_HEADER:
            pos = next((i for i, k in enumerate(keys) if k.startswith("factorvalue")), len(header))
            header.insert(pos, _SDRF_STANDARD_HEADER[field])
            for r in body:
                r.insert(pos, _format_cell("", value, field in _KV_FIELDS, field))
            keys = [_sdrf_header_key(h) for h in header]
            kinds = [classify_column(h) for h in header]
            stats["added"] += 1
    stats["accession_drift_fixed"] = _fix_accession_drift(header, body)
    return header, body, stats, actions


#per paper evaluation

def _variant_note(u: dict) -> list[str]:
    v = u.get("variants") or []
    if len(v) < 2:
        return []
    acs = [_kv_parse(x).get("AC", "") for x in v]
    return [f"accession: {len(v)} cells differ only in AC ({acs[0]} ... {acs[-1]}); "
            f"judged once, verdict applied to all; AC reset in the filtered SDRF when it "
            f"counts up row by row"]


def evaluate_paper_with_geval(paper_id: str, header: list[str], body: list[list[str]],
                              source_text: str, recover: bool) -> dict:
    """judge all values of one SDRF CONCURRENTLY using GEval then recover and filter"""
    model = _get_judge_model()
    model.set_paper_context(source_text, paper_id=paper_id)
    paper_canon_lower = _canon(source_text).lower()
    units, columns = build_units(header, body)
    field_values = _column_values(columns)
    _, run_labels = _run_keys(header, body)
    n_runs = len(body)

    coverage_rows = []
    for c in columns:
        distinct = []
        for cell in c["cells"]:
            if not _is_not_extracted(cell) and cell.strip() not in distinct:
                distinct.append(cell.strip())
        status = ("skipped" if c["kind"] == "skip" else
                  "judged" if distinct else "all_not_available")
        coverage_rows.append({"paper_id": paper_id, "sdrf_column": c["header"],
                              "column_index": c["index"], "annotation_type": c["field"],
                              "kind": c["kind"], "n_distinct": len(distinct),
                              "distinct_values": " | ".join(_readable(d) for d in distinct)[:500],
                              "status": status})

    contexts = []
    for u in units:
        if u["kind"] != "runs":
            u["example_labels"] = [run_labels[ri] for ri in u["rows"][:MAX_EXAMPLE_RUNS]]
        related = {f: field_values.get(f, []) for f in _RELATED_FIELDS.get(u["field"], [])
                   if u["kind"] != "runs"}
        checks = unit_checks(u, paper_canon_lower)
        u["checks"] = checks
        contexts.append(_unit_context(u, related or None, n_runs, checks))

    n_cols = sum(1 for c in columns if c["kind"] != "skip")
    print(f"    SDRF: {n_runs} runs, {len(header)} columns ({n_cols} judged columns)")
    print(f"    Evaluation rows: {len(units)}")

    results: list[dict] = [{**_error_result(""), "match_type": "NOT_JUDGED"} for _ in units]
    if USE_LLM_JUDGE and units:
        print(f"    [judge] {len(units)} values with {MAX_WORKERS} workers (GEval) ...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_run_geval_semantic, ctx, u["kind"], u.get("run_keys")): i
                       for i, (u, ctx) in enumerate(zip(units, contexts))}
            for n_done, future in enumerate(as_completed(futures), 1):
                i = futures[future]
                u = units[i]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    results[i] = _error_result(f"Thread error: {exc}")
                r = results[i]
                if r.get("verdict") is not None:
                    cv_info = (f"  corrected='{r['corrected_value'][:40]}'"
                               if r.get("corrected_value") else "")
                    print(f"      [{n_done}/{len(units)}] {u['header']}: '{u['readable'][:40]}' "
                          f"verdict={r['verdict']}  correct={r['value_correct']}  "
                          f"halluc={r['hallucination']}  mismatch={r['type_mismatch']}{cv_info}")
                else:
                    print(f"      [{n_done}/{len(units)}] {u['header']}: '{u['readable'][:40]}' ERROR")

    #per-sample values the text cannot confirm; the recovery pass below sees the verdicts
    #as the judge gave them so its requests (and the cache) are unchanged
    pre_guard_verdicts = [r.get("verdict") for r in results]
    if USE_LLM_JUDGE:
        _sample_sheet_guard(units, results, columns, source_text)

    #accepted and rejected values per field after review then the recovery pass
    accepted, rejected, moved = defaultdict(list), defaultdict(list), defaultdict(list)
    for u, r, pre in zip(units, results, pre_guard_verdicts):
        if u["kind"] == "runs":
            continue
        if pre == "low":
            rejected[u["field"]].append(u["readable"])
            if r.get("type_mismatch") and r.get("correct_type"):
                moved[r["correct_type"]].append(u["readable"])
        elif r.get("verdict") == "medium" and r.get("corrected_value"):
            accepted[u["field"]].append(_readable(r["corrected_value"]))
        else:
            accepted[u["field"]].append(u["readable"])

    recovered, recovered_rows = {}, []
    if recover and USE_LLM_JUDGE:
        recovered, recovered_rows = recover_missed_values(paper_id, accepted, rejected, moved,
                                                          paper_canon_lower)

    new_header, new_body, filter_stats, actions = apply_to_sdrf(header, body, units, results,
                                                                recovered)
    for row in recovered_rows:
        if row["accepted"]:
            f = row["annotation_type"]
            has_col = any(classify_column(h)[1] == f for h in new_header)
            row["sdrf_column"] = (next(h for h in new_header if classify_column(h)[1] == f)
                                  if has_col else "(no column: CSV only)")

    eval_rows = []
    for i, (u, r) in enumerate(zip(units, results)):
        eval_rows.append({
            "paper_id": paper_id, "sdrf_column": u["header"],
            "annotation_type": u["field"], "extracted_value": u["cell"],
            "all_values_for_field": " | ".join([u["readable"]] + u["siblings"]),
            "verdict": r.get("verdict"), "type_mismatch": r.get("type_mismatch"),
            "correct_type": r.get("correct_type"), "value_correct": r.get("value_correct"),
            "value_complete": r.get("value_complete"), "hallucination": r.get("hallucination"),
            "issue_summary": r.get("issue_summary", ""), "corrected_value": r.get("corrected_value"),
            "match_type": r.get("match_type"), "geval_score": r.get("geval_score"),
            "kind": u["kind"], "column_index": ",".join(str(c) for c in u["columns"]),
            "n_runs": len(u["rows"]), "filter_action": actions.get(i, ""),
            "checks": " | ".join(u.get("checks", []) + _variant_note(u)),
        })

    return {"eval_rows": eval_rows, "coverage_rows": coverage_rows,
            "recovered_rows": recovered_rows, "filtered": (new_header, new_body),
            "filter_stats": filter_stats,
            "n_recovered": sum(1 for r in recovered_rows if r["accepted"])}


#incremental CSV writers and statistics

#column names for the main annotation review CSV
_REVIEW_CSV_FIELDS = [
    "paper_id", "annotation_type", "extracted_value", "verdict", "match_type",
    "type_mismatch", "correct_type", "value_correct", "value_complete", "hallucination",
    "corrected_value", "issue_summary",
]
#same rows with every detail for debugging (llm_judge_annotation_review_full.csv)
_REVIEW_FULL_CSV_FIELDS = [
    "paper_id", "sdrf_column", "annotation_type", "extracted_value", "all_values_for_field",
    "verdict", "match_type", "type_mismatch", "correct_type", "value_correct",
    "value_complete", "hallucination", "corrected_value", "issue_summary",
    "geval_score", "kind", "column_index", "n_runs", "filter_action", "checks",
]


def _review_rows(eval_rows: list[dict]) -> list[dict]:
    """rows for the review CSV: a column the judge has no field for is named by its header"""
    out = []
    for r in eval_rows:
        row = dict(r)
        if row.get("annotation_type") == _GENERIC_FIELD:
            row["annotation_type"] = row.get("sdrf_column", _GENERIC_FIELD)
        out.append(row)
    return out

#column names for the per paper statistics CSV (uploaded judge columns first)
_STATS_CSV_FIELDS = [
    "paper_id", "total_extracted",
    "judge_n_correct", "judge_n_hallucinated", "judge_n_mismatch",
    "judge_n_wrong", "judge_n_incomplete", "judge_n_corrected",
    "judge_accuracy",
    "judge_n_high", "judge_n_medium", "judge_n_low", "judge_n_error",
    "judge_accuracy_excl_errors", "judge_n_recovered", "n_runs", "n_columns",
    "judge_n_unverifiable", "judge_accuracy_verifiable",
]

#column names for the column coverage CSV
_COVERAGE_CSV_FIELDS = [
    "paper_id", "sdrf_column", "column_index", "annotation_type", "kind",
    "n_distinct", "distinct_values", "status",
]

_RECOVERED_CSV_FIELDS = [
    "paper_id", "annotation_type", "sdrf_column", "recovered_value", "evidence",
    "evidence_check", "verdict", "final_value", "accepted", "reason", "issue_summary",
]
_FILTER_CSV_FIELDS = ["file_name", "total", "kept", "removed", "corrected", "added",
                      "unchanged_low_run_columns", "kept_unverifiable", "accession_drift_fixed"]
_PER_FIELD_CSV_FIELDS = [
    "annotation_type", "n_papers_with_column", "judge_n_total", "judge_n_high",
    "judge_n_medium", "judge_n_low", "judge_n_error", "judge_n_hallucinated",
    "judge_n_mismatch", "judge_n_wrong_value", "judge_n_corrected", "judge_n_recovered",
    "judge_accuracy", "judge_n_unverifiable", "judge_accuracy_verifiable",
]


#longest text written to one CSV cell excel holds at most 32767 characters per cell and
#splits longer cells over several rows which breaks the table run mappings longer than
#this are written grouped by value; the full mapping stays in json_outputs/<PXD>.json
CSV_MAX_CELL_CHARS = 30000
_RUN_GROUP_EXAMPLES = 3


def _compact_run_mapping(text: str) -> str:
    """'a.raw = 1 | b.raw = 1 | c.raw = 2' -> '1 = 2 runs (a.raw, b.raw) | 2 = 1 run (c.raw)'"""
    parts = [p.strip() for p in text.split(" | ")]
    if len(parts) < 2 or not all(" = " in p for p in parts):
        return text
    groups: dict[str, list[str]] = {}
    for p in parts:
        key, value = p.split(" = ", 1)
        groups.setdefault(value.strip(), []).append(key.strip())
    out = []
    for value, keys in groups.items():
        shown = keys[:_RUN_GROUP_EXAMPLES]
        more = f", ... {keys[-1]}" if len(keys) > _RUN_GROUP_EXAMPLES else ""
        noun = "run" if len(keys) == 1 else "runs"
        out.append(f"{value} = {len(keys)} {noun} ({', '.join(shown)}{more})")
    return f"[{len(parts)} runs, grouped by value] " + " | ".join(out)


def _csv_cell(value):
    """a cell value that fits one spreadsheet cell; long run mappings are grouped first"""
    if not isinstance(value, str) or len(value) <= CSV_MAX_CELL_CHARS:
        return value
    lines = []
    for line in value.split("\n"):
        if len(line) > 2000 and " = " in line and " | " in line:
            label, sep, rest = line.partition(":")
            if sep and " = " not in label:
                line = f"{label}:{(' ' if rest[:1] == ' ' else '')}" \
                       f"{_compact_run_mapping(rest.strip())}"
            else:
                line = _compact_run_mapping(line)
        lines.append(line)
    value = "\n".join(lines)
    if len(value) > CSV_MAX_CELL_CHARS:
        note = " ... [truncated; full text in json_outputs]"
        value = value[:CSV_MAX_CELL_CHARS - len(note)] + note
    return value


class _IncrementalCSV:
    """write CSV rows incrementally to disk flushing after each batch"""

    def __init__(self, path: str, fieldnames: list[str]):
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames,
                                      quoting=csv.QUOTE_ALL, extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()

    def append_rows(self, rows: list[dict]) -> None:
        for row in rows:
            self._writer.writerow({k: _csv_cell(v) for k, v in row.items()})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _is_true(v) -> bool:
    return v is True or str(v) == "True"


def _is_false(v) -> bool:
    return v is False or str(v) == "False"


def _compute_single_paper_stats(paper_id: str, eval_rows: list[dict],
                                extra: dict | None = None) -> dict:
    """compute quality category counts and accuracy for one paper's evaluation rows"""
    total = len(eval_rows)
    rec = {"paper_id": paper_id, "total_extracted": total}
    judged = [r for r in eval_rows if r.get("value_correct") not in (None, "")
              or str(r.get("verdict") or "") == _UNVERIFIABLE]
    if judged:
        n_correct = sum(1 for r in eval_rows
                        if _is_false(r.get("type_mismatch"))
                        and _is_true(r.get("value_correct"))
                        and _is_true(r.get("value_complete"))
                        and _is_false(r.get("hallucination")))
        rec["judge_n_correct"] = n_correct
        rec["judge_n_hallucinated"] = sum(1 for r in eval_rows if _is_true(r.get("hallucination")))
        rec["judge_n_mismatch"] = sum(1 for r in eval_rows if _is_true(r.get("type_mismatch")))
        rec["judge_n_wrong"] = sum(1 for r in eval_rows
                                   if _is_false(r.get("value_correct"))
                                   and not _is_true(r.get("hallucination")))
        rec["judge_n_incomplete"] = sum(1 for r in eval_rows
                                        if _is_true(r.get("value_correct"))
                                        and _is_false(r.get("value_complete")))
        rec["judge_n_corrected"] = sum(1 for r in eval_rows
                                       if r.get("corrected_value") not in (None, ""))
        rec["judge_accuracy"] = round(n_correct / total, 4) if total > 0 else 0.0
        verdicts = [str(r.get("verdict") or "") for r in eval_rows]
        rec["judge_n_high"] = verdicts.count("high")
        rec["judge_n_medium"] = verdicts.count("medium")
        rec["judge_n_low"] = verdicts.count("low")
        rec["judge_n_error"] = sum(1 for r in eval_rows if r.get("match_type") == "ERROR")
        n_ok = total - rec["judge_n_error"]
        rec["judge_accuracy_excl_errors"] = round(n_correct / n_ok, 4) if n_ok > 0 else 0.0
        rec["judge_n_unverifiable"] = verdicts.count(_UNVERIFIABLE)
        n_ver = n_ok - rec["judge_n_unverifiable"]
        rec["judge_accuracy_verifiable"] = round(n_correct / n_ver, 4) if n_ver > 0 else 0.0
    if extra:
        rec.update(extra)
    return rec


def _compute_per_field_stats(all_rows: list[dict], all_cov: list[dict],
                             all_rec: list[dict]) -> list[dict]:
    rows_by, rec_by, papers_by = defaultdict(list), defaultdict(int), defaultdict(set)
    for r in all_rows:
        rows_by[r["annotation_type"]].append(r)
    for r in all_cov:
        if r["status"] != "skipped" and r["annotation_type"]:
            papers_by[r["annotation_type"]].add(r["paper_id"])
    for r in all_rec:
        if _is_true(r.get("accepted")):
            rec_by[r["annotation_type"]] += 1
    out = []
    for name in _FIELD_ORDER + [_GENERIC_FIELD]:
        if name not in rows_by and name not in rec_by and name not in papers_by:
            continue
        stat = _compute_single_paper_stats(name, rows_by.get(name, []))
        out.append({
            "annotation_type": name,
            "n_papers_with_column": len(papers_by.get(name, ())),
            "judge_n_total": stat["total_extracted"],
            "judge_n_high": stat.get("judge_n_high", 0),
            "judge_n_medium": stat.get("judge_n_medium", 0),
            "judge_n_low": stat.get("judge_n_low", 0),
            "judge_n_error": stat.get("judge_n_error", 0),
            "judge_n_hallucinated": stat.get("judge_n_hallucinated", 0),
            "judge_n_mismatch": stat.get("judge_n_mismatch", 0),
            "judge_n_wrong_value": sum(1 for r in rows_by.get(name, [])
                                       if r.get("match_type") == "NO_MATCH"),
            "judge_n_corrected": stat.get("judge_n_corrected", 0),
            "judge_n_recovered": rec_by.get(name, 0),
            "judge_accuracy": stat.get("judge_accuracy", ""),
            "judge_n_unverifiable": stat.get("judge_n_unverifiable", 0),
            "judge_accuracy_verifiable": stat.get("judge_accuracy_verifiable", ""),
        })
    return out


def _write_paper_json(pxd_id: str, eval_rows: list[dict], json_dir: str) -> None:
    """write evaluation results for one paper to a JSON file"""
    os.makedirs(json_dir, exist_ok=True)
    entries = []
    for row in eval_rows:
        entries.append({
            "annotation_type": row.get("annotation_type"),
            "sdrf_column": row.get("sdrf_column"),
            "extracted_value": row.get("extracted_value"),
            "verdict": row.get("verdict"),
            "type_mismatch": row.get("type_mismatch"),
            "value_correct": row.get("value_correct"),
            "value_complete": row.get("value_complete"),
            "hallucination": row.get("hallucination"),
            "corrected_value": row.get("corrected_value"),
            "issue_summary": _parse_checks_to_dict(row.get("issue_summary") or ""),
        })
    path = os.path.join(json_dir, f"{pxd_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"paper_id": pxd_id, "annotations": entries}, f, indent=2, ensure_ascii=False)
    print(f"    JSON written: {path}")


#plotting (same charts and file names as the uploaded judges)

def plot_results(df: pd.DataFrame, per_paper_df: pd.DataFrame, model_name: str, out_dir: str):
    """generate and save three summary PNG charts from evaluation results"""
    if df.empty or per_paper_df.empty:
        return

    BG, GRID_C, TEXT_C, SUB_C, SPINE_C = "white", "#E5E7EB", "#111827", "#6B7280", "#D1D5DB"
    CLR = dict(correct="#27ae60", hall="#e74c3c", mismatch="#9b59b6",
               wrong="#e67e22", incomplete="#3498db")

    def style_ax(ax):
        """apply consistent background grid and spine styling to a matplotlib Axes"""
        ax.set_facecolor(BG)
        ax.yaxis.grid(True, color=GRID_C, lw=0.8, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(SPINE_C)
        ax.tick_params(colors=TEXT_C, labelsize=9)

    required = ["judge_n_correct", "judge_n_hallucinated", "judge_n_mismatch",
                "judge_n_wrong", "judge_n_incomplete"]
    if not all(c in per_paper_df.columns for c in required):
        print("  [plot] skipping -- judge columns missing from per_paper_df")
        return

    names = per_paper_df["paper_id"].tolist()
    x = np.arange(len(names))
    w = 0.6
    num = lambda c: pd.to_numeric(per_paper_df[c], errors="coerce").fillna(0).values
    n_correct, n_hall, n_mismatch = num("judge_n_correct"), num("judge_n_hallucinated"), num("judge_n_mismatch")
    n_wrong, n_incomplete, totals = num("judge_n_wrong"), num("judge_n_incomplete"), num("total_extracted")

    #stacked bar chart showing annotation quality categories per paper
    fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.7), 7), facecolor=BG)
    style_ax(ax)
    b = np.zeros(len(names))
    for arr, col, lbl in [
        (n_correct, CLR["correct"], "Correct"),
        (n_hall, CLR["hall"], "Hallucinated"),
        (n_mismatch, CLR["mismatch"], "Type Mismatch"),
        (n_wrong, CLR["wrong"], "Wrong Value"),
        (n_incomplete, CLR["incomplete"], "Incomplete"),
        (num("judge_n_unverifiable") if "judge_n_unverifiable" in per_paper_df.columns
         else np.zeros(len(names)), "#bdc3c7", "Not in text"),
    ]:
        ax.bar(x, arr, w, bottom=b, label=lbl, color=col, edgecolor="white", lw=0.5, alpha=0.92)
        b += arr
    for i, tot in enumerate(totals):
        ax.text(i, b[i] + 0.3, f"n={int(tot)}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color=TEXT_C)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_C)
    ax.set_ylabel("Number of Annotations", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} Annotation Quality per Paper",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE_C, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_annotation_quality_counts.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: llm_judge_annotation_quality_counts.png")

    if "judge_accuracy" not in per_paper_df.columns:
        return
    acc_vals = pd.to_numeric(per_paper_df["judge_accuracy"], errors="coerce").fillna(0).tolist()
    overall = float(np.mean(acc_vals)) if acc_vals else 0.0
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.55), 6), facecolor=BG)
    style_ax(ax)
    #colour bars green above 70% orange above 50% red below 50%
    bar_colors = ["#27ae60" if v >= 0.7 else "#e67e22" if v >= 0.5 else "#e74c3c"
                  for v in acc_vals]
    bars = ax.bar(x, acc_vals, color=bar_colors, alpha=0.88, edgecolor="white",
                  linewidth=0.8, zorder=3)
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=TEXT_C)
    ax.axhline(overall, color="#2471a3", lw=2, linestyle="--", label=f"Mean = {overall:.3f}")
    ax.axhline(0.7, color="#e74c3c", lw=1.5, linestyle=":", alpha=0.7, label="Threshold = 0.70")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_C)
    ax.set_ylabel("Accuracy (Correct / Total Extracted)", fontsize=11, color=SUB_C)
    ax.set_ylim([0, 1.18])
    ax.set_title(f"{model_name.upper()} LLM Judge Accuracy per Paper",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor=SPINE_C)
    ax.text(0.98, 0.97, f"Overall: {overall:.1%}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold", color="#27ae60",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor="#27ae60", alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_accuracy.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: llm_judge_accuracy.png")

    #aggregate bar chart summing all quality category counts across every paper
    agg_labels = ["Correct", "Hallucinated", "Type Mismatch", "Wrong Value", "Incomplete"]
    agg_vals = [int(n_correct.sum()), int(n_hall.sum()), int(n_mismatch.sum()),
                int(n_wrong.sum()), int(n_incomplete.sum())]
    agg_colors = [CLR["correct"], CLR["hall"], CLR["mismatch"], CLR["wrong"], CLR["incomplete"]]
    grand_total = sum(agg_vals)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    style_ax(ax)
    bars5 = ax.bar(agg_labels, agg_vals, color=agg_colors, edgecolor="white",
                   linewidth=1.0, alpha=0.92, zorder=3)
    for bar, val in zip(bars5, agg_vals):
        pct = val / grand_total * 100 if grand_total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_C)
    overall_acc = agg_vals[0] / grand_total * 100 if grand_total > 0 else 0
    ax.set_ylabel("Total Count (all papers)", fontsize=11, color=SUB_C)
    ax.set_title(f"{model_name.upper()} Aggregate Annotation Quality (total = {grand_total})",
                 fontsize=13, fontweight="bold", color=TEXT_C, pad=12)
    ax.text(0.5, 0.97, f"Overall accuracy: {overall_acc:.1f}%",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold", color=CLR["correct"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1",
                      edgecolor=CLR["correct"], alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_aggregate.png"),
                dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: llm_judge_aggregate.png")


def plot_match_distribution(df: pd.DataFrame, output_dir: str):
    """match category distribution per paper and in aggregate (judge_foodomics_V3.py charts)"""
    if df.empty or "match_type" not in df.columns:
        return
    KEYS = ["CORRECT", "PARTIAL", "TYPE_MISMATCH", "HALLUCINATED", "NO_MATCH", "NOT_IN_TEXT"]
    COLORS = ["#10B981", "#3B82F6", "#8B5CF6", "#EF4444", "#F59E0B", "#9CA3AF"]
    LABELS = ["Correct", "Partial", "Type Mismatch", "Hallucinated", "Wrong Value",
              "Not in text"]

    paper_ids = sorted(df["paper_id"].astype(str).unique())
    counts = {pid: {k: 0 for k in KEYS} for pid in paper_ids}
    for pid, mt in zip(df["paper_id"].astype(str), df["match_type"].astype(str)):
        if mt in ("ERROR", "NOT_JUDGED", "", "nan"):
            continue  #judge errors are not wrong values
        counts[pid][mt if mt in KEYS else "NO_MATCH"] += 1

    names = paper_ids
    x = np.arange(len(names))
    w = 0.6
    data = {k: np.array([counts[p][k] for p in names], float) for k in KEYS}
    totals = sum(data[k] for k in KEYS)

    fig, axes = plt.subplots(1, 2, figsize=(max(18, len(names) * 1.4), 7))
    fig.suptitle("Match Category Distribution per Paper", fontsize=14, fontweight="bold")
    for ax, normalise in zip(axes, [False, True]):
        denom = np.where(totals == 0, 1, totals)
        b = np.zeros(len(names))
        for k, col, lbl in zip(KEYS, COLORS, LABELS):
            pd_ = data[k] / denom * 100 if normalise else data[k]
            bars = ax.bar(x, pd_, w, bottom=b, color=col, alpha=0.88,
                          edgecolor="white", linewidth=0.5, label=lbl)
            for bar, val, bot in zip(bars, pd_, b):
                if val >= (5 if normalise else 2):
                    ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                            f"{val:.0f}%" if normalise else str(int(val)),
                            ha="center", va="center", fontsize=7, fontweight="bold", color="white")
            b += pd_
        if not normalise:
            for i, tot in enumerate(totals):
                ax.text(i, tot + 0.3, f"n={int(tot)}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#333")
        else:
            ax.set_ylim(0, 115)
            ax.axhline(100, color="#aaa", lw=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("% of Annotations" if normalise else "Number of Annotations", fontsize=11)
        ax.set_title("Match Distribution (normalised %)" if normalise
                     else "Match Distribution (raw counts)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "match_distribution_per_paper.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  Saved: match_distribution_per_paper.png")

    agg = {k: int(sum(data[k])) for k in KEYS}
    grand = sum(agg.values())
    fig, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Aggregate Match Category Distribution  (total = {grand})",
                 fontsize=14, fontweight="bold")
    ax3 = axes2[0]
    wlbls = [f"{lbl}\n{agg[k]} ({agg[k] / grand * 100:.1f}%)" if grand else lbl
             for lbl, k in zip(LABELS, KEYS)]
    if grand:
        wedges, _ = ax3.pie([agg[k] for k in KEYS], colors=COLORS, startangle=90,
                            wedgeprops=dict(edgecolor="white", linewidth=1.5))
        ax3.legend(wedges, wlbls, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        ax3.add_artist(plt.Circle((0, 0), 0.55, color="white"))
    ax3.set_title("Donut breakdown", fontsize=11)
    matched_pct = (agg["CORRECT"] + agg["PARTIAL"]) / grand * 100 if grand else 0
    ax3.text(0, 0, f"{matched_pct:.1f}%\nmatched", ha="center", va="center",
             fontsize=11, fontweight="bold", color="#333")
    ax4 = axes2[1]
    xb = np.arange(len(KEYS))
    bars4 = ax4.bar(xb, [agg[k] for k in KEYS], color=COLORS, alpha=0.88,
                    edgecolor="white", linewidth=1.0)
    for bar, val in zip(bars4, [agg[k] for k in KEYS]):
        pct = val / grand * 100 if grand else 0
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_xticks(xb)
    ax4.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=9)
    ax4.set_ylabel("Total count (all papers)", fontsize=11)
    ax4.set_title("Bar breakdown", fontsize=11)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    ax4.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "match_distribution_aggregate.png"), dpi=300,
                bbox_inches="tight")
    plt.close()
    print("  Saved: match_distribution_aggregate.png")


def plot_per_field(per_field_df: pd.DataFrame, out_dir: str, title: str):
    """verdict counts per SDRF field the view for finding the weakest columns"""
    if per_field_df.empty:
        return
    df = per_field_df.copy()
    order = {f: i for i, f in enumerate(_FIELD_ORDER + [_GENERIC_FIELD])}
    df["order"] = df["annotation_type"].map(order).fillna(999)
    df = df.sort_values("order")
    y = np.arange(len(df))
    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0).values
    fig, ax = plt.subplots(figsize=(12, max(6, 0.32 * len(df))))
    left = np.zeros(len(df))
    for col, color, label in [("judge_n_high", "#27ae60", "High"),
                              ("judge_n_medium", "#3498db", "Medium"),
                              ("judge_n_low", "#e74c3c", "Low"),
                              ("judge_n_unverifiable", "#bdc3c7", "Not in text"),
                              ("judge_n_error", "#95a5a6", "Error"),
                              ("judge_n_recovered", "#9b59b6", "Recovered")]:
        if col not in df.columns:
            continue
        vals = num(col)
        ax.barh(y, vals, left=left, color=color, label=label, edgecolor="white", lw=0.5)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(df["annotation_type"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of judged values")
    ax.set_title(f"{title.upper()} Judge Verdicts per SDRF Field", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "llm_judge_per_field.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: llm_judge_per_field.png")


#main pipeline (incremental output after each paper)

def load_manuscript(pxd_id: str) -> str:
    """load the manuscript text for a PXD and drop repeated paragraphs"""
    path = Path(DATA_ROOT) / SPLIT / pxd_id / "manuscript.txt"
    if not path.exists():
        print(f"    WARNING: manuscript not found: {path}")
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    seen, kept, n_dup = set(), [], 0
    for line in text.splitlines():
        key = _norm(line)
        if len(key) >= DEDUP_MIN_CHARS:
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
        kept.append(line)
    result = "\n".join(kept).strip()
    print(f"    Manuscript: {len(text):,} chars total, using {len(result):,} chars"
          + (f" ({n_dup} repeated paragraph(s) dropped)" if n_dup else ""))
    return result


def discover_pxds() -> list[str]:
    split_dir = Path(DATA_ROOT) / SPLIT
    if not split_dir.is_dir():
        raise SystemExit(f"Split folder not found: {split_dir}")
    pxds = sorted(p.name for p in split_dir.iterdir()
                  if p.is_dir() and p.name.upper().startswith("PXD"))
    if ONLY_PXDS:
        wanted = {p.strip().upper() for p in ONLY_PXDS}
        missing = wanted - set(pxds)
        if missing:
            print(f"  WARNING: not in {SPLIT}: {', '.join(sorted(missing))}")
        pxds = [p for p in pxds if p in wanted]
    if LIMIT and LIMIT < len(pxds):
        pxds = pxds[:LIMIT]
        print(f"  Limit applied: running first {LIMIT} papers")
    return pxds


def process_split(out_dir: str, recover: bool):
    """judge every SDRF of the split"""
    pxd_ids = discover_pxds()
    print(f"  Papers discovered: {len(pxd_ids)}")
    n_papers = len(pxd_ids)

    review_csv = _IncrementalCSV(os.path.join(out_dir, "llm_judge_annotation_review.csv"),
                                 _REVIEW_CSV_FIELDS)
    review_full_csv = _IncrementalCSV(
        os.path.join(out_dir, "llm_judge_annotation_review_full.csv"), _REVIEW_FULL_CSV_FIELDS)
    stats_csv = _IncrementalCSV(os.path.join(out_dir, "llm_judge_per_paper.csv"),
                                _STATS_CSV_FIELDS)
    coverage_csv = _IncrementalCSV(os.path.join(out_dir, "llm_judge_coverage.csv"),
                                   _COVERAGE_CSV_FIELDS)
    recovered_csv = _IncrementalCSV(os.path.join(out_dir, "llm_judge_recovered.csv"),
                                    _RECOVERED_CSV_FIELDS)
    filtered_dir = os.path.join(out_dir, "filtered_sdrf")
    os.makedirs(filtered_dir, exist_ok=True)
    filter_csv = _IncrementalCSV(os.path.join(filtered_dir, "_filter_summary.csv"),
                                 _FILTER_CSV_FIELDS)
    json_out_dir = os.path.join(out_dir, "json_outputs")
    os.makedirs(json_out_dir, exist_ok=True)

    print()
    print("Output mode: INCREMENTAL -- CSVs, JSON and filtered SDRF written after each paper.")
    print(f"  Review CSV    : {os.path.join(out_dir, 'llm_judge_annotation_review.csv')}")
    print(f"  Per-paper CSV : {os.path.join(out_dir, 'llm_judge_per_paper.csv')}")
    print(f"  Filtered SDRF : {filtered_dir}/")
    print()

    all_rows, all_cov_rows, all_rec_rows, per_file_stats, skipped = [], [], [], [], []

    for paper_idx, pxd_id in enumerate(pxd_ids, 1):
        t_start = time.time()
        print(f"  PAPER [{paper_idx}/{n_papers}]: {pxd_id}")

        source_text = load_manuscript(pxd_id)
        if not source_text:
            print("    SKIP: no manuscript text found.")
            skipped.append({"paper_id": pxd_id, "reason": "no manuscript"})
            continue
        sdrf_path = find_sdrf(pxd_id)
        if sdrf_path is None:
            print("    SKIP: no SDRF file found.")
            skipped.append({"paper_id": pxd_id, "reason": "no sdrf"})
            continue
        header, body = read_sdrf(sdrf_path)
        if not header or not body:
            print(f"    SKIP: empty SDRF {sdrf_path}")
            skipped.append({"paper_id": pxd_id, "reason": "empty sdrf"})
            continue
        print(f"    SDRF loaded: {sdrf_path}")

        result = evaluate_paper_with_geval(pxd_id, header, body, source_text, recover)
        eval_rows = result["eval_rows"]

        coverage_csv.append_rows(result["coverage_rows"])
        review_csv.append_rows(_review_rows(eval_rows))
        review_full_csv.append_rows(_review_rows(eval_rows))
        recovered_csv.append_rows(result["recovered_rows"])
        paper_stat = _compute_single_paper_stats(pxd_id, eval_rows, {
            "judge_n_recovered": result["n_recovered"], "n_runs": len(body),
            "n_columns": len(header)})
        stats_csv.append_rows([paper_stat])

        if USE_LLM_JUDGE:
            _write_paper_json(pxd_id, eval_rows, json_out_dir)
            new_header, new_body = result["filtered"]
            out_path = os.path.join(filtered_dir, sdrf_path.name)
            write_sdrf(out_path, new_header, new_body)
            filter_csv.append_rows([{"file_name": pxd_id, **result["filter_stats"]}])
            print(f"    Filtered SDRF written: {out_path}")

        all_rows.extend(eval_rows)
        all_cov_rows.extend(result["coverage_rows"])
        all_rec_rows.extend(result["recovered_rows"])
        per_file_stats.append(paper_stat)

        elapsed = time.time() - t_start
        print(f"\n  PAPER COMPLETE: {pxd_id}  ({elapsed:.1f}s)")
        print(f"    total={paper_stat.get('total_extracted', 0)}  "
              f"correct={paper_stat.get('judge_n_correct', 0)}  "
              f"halluc={paper_stat.get('judge_n_hallucinated', 0)}  "
              f"mismatch={paper_stat.get('judge_n_mismatch', 0)}  "
              f"wrong={paper_stat.get('judge_n_wrong', 0)}  "
              f"incomplete={paper_stat.get('judge_n_incomplete', 0)}  "
              f"corrected={paper_stat.get('judge_n_corrected', 0)}  "
              f"recovered={result['n_recovered']}  error={paper_stat.get('judge_n_error', 0)}")
        acc = paper_stat.get("judge_accuracy")
        if acc is not None:
            print(f"    accuracy={acc:.1%}")
        print("    results written to disk.\n")

    for w in (review_csv, review_full_csv, stats_csv, coverage_csv, recovered_csv, filter_csv):
        w.close()

    per_field = _compute_per_field_stats(all_rows, all_cov_rows, all_rec_rows)
    w = _IncrementalCSV(os.path.join(out_dir, "llm_judge_per_field.csv"), _PER_FIELD_CSV_FIELDS)
    w.append_rows(per_field)
    w.close()
    if skipped:
        w = _IncrementalCSV(os.path.join(out_dir, "llm_judge_skipped.csv"), ["paper_id", "reason"])
        w.append_rows(skipped)
        w.close()

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    per_paper_df = pd.DataFrame(per_file_stats) if per_file_stats else pd.DataFrame()
    per_field_df = pd.DataFrame(per_field) if per_field else pd.DataFrame()
    return df, per_paper_df, per_field_df


def _plots_from_csv(out_dir: str) -> None:
    """rebuild the summary plots from the CSVs already in out_dir"""
    df = pd.read_csv(os.path.join(out_dir, "llm_judge_annotation_review_full.csv"), dtype=str)
    per_paper_df = pd.read_csv(os.path.join(out_dir, "llm_judge_per_paper.csv"),
                               dtype={"paper_id": str})
    per_field_df = pd.read_csv(os.path.join(out_dir, "llm_judge_per_field.csv"))
    title = os.path.basename(os.path.normpath(out_dir))
    plot_results(df, per_paper_df, title, out_dir)
    plot_match_distribution(df, out_dir)
    plot_per_field(per_field_df, out_dir, title)


def main():
    """parse command line arguments and run the full evaluation pipeline"""
    global DATA_ROOT, SPLIT, SDRF_DIR, USE_LLM_JUDGE, LIMIT, MAX_WORKERS, CACHE_ENABLED
    global ONLY_PXDS, _JUDGE_CACHE_DIR, _judge_model

    #a crash prints the Python stack instead of a bare "segmentation fault"
    faulthandler.enable()
    #larger stack for every thread created from here on (the worker pools)
    try:
        threading.stack_size(THREAD_STACK_MB * 1024 * 1024)
    except (ValueError, RuntimeError) as exc:
        print(f"  [warning] could not set thread stack size: {exc}")

    parser = argparse.ArgumentParser(
        description="GEval LLM judge for pipeline SDRF-Proteomics files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --only PXD000070 PXD002029 PXD000312 --outdir test_3 --no-judge   (check inputs, no API calls)
  --only PXD000070 PXD002029 PXD000312 --outdir test_3              (judge three datasets)
  (no arguments)                                                    (judge the whole train set)
  --plots-only --outdir test_3
""")
    parser.add_argument("--data-root", default=None, help=f"default: {DATA_ROOT}")
    parser.add_argument("--split", default=None, choices=["train_set", "test_set"],
                        help=f"default: {SPLIT}")
    parser.add_argument("--sdrf-dir", default=None,
                        help="folder with the SDRF files to judge (default: the split folder)")
    parser.add_argument("--outdir", default=None,
                        help="evaluation folder (default: RESULTS_ROOT/evaluation_Gemma_<split>)")
    parser.add_argument("--only", nargs="+", default=None, metavar="PXD")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge API calls")
    parser.add_argument("--no-recover", action="store_true",
                        help="skip the recovery pass for values the SDRF does not hold")
    parser.add_argument("--no-cache", action="store_true", help="disable the disk response cache")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N papers")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Max parallel LLM calls (default: {MAX_WORKERS})")
    parser.add_argument("--plots-only", action="store_true",
                        help="redraw the plots from the CSVs in --outdir, no judging")
    args = parser.parse_args()

    DATA_ROOT = args.data_root or DATA_ROOT
    SPLIT = args.split or SPLIT
    SDRF_DIR = args.sdrf_dir if args.sdrf_dir is not None else SDRF_DIR
    if args.no_judge:
        USE_LLM_JUDGE = False
    if args.limit:
        LIMIT = args.limit
    if args.workers:
        MAX_WORKERS = args.workers
    if args.no_cache:
        CACHE_ENABLED = False
    if args.only:
        ONLY_PXDS = [p for item in args.only for p in item.split(",") if p.strip()]
    recover = RECOVER_MISSED and not args.no_recover

    out_dir = args.outdir or os.path.join(RESULTS_ROOT, f"evaluation_Gemma_{SPLIT}")
    if args.plots_only:
        _plots_from_csv(out_dir)
        return
    os.makedirs(out_dir, exist_ok=True)
    _JUDGE_CACHE_DIR = os.path.join(RESULTS_ROOT, f".prompt_cache_{_CACHE_MODEL_SLUG}")
    _judge_model = None  #force re-init with the cache dir

    sdrf_src = SDRF_DIR or f"{Path(DATA_ROOT) / SPLIT}/<PXD>/<PXD>.sdrf.tsv"
    print(f"  SPLIT          : {Path(DATA_ROOT) / SPLIT}")
    print(f"  SDRF input     : {sdrf_src}")
    print(f"  Manuscripts    : {Path(DATA_ROOT) / SPLIT}/<PXD>/manuscript.txt")
    print(f"  Output folder  : {out_dir}")
    print(f"  Judge model    : {EVALUATION_MODEL}  (via OpenRouter)")
    print(f"  Temperature    : {MODEL_TEMPERATURE}")
    print(f"  Thinking       : {'ENABLED' if MODEL_ENABLE_THINKING else 'DISABLED'}")
    print(f"  Disk cache     : {'ENABLED' if CACHE_ENABLED else 'DISABLED'}  dir={_JUDGE_CACHE_DIR}")
    print(f"  Prompt         : {PROMPT_VERSION}  fingerprint={_PROMPT_FINGERPRINT}")
    print(f"  LLM judge      : {'ENABLED' if USE_LLM_JUDGE else 'DISABLED'}")
    print(f"  Recover missed : {'ON' if recover and USE_LLM_JUDGE else 'OFF'}")
    print(f"  Limit          : {LIMIT if LIMIT else 'ALL papers'}")
    print(f"  Concurrency    : {MAX_WORKERS} workers")
    print()

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"data_root": DATA_ROOT, "split": SPLIT, "sdrf_dir": SDRF_DIR,
                   "judge_model": EVALUATION_MODEL, "thinking": MODEL_ENABLE_THINKING,
                   "prompt_version": PROMPT_VERSION, "prompt_fingerprint": _PROMPT_FINGERPRINT,
                   "recover": recover, "use_llm_judge": USE_LLM_JUDGE,
                   "sdrf_add_missing_columns": SDRF_ADD_MISSING_COLUMNS,
                   "skip_columns": sorted(SKIP_COLUMNS)}, f, indent=2)

    df, per_paper_df, per_field_df = process_split(out_dir, recover)

    if df.empty:
        print("ERROR: No data processed.")
        return

    print(f"\n  Annotation review : {os.path.join(out_dir, 'llm_judge_annotation_review.csv')}")
    print(f"  Total rows        : {len(df)}")

    if "corrected_value" in df.columns:
        medium_df = df[(df["verdict"] == "medium") & df["corrected_value"].notna()]
        if not medium_df.empty:
            print(f"\n  Medium verdicts with corrections: {len(medium_df)}")
            for _, row in medium_df.head(10).iterrows():
                print(f"    {row['sdrf_column']}: "
                      f"'{str(row['extracted_value'])[:35]}' "
                      f"corrected to '{str(row['corrected_value'])[:35]}'")

    cache_stats = _get_judge_model().get_cache_stats()
    print("\nCACHE STATISTICS")
    for k, v in cache_stats.items():
        print(f"  {k:<35} {v}")

    title = os.path.basename(os.path.normpath(out_dir))
    if USE_LLM_JUDGE:
        print("\nGenerating summary plots...")
        plot_results(df, per_paper_df, title, out_dir)
        plot_match_distribution(df, out_dir)
        plot_per_field(per_field_df, out_dir, title)

    if not per_paper_df.empty and "judge_accuracy" in per_paper_df.columns:
        print(f"\nLLM Judge mean accuracy: "
              f"{float(pd.to_numeric(per_paper_df['judge_accuracy'], errors='coerce').mean()):.1%}")

    print(f"\nFINAL SUMMARY  ({len(per_paper_df)} papers)")
    if not per_paper_df.empty and "judge_accuracy" in per_paper_df.columns:
        for _, stat in per_paper_df.iterrows():
            print(f"  {str(stat['paper_id']):<30} "
                  f"values={stat.get('total_extracted', 0)}  "
                  f"correct={stat.get('judge_n_correct', 0)}  "
                  f"halluc={stat.get('judge_n_hallucinated', 0)}  "
                  f"accuracy={stat.get('judge_accuracy', 0):.2f}")

    print("\nOutput files:")
    print(f"  Filtered SDRF : {os.path.join(out_dir, 'filtered_sdrf')}/<PXD>.sdrf.tsv")
    print(f"  Review CSV    : {os.path.join(out_dir, 'llm_judge_annotation_review.csv')}")
    print(f"  Review (full) : {os.path.join(out_dir, 'llm_judge_annotation_review_full.csv')}")
    print(f"  Per-paper CSV : {os.path.join(out_dir, 'llm_judge_per_paper.csv')}")
    print(f"  Per-field CSV : {os.path.join(out_dir, 'llm_judge_per_field.csv')}")
    print(f"  Coverage CSV  : {os.path.join(out_dir, 'llm_judge_coverage.csv')}")
    print(f"  Recovered CSV : {os.path.join(out_dir, 'llm_judge_recovered.csv')}")
    print(f"  JSON outputs  : {os.path.join(out_dir, 'json_outputs')}/<PXD_ID>.json")
    print(f"  Plots         : {out_dir}/llm_judge_*.png, match_distribution_*.png")
    print(f"\nAll outputs saved to: {out_dir}")
    print("\nPIPELINE COMPLETE.")


if __name__ == "__main__":
    main()