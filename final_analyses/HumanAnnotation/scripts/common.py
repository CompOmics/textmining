"""Loading and agreement statistics for the human-annotation analysis.

Reads exclusively from EuBIC-annotation/ at the repository root: SingleHuman,
MultiHuman/<annotator>/, HarmonizedHuman, and the three model sets under
s0_model_iaa_27/. No PRIDE census, no external crosswalk table.

Computation only. All drawing lives in build_figure1.py.
"""
import itertools
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent          # final_analyses/HumanAnnotation/scripts
ANALYSIS = HERE.parent                          # final_analyses/HumanAnnotation
REPO = ANALYSIS.parents[1]                      # repository root
# All source annotations live in EuBIC-annotation/ at the repository root.
# Nothing in this analysis reads data from anywhere else.
ANN_ROOT = REPO / "EuBIC-annotation"
RESULTS = ANALYSIS / "results"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SEMANTIC_THRESHOLD = 0.70  # same threshold the project uses in framework/normalization

# -----------------------------------------------------------------------------
# Self-contained label -> {biological, technical, experimental_design}
# assignment, built directly from the 68 raw labels observed in
# EuBIC-annotation (SingleHuman/GPT/HarmonizedHuman/MultiHuman), not from any
# external crosswalk file.
# -----------------------------------------------------------------------------
LABEL_CATEGORY = {
    # biological
    "Organism": "biological", "OrganismPart": "biological", "Disease": "biological",
    "CellLine": "biological", "CellType": "biological", "Strain": "biological",
    "Sex": "biological", "Age": "biological", "DevelopmentalStage": "biological",
    "MaterialType": "biological", "CellPart": "biological", "Specimen": "biological",
    "TumorStage": "biological", "TumorSite": "biological", "TumorGrade": "biological",
    "TumorCellularity": "biological", "AnatomicSiteTumor": "biological",
    "OriginSiteDisease": "biological", "GeneticModification": "biological",
    "GrowthRate": "biological", "Bait": "biological", "SourceName": "biological",
    "DiseaseTreatment": "biological", "SpikedCompound": "biological",
    "Compound": "biological", "ConcentrationOfCompound": "biological",
    "PooledSample": "biological", "Staining": "biological", "Depletion": "biological",
    "Treatment": "biological",
    # technical
    "Modification": "technical", "Label": "technical", "AcquisitionMethod": "technical",
    "CleavageAgent": "technical", "Separation": "technical", "Instrument": "technical",
    "FragmentationMethod": "technical", "ReductionReagent": "technical",
    "PrecursorMassTolerance": "technical", "AlkylationReagent": "technical",
    "GradientTime": "technical", "FlowRateChromatogram": "technical",
    "Temperature": "technical", "Chromatography": "technical",
    "FractionationMethod": "technical", "NumberOfMissedCleavages": "technical",
    "MS2MassAnalyzer": "technical", "IonizationType": "technical",
    "FragmentMassTolerance": "technical", "CollisionEnergy": "technical",
    "EnrichmentMethod": "technical", "SupplementaryFile": "technical",
    "AssayName": "technical", "SyntheticPeptide": "technical",
    # experimental design
    "FactorValue": "experimental_design", "SampleTreatment": "experimental_design",
    "NumberOfSamples": "experimental_design", "NumberOfBiologicalReplicates": "experimental_design",
    "NumberOfTechnicalReplicates": "experimental_design", "BiologicalReplicate": "experimental_design",
    "TechnicalReplicate": "experimental_design", "Experiment": "experimental_design",
    "TechnologyType": "experimental_design", "Time": "experimental_design",
    "SamplingTime": "experimental_design", "NumberOfFractions": "experimental_design",
    "FractionIdentifier": "experimental_design", "FractionationFraction": "experimental_design",
}

# Minimum number of the 27 papers (by HarmonizedHuman consensus) a label must
# be mentioned in to be included at all. Below this, kappa and value-agreement
# numbers are noise (e.g. n=1 instance) rather than a signal -- Bait,
# Depletion, GeneticModification, OriginSiteDisease etc. are excluded by this.
MIN_MENTIONS = 3
# Outline's documented HarmonizedHuman consensus rule includes "exclusion of
# entity types assigned by only one annotator" -- enforce that here too, not
# just the mention-count filter above. A label tagged by <2 unique MultiHuman
# identities across the ENTIRE corpus can't have been a real multi-annotator
# consensus, so it's dropped from every downstream analysis (including
# HarmonizedHuman/SingleHuman/GPT presence and value agreement).
MIN_UNIQUE_ANNOTATORS = 2
CAT_COLORS = {"biological": "#3b6fa0", "technical": "#d9822b", "experimental_design": "#3f8f5f"}
CAT_ORDER = ["biological", "technical", "experimental_design"]
CAT_LABEL = {"biological": "Biological", "technical": "Technical", "experimental_design": "Experimental design"}

# -----------------------------------------------------------------------------
# 1. Load raw labels + values directly from EuBIC-annotation
# -----------------------------------------------------------------------------
def brat_entries(path: Path):
    """{label: [value_text, ...]} from a real BRAT T-line file."""
    out = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        label = parts[1].split()[0]
        out[label].append(parts[2].strip())
    return out


def flat_entries(path: Path):
    """{label: [value_text, ...]} from a 'Label : value' flat .ann file
    (GPT/ and HarmonizedHuman/ use this, not real BRAT spans)."""
    out = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        label, val = line.split(":", 1)
        label, val = label.strip(), val.strip()
        if label and val:
            out[label].append(val)
    return out


# The three model annotation sets arrived on 2026-08-31 in
# EuBIC-annotation/s0_model_iaa_27/. They differ from the human sets in two ways
# that have to be handled rather than assumed away:
#   1. their T-lines are delimited by "|||" and not by tabs, so the standard
#      BRAT reader returns nothing at all rather than failing loudly;
#   2. they are keyed by PXD accession, while every human set is keyed by PMID.
MODEL_DIRS = {
    "Qwen": "qwen3_8_27b",
    "Gemma": "gemma4_31b",
    "GLM": "glm4_7",
}
MODEL_ROOT_NAME = "s0_model_iaa_27/final_annotations"

# Identities loaded from the corpus but excluded from the analysis.
# GPT is the original EuBIC-era model annotation. It is superseded by the three
# current models above, its exact version was never recorded, and keeping a
# stale model alongside three pinned ones invites the question "which GPT?"
# while adding nothing. The .ann files stay in EuBIC-annotation/GPT/; drop the
# name from this set to bring it back.
EXCLUDE_IDENTITIES = {"GPT"}


def pipe_entries(path: Path):
    """{label: [value_text, ...]} from a '|||'-delimited T-line file."""
    out = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("|||")
        if len(parts) < 3:
            continue
        label = parts[1].split()[0]
        value = parts[2].strip()
        if label and value:
            out[label].append(value)
    return out


def pxd_to_pmid():
    """PXD accession -> PMID, from the committed mapping table.

    Built by matching manuscript text between the human .txt files and
    pmc_abstract_methods_nosubheadings/, then verified one-to-one across all 27.
    """
    import csv as _csv
    path = ANN_ROOT / "pxd_pmid_mapping.csv"
    with open(path) as f:
        return {r["pxd"]: r["pmid"] for r in _csv.DictReader(f)}


def load_models():
    """{model_name: {pmid: {label: [values]}}} for the three model sets."""
    root = ANN_ROOT / MODEL_ROOT_NAME
    if not root.exists():
        return {}
    xmap = pxd_to_pmid()
    data = defaultdict(dict)
    for name, dirname in MODEL_DIRS.items():
        for p in sorted((root / dirname / "ann_files").glob("*.ann")):
            pmid = xmap.get(p.stem)
            if pmid is None:          # unmapped accession, skip rather than guess
                continue
            data[name][pmid] = pipe_entries(p)
    return data


def load_annotators():
    """{annotator_name: {doc_id: {label: [values]}}}.
    'Ian' (MultiHuman/Ian/) is dropped: verified byte-identical to
    SingleHuman, would double-count one rater as two.
    """
    data = defaultdict(dict)
    for p in sorted((ANN_ROOT / "SingleHuman").glob("*.ann")):
        data["SingleHuman"][p.stem] = brat_entries(p)
    for p in sorted((ANN_ROOT / "GPT").glob("*.ann")):
        data["GPT"][p.stem] = flat_entries(p)
    for p in sorted((ANN_ROOT / "HarmonizedHuman").glob("*.ann")):
        data["HarmonizedHuman"][p.stem.removesuffix("_harmonized")] = flat_entries(p)
    for p in sorted(ANN_ROOT.glob("MultiHuman/*/*/*.ann")):
        annotator = p.parts[-3]
        if annotator.lower() == "ian":
            continue
        data[annotator][p.stem] = brat_entries(p)
    for name, docs in load_models().items():
        data[name] = docs
    return data


def anonymize(annotator_docs):
    special = {"SingleHuman", "GPT", "HarmonizedHuman"} | set(MODEL_DIRS)
    real_names = sorted(n for n in annotator_docs if n not in special)
    id_of = {real: f"Annotator{i + 1}" for i, real in enumerate(real_names)}
    renamed = {id_of.get(k, k): v for k, v in annotator_docs.items()}
    mapping = {anon: real for real, anon in id_of.items()}
    return renamed, mapping


def pair_category(a, b):
    """Classify an annotator pair.

    The model roles matter: without them Qwen, Gemma and GLM fall through to
    "multi" and are silently counted as human annotators, which would
    contaminate every human-human figure with model agreement.
    """
    def role(n):
        if n == "GPT":
            return "gpt"
        if n in MODEL_DIRS:
            return "model"
        if n == "HarmonizedHuman":
            return "harmonized"
        if n == "SingleHuman":
            return "single"
        return "multi"
    ra, rb = role(a), role(b)
    roles = {ra, rb}
    models = {"gpt", "model"}
    if roles <= models:
        return "model vs. model"
    if roles & models:
        # keep GPT's historical category so earlier tables stay comparable
        if "gpt" in roles:
            return "GPT vs. any human annotator"
        return "model vs. any human annotator"
    if "harmonized" in roles:
        return "HarmonizedHuman vs. any human annotator"
    if "single" in roles:
        return "SingleHuman vs. MultiHuman annotator"
    return "MultiHuman annotator vs. MultiHuman annotator"


# -----------------------------------------------------------------------------
# 2. Presence/absence Cohen's kappa
# -----------------------------------------------------------------------------
def build_presence(annotator_docs, labels):
    presence = {}
    for annotator, docs in annotator_docs.items():
        presence[annotator] = {
            doc_id: {lab: int(bool(entries.get(lab))) for lab in labels}
            for doc_id, entries in docs.items()
        }
    return presence


def cohen_kappa(a, b):
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    p0 = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (p0 - pe) / (1 - pe)


def shared_docs(store, a, b):
    return sorted(set(store.get(a, {})) & set(store.get(b, {})))


def pooled_kappa(presence, a, b, labels):
    docs = shared_docs(presence, a, b)
    av, bv = [], []
    for d in docs:
        for lab in labels:
            av.append(presence[a][d][lab])
            bv.append(presence[b][d][lab])
    return cohen_kappa(av, bv), len(docs)


def per_label_kappa(presence, a, b, label):
    docs = shared_docs(presence, a, b)
    av = [presence[a][d][label] for d in docs]
    bv = [presence[b][d][label] for d in docs]
    return cohen_kappa(av, bv), len(docs)


# -----------------------------------------------------------------------------
# 3. SapBERT value-level agreement
# -----------------------------------------------------------------------------
class SapBertScorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
        self.model = AutoModel.from_pretrained(SAPBERT_MODEL)
        self.model.eval()
        self._cache = {}

    def embed(self, texts, batch_size=64):
        uncached = [t for t in texts if t not in self._cache]
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i + batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
            cls = torch.nn.functional.normalize(out[:, 0, :], dim=1)
            for t, vec in zip(batch, cls):
                self._cache[t] = vec
        return torch.stack([self._cache[t] for t in texts])

    def max_similarity(self, values_a, values_b):
        if not values_a or not values_b:
            return None
        ea = self.embed(values_a)
        eb = self.embed(values_b)
        sims = ea @ eb.T
        return float(sims.max())


def value_agreement(annotator_docs, presence, labels, scorer: SapBertScorer):
    annotators = sorted(annotator_docs)
    rows = []
    for a, b in itertools.combinations(annotators, 2):
        docs = shared_docs(presence, a, b)
        for d in docs:
            for lab in labels:
                va = annotator_docs[a][d].get(lab, [])
                vb = annotator_docs[b][d].get(lab, [])
                if not va or not vb:
                    continue  # not a both-tagged case; presence kappa covers this
                exact = bool({v.strip().lower() for v in va} & {v.strip().lower() for v in vb})
                sim = None if exact else scorer.max_similarity(va, vb)
                semantic = (sim is not None) and (sim >= SEMANTIC_THRESHOLD)
                rows.append({
                    "label": lab, "category": LABEL_CATEGORY[lab],
                    "annotator_a": a, "annotator_b": b, "doc": d,
                    "exact_match": exact, "semantic_match": semantic,
                    "similarity": sim,
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. Value-category Cohen's kappa: exact string identity vs. SapBERT-cluster
# identity. This generalizes the presence/absence kappa above: each item is
# still a (doc, label), but each rater's "category" is now the ACTUAL VALUE
# they extracted (or ABSENT), not just whether they tagged the label at all.
# Two annotators who both tagged a label but wrote different, unrelated
# values now correctly count as a disagreement, which plain presence-kappa
# could not distinguish from real agreement.
# -----------------------------------------------------------------------------
ABSENT = "\0ABSENT"


def normalize_value(v: str) -> str:
    return " ".join(v.strip().lower().split())


def canonical_string_category(values):
    if not values:
        return ABSENT
    return "; ".join(sorted({normalize_value(v) for v in values}))


def cluster_label_values(annotator_docs, label, scorer: "SapBertScorer", threshold=SEMANTIC_THRESHOLD):
    """Union-find clustering of every distinct normalized value string used
    for this label (across all annotators/docs), using SapBERT cosine
    similarity >= threshold as the edge condition. Returns {value_str: cluster_id}."""
    distinct = set()
    for docs in annotator_docs.values():
        for entries in docs.values():
            for v in entries.get(label, []):
                distinct.add(normalize_value(v))
    distinct = sorted(distinct)
    if not distinct:
        return {}
    if len(distinct) == 1:
        return {distinct[0]: 0}

    embs = scorer.embed(distinct)
    sims = (embs @ embs.T).numpy()
    parent = list(range(len(distinct)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if sims[i, j] >= threshold:
                union(i, j)

    roots = {find(i) for i in range(len(distinct))}
    root_to_id = {r: cid for cid, r in enumerate(sorted(roots))}
    return {distinct[i]: root_to_id[find(i)] for i in range(len(distinct))}


def canonical_sapbert_category(values, cluster_map):
    if not values:
        return ABSENT
    cluster_ids = {cluster_map[normalize_value(v)] for v in values}
    return "cl:" + ",".join(str(c) for c in sorted(cluster_ids))


def value_categories(annotator_docs, labels, scheme, scorer=None):
    """{label: {annotator: {doc: category}}} under a given scheme
    ('string' or 'sapbert')."""
    cluster_maps = {}
    if scheme == "sapbert":
        for lab in labels:
            cluster_maps[lab] = cluster_label_values(annotator_docs, lab, scorer)

    out = {}
    for lab in labels:
        per_annotator = {}
        for annotator, docs in annotator_docs.items():
            cats = {}
            for doc, entries in docs.items():
                vals = entries.get(lab, [])
                if scheme == "string":
                    cats[doc] = canonical_string_category(vals)
                else:
                    cats[doc] = canonical_sapbert_category(vals, cluster_maps[lab])
            per_annotator[annotator] = cats
        out[lab] = per_annotator
    return out


def multicat_kappa(cats_a: dict, cats_b: dict):
    """Cohen's kappa over paired categorical judgments (any number of
    categories, not just binary)."""
    docs = sorted(set(cats_a) & set(cats_b))
    if not docs:
        return float("nan"), 0
    a = [cats_a[d] for d in docs]
    b = [cats_b[d] for d in docs]
    n = len(a)
    p0 = sum(1 for x, y in zip(a, b) if x == y) / n
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    all_cats = set(ca) | set(cb)
    pe = sum((ca.get(c, 0) / n) * (cb.get(c, 0) / n) for c in all_cats)
    if abs(1 - pe) < 1e-12:
        return float("nan"), n
    return (p0 - pe) / (1 - pe), n


def pooled_multicat_kappa(value_cats: dict, labels, a, b):
    """Pool every (label, doc) pair sharing both annotators into one kappa."""
    pooled_a, pooled_b = {}, {}
    for lab in labels:
        cats = value_cats[lab]
        if a not in cats or b not in cats:
            continue
        docs = sorted(set(cats[a]) & set(cats[b]))
        for d in docs:
            key = (lab, d)
            pooled_a[key] = cats[a][d]
            pooled_b[key] = cats[b][d]
    return multicat_kappa(pooled_a, pooled_b)


# -----------------------------------------------------------------------------
# 5. Decomposition of presence agreement: is high model-model agreement just
# both models saying "absent"?
#
# Presence kappa is scored over every (document, label) slot, and most slots
# are empty for most raters, so two sparse annotators agree on a great many
# absences. Kappa is meant to subtract exactly that, but kappa is known to
# misbehave when the marginals are skewed, so "kappa corrects for it" is an
# assumption to test rather than a defence to offer.
# -----------------------------------------------------------------------------
def identity_group(a, b):
    """'human-human', 'model-human' or 'model-model' for one pair."""
    am, bm = a in MODEL_DIRS, b in MODEL_DIRS
    if am and bm:
        return "model-model"
    return "model-human" if (am or bm) else "human-human"


def decompose_pair(presence, labels, a, b):
    """Full 2x2 table for one annotator pair, plus the three kappa-free
    diagnostics computed from it.

      chance_agreement    pe. If model-model kappa were inflated by shared
                          sparsity, pe would be HIGHER for the model pairs.
      both_absent_share   n00/n, the direct measure of the worry.
      positive_agreement  2*n11 / (2*n11 + n10 + n01), the Dice/F1 form. The
                          both-absent cell does not appear in it, so it cannot
                          be inflated by shared silence at all. If the result
                          survives here it is real.

    Also returns each rater's prevalence, the share of slots they mark present,
    because that is where the mechanism turns out to be visible.
    """
    n11 = n10 = n01 = n00 = 0
    for doc in shared_docs(presence, a, b):
        for lab in labels:
            x = bool(presence[a][doc].get(lab, False))
            y = bool(presence[b][doc].get(lab, False))
            if x and y:
                n11 += 1
            elif x:
                n10 += 1
            elif y:
                n01 += 1
            else:
                n00 += 1
    n = n11 + n10 + n01 + n00
    if not n:
        return None
    p0 = (n11 + n00) / n
    pa, pb = (n11 + n10) / n, (n11 + n01) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    denom = 2 * n11 + n10 + n01
    return {
        "group": identity_group(a, b), "annotator_a": a, "annotator_b": b,
        "kappa": (p0 - pe) / (1 - pe) if pe < 1 else float("nan"),
        "raw_agreement": p0,
        "chance_agreement": pe,
        "positive_agreement": (2 * n11 / denom) if denom else float("nan"),
        "both_absent_share": n00 / n,
        "prevalence_a": pa, "prevalence_b": pb,
        "n_slots": n,
    }


def prevalence(presence, labels, rater):
    """Share of all (document, label) slots this rater marks present."""
    docs = presence[rater]
    tot = len(docs) * len(labels)
    pos = sum(1 for d in docs for lab in labels if docs[d].get(lab, False))
    return pos / tot if tot else float("nan")


# -----------------------------------------------------------------------------
# Shared entry point: load annotators, anonymize, and apply both label
# filters (mention-count + single-annotator-exclusion). Both scripts call
# this so they always operate on the identical label/annotator set.
# -----------------------------------------------------------------------------
def load_filtered_data():
    annotator_docs = load_annotators()
    annotator_docs = {k: v for k, v in annotator_docs.items()
                      if k not in EXCLUDE_IDENTITIES}
    annotator_docs, name_mapping = anonymize(annotator_docs)
    special = {"SingleHuman", "GPT", "HarmonizedHuman"} | set(MODEL_DIRS)
    multihuman = set(annotator_docs) - special
    models_present = [m for m in MODEL_DIRS if m in annotator_docs]
    annotators = ([n for n in ("SingleHuman", "GPT", "HarmonizedHuman")
                   if n in annotator_docs]
                  + sorted(multihuman, key=lambda n: int(n.removeprefix("Annotator")))
                  + models_present)

    all_labels = sorted(LABEL_CATEGORY)
    harmonized_docs = annotator_docs["HarmonizedHuman"]
    n_docs_total = len(harmonized_docs)
    mention_counts = {lab: sum(1 for entries in harmonized_docs.values() if entries.get(lab)) for lab in all_labels}

    unique_annotator_counts = {}
    for lab in all_labels:
        unique_annotator_counts[lab] = sum(
            1 for a in multihuman if any(entries.get(lab) for entries in annotator_docs[a].values())
        )

    labels = [lab for lab in all_labels
              if mention_counts[lab] >= MIN_MENTIONS and unique_annotator_counts[lab] >= MIN_UNIQUE_ANNOTATORS]
    dropped_low_mentions = [lab for lab in all_labels if mention_counts[lab] < MIN_MENTIONS]
    dropped_single_annotator = [lab for lab in all_labels
                                  if mention_counts[lab] >= MIN_MENTIONS and unique_annotator_counts[lab] < MIN_UNIQUE_ANNOTATORS]

    return {
        "annotator_docs": annotator_docs, "name_mapping": name_mapping,
        "multihuman": multihuman, "annotators": annotators, "labels": labels,
        "all_labels": all_labels, "mention_counts": mention_counts,
        "unique_annotator_counts": unique_annotator_counts,
        "dropped_low_mentions": dropped_low_mentions,
        "dropped_single_annotator": dropped_single_annotator,
        "n_docs_total": n_docs_total,
    }
