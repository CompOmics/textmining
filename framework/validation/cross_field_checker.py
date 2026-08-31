"""
Cross-Field Ontology Consistency Checker
=========================================
Post-normalization check using CLO, CL, UBERON and DOID to detect
cross-field hallucinations in biological metadata records.

Four relationship checks are performed:

1. cell_line (CLO)  → derives_from → species  (NCBITaxon)   vs extracted species
2. cell_line (CLO)  → derives_from → tissue   (BTO/UBERON)  vs extracted tissue
3. disease   (DOID) → located_in  → tissue   (UBERON)       vs extracted tissue
4. cell_type (CL)   → part_of     → tissue   (UBERON)       vs extracted tissue

All ontology file access is lazy and gracefully degraded — if a file is
missing the corresponding check is silently skipped.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static NCBITaxon → accepted name sets  (covers the most common organisms)
# ---------------------------------------------------------------------------
_TAXON_NAMES: Dict[str, str] = {
    "NCBITaxon_9606":  "Homo sapiens",
    "NCBITaxon_10090": "Mus musculus",
    "NCBITaxon_10116": "Rattus norvegicus",
    "NCBITaxon_9544":  "Macaca mulatta",
    "NCBITaxon_9031":  "Gallus gallus",
    "NCBITaxon_7227":  "Drosophila melanogaster",
    "NCBITaxon_6239":  "Caenorhabditis elegans",
    "NCBITaxon_3702":  "Arabidopsis thaliana",
    "NCBITaxon_4932":  "Saccharomyces cerevisiae",
    "NCBITaxon_562":   "Escherichia coli",
    "NCBITaxon_9913":  "Bos taurus",
    "NCBITaxon_9823":  "Sus scrofa",
    "NCBITaxon_9615":  "Canis lupus familiaris",
    "NCBITaxon_9986":  "Oryctolagus cuniculus",
}

_TAXON_ALIASES: Dict[str, frozenset] = {
    "NCBITaxon_9606":  frozenset({"homo sapiens", "human", "h. sapiens", "h sapiens"}),
    "NCBITaxon_10090": frozenset({"mus musculus", "mouse", "m. musculus", "m musculus"}),
    "NCBITaxon_10116": frozenset({"rattus norvegicus", "rat", "r. norvegicus"}),
    "NCBITaxon_9544":  frozenset({"macaca mulatta", "rhesus macaque", "rhesus monkey"}),
    "NCBITaxon_9031":  frozenset({"gallus gallus", "chicken"}),
    "NCBITaxon_7227":  frozenset({"drosophila melanogaster", "fruit fly", "drosophila"}),
    "NCBITaxon_6239":  frozenset({"caenorhabditis elegans", "c. elegans", "worm"}),
    "NCBITaxon_3702":  frozenset({"arabidopsis thaliana", "thale cress", "arabidopsis"}),
    "NCBITaxon_4932":  frozenset({"saccharomyces cerevisiae", "yeast", "s. cerevisiae"}),
    "NCBITaxon_562":   frozenset({"escherichia coli", "e. coli"}),
    "NCBITaxon_9913":  frozenset({"bos taurus", "cow", "bovine", "cattle"}),
    "NCBITaxon_9823":  frozenset({"sus scrofa", "pig", "swine", "porcine"}),
    "NCBITaxon_9615":  frozenset({"canis lupus familiaris", "dog", "canine"}),
    "NCBITaxon_9986":  frozenset({"oryctolagus cuniculus", "rabbit"}),
}

# OWL namespace shortcuts
_OWL  = "http://www.w3.org/2002/07/owl#"
_RDF  = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"

# RO:0001000 = derives_from
_DERIVES_FROM_IRI = "http://purl.obolibrary.org/obo/RO_0001000"

_SKIP_VALUES = frozenset({"unknown", "n/a", "na", "none", ""})


class CrossFieldConsistencyChecker:
    """
    Post-normalization cross-field ontology consistency checker.

    Usage::

        checker = CrossFieldConsistencyChecker()
        flags   = checker.check(record)   # record is a metadata dict

    ``flags`` is a (possibly empty) list of dicts; each dict has the keys::

        type         – e.g. "cell_line_species_mismatch"
        field_a      – first field name
        value_a      – first field value
        ontology_id_a
        field_b      – second field name
        value_b      – second field value
        ontology_id_b
        expected_b   – expected value for field_b (from ontology)
        source       – which ontology relationship was used
    """

    def __init__(
        self,
        ontology_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        base = Path(__file__).resolve().parent.parent
        self._ontology_dir = Path(ontology_dir) if ontology_dir else base / "ontologies"
        self._cache_dir    = Path(cache_dir)    if cache_dir    else base / "ontology_cache"

        # Lazy-loaded caches (None = not yet attempted)
        self._clo_map:         Optional[Dict] = None
        self._clo_label_idx:   Optional[Dict] = None
        self._bto_uberon_map:  Optional[Dict] = None
        self._doid_tissue_map: Optional[Dict] = None
        self._cl_tissue_map:   Optional[Dict] = None

    # =========================================================================
    # Public API
    # =========================================================================

    def check(self, record: Dict) -> List[Dict]:
        """
        Run all cross-field consistency checks on *record*.

        Args:
            record: Extracted/normalised metadata dict.  Field values may be
                    ``["value", "evidence"]`` lists **or** ``{"value": ...,
                    "ontology_id": ...}`` dicts.

        Returns:
            List of flag dicts; empty if no inconsistencies detected.
        """
        flags: List[Dict] = []

        cell_line_val,  cell_line_id  = self._get_field(record, "cell_line")
        species_val,    species_id    = self._get_field(record, "species", "organism")
        tissue_val,     tissue_id     = self._get_field(record, "tissue")
        disease_val,    disease_id    = self._get_field(record, "disease", "disease_state")
        cell_type_val,  cell_type_id  = self._get_field(record, "cell_type")

        # ── CLO-based checks ─────────────────────────────────────────────────
        if cell_line_val and cell_line_val.lower() not in _SKIP_VALUES:
            resolved_clo = cell_line_id or self._lookup_clo_id_by_name(cell_line_val)
            if resolved_clo:
                entry = self._get_clo_entry(resolved_clo)
                if entry:
                    if species_val and species_val.lower() not in _SKIP_VALUES:
                        f = self._check_clo_species(
                            cell_line_val, resolved_clo, entry,
                            species_val, species_id,
                        )
                        if f:
                            flags.append(f)

                    if tissue_val and tissue_val.lower() not in _SKIP_VALUES:
                        f = self._check_clo_tissue(
                            cell_line_val, resolved_clo, entry,
                            tissue_val, tissue_id,
                        )
                        if f:
                            flags.append(f)

        # ── DOID → tissue ────────────────────────────────────────────────────
        if (disease_val and disease_val.lower() not in _SKIP_VALUES
                and tissue_val and tissue_val.lower() not in _SKIP_VALUES
                and disease_id):
            f = self._check_doid_tissue(disease_val, disease_id, tissue_val, tissue_id)
            if f:
                flags.append(f)

        # ── CL → tissue ──────────────────────────────────────────────────────
        if (cell_type_val and cell_type_val.lower() not in _SKIP_VALUES
                and tissue_val and tissue_val.lower() not in _SKIP_VALUES
                and cell_type_id):
            f = self._check_cl_tissue(cell_type_val, cell_type_id, tissue_val, tissue_id)
            if f:
                flags.append(f)

        return flags

    # =========================================================================
    # Field extraction helpers
    # =========================================================================

    @staticmethod
    def _get_field(record: Dict, *field_names: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Return *(value, ontology_id)* for the first matching field_name.

        Handles:
        * ``["value", "evidence"]`` list format
        * ``{"value": ..., "ontology_id": ...}`` dict format
        * ``{"resolved": ..., "ontology_id": ...}`` dict format
        * plain ``str``
        """
        for name in field_names:
            raw = record.get(name)
            if raw is None:
                continue

            if isinstance(raw, list):
                if raw and all(isinstance(item, dict) for item in raw):
                    first = next(
                        (
                            item for item in raw
                            if str(item.get("value", "")).strip().lower()
                            not in _SKIP_VALUES
                        ),
                        raw[0],
                    )
                    v = first.get("value") or first.get("resolved") or ""
                    oid = first.get("ontology_id") or first.get("id")
                    v = str(v).strip()
                    oid = str(oid).strip() if oid else None
                    return (v or None), oid
                v = str(raw[0]).strip() if raw else ""
                return (v or None), None

            if isinstance(raw, dict):
                v = (raw.get("value") or raw.get("resolved") or
                     raw.get("ontology_name") or "")
                oid = raw.get("ontology_id") or raw.get("id")
                v = str(v).strip()
                oid = str(oid).strip() if oid else None
                return (v or None), oid

            if isinstance(raw, str):
                v = raw.strip()
                return (v or None), None

        return None, None

    # =========================================================================
    # Check implementations
    # =========================================================================

    def _check_clo_species(
        self,
        cell_line_val: str,
        clo_id: str,
        clo_entry: Dict,
        species_val: str,
        species_id: Optional[str],
    ) -> Optional[Dict]:
        expected_ncbi = clo_entry.get("species_ncbi")
        if not expected_ncbi:
            return None

        if self._species_matches(species_val, expected_ncbi):
            return None

        return {
            "type": "cell_line_species_mismatch",
            "field_a": "cell_line", "value_a": cell_line_val, "ontology_id_a": clo_id,
            "field_b": "species",   "value_b": species_val,   "ontology_id_b": species_id,
            "expected_b": _TAXON_NAMES.get(expected_ncbi, expected_ncbi),
            "source": "CLO derives_from",
        }

    def _check_clo_tissue(
        self,
        cell_line_val: str,
        clo_id: str,
        clo_entry: Dict,
        tissue_val: str,
        tissue_id: Optional[str],
    ) -> Optional[Dict]:
        clo_bto    = clo_entry.get("tissue_bto")
        clo_uberon = clo_entry.get("tissue_uberon")

        if not clo_bto and not clo_uberon:
            return None

        # Build expected UBERON ID set
        expected: set = set()
        if clo_uberon:
            expected.add(self._norm_id(clo_uberon))
        if clo_bto:
            for uid in self._bto_to_uberon(clo_bto):
                expected.add(self._norm_id(uid))

        if not expected:
            return None

        # We can only compare reliably if the record carries an ontology ID
        if not tissue_id:
            return None

        if self._norm_id(tissue_id) in expected:
            return None

        best_expected = clo_uberon or clo_bto
        return {
            "type": "cell_line_tissue_mismatch",
            "field_a": "cell_line", "value_a": cell_line_val, "ontology_id_a": clo_id,
            "field_b": "tissue",    "value_b": tissue_val,    "ontology_id_b": tissue_id,
            "expected_b": best_expected,
            "source": "CLO derives_from",
        }

    def _check_doid_tissue(
        self,
        disease_val: str,
        disease_id: str,
        tissue_val: str,
        tissue_id: Optional[str],
    ) -> Optional[Dict]:
        doid_map = self._load_doid_tissue_map()
        norm_did = self._norm_id(disease_id)
        expected = [self._norm_id(u) for u in doid_map.get(norm_did, [])]

        if not expected:
            return None
        if not tissue_id:
            return None
        if self._norm_id(tissue_id) in expected:
            return None

        return {
            "type": "disease_tissue_mismatch",
            "field_a": "disease", "value_a": disease_val, "ontology_id_a": disease_id,
            "field_b": "tissue",  "value_b": tissue_val,  "ontology_id_b": tissue_id,
            "expected_b": ", ".join(doid_map.get(norm_did, [])),
            "source": "DOID located_in",
        }

    def _check_cl_tissue(
        self,
        cell_type_val: str,
        cl_id: str,
        tissue_val: str,
        tissue_id: Optional[str],
    ) -> Optional[Dict]:
        cl_map = self._load_cl_tissue_map()
        norm_cid = self._norm_id(cl_id)
        expected = [self._norm_id(u) for u in cl_map.get(norm_cid, [])]

        if not expected:
            return None
        if not tissue_id:
            return None
        if self._norm_id(tissue_id) in expected:
            return None

        return {
            "type": "cell_type_tissue_mismatch",
            "field_a": "cell_type", "value_a": cell_type_val, "ontology_id_a": cl_id,
            "field_b": "tissue",    "value_b": tissue_val,    "ontology_id_b": tissue_id,
            "expected_b": ", ".join(cl_map.get(norm_cid, [])),
            "source": "CL part_of",
        }

    # =========================================================================
    # Species / ID comparison helpers
    # =========================================================================

    @staticmethod
    def _norm_id(oid: str) -> str:
        """Normalise ontology ID to uppercase colon form: UBERON:0001234."""
        return oid.replace("_", ":").upper()

    @staticmethod
    def _species_matches(species_val: str, expected_ncbi: str) -> bool:
        """Return True if *species_val* is consistent with *expected_ncbi* taxon."""
        sv = species_val.lower().strip()
        if sv in _TAXON_ALIASES.get(expected_ncbi, frozenset()):
            return True
        canon = _TAXON_NAMES.get(expected_ncbi, "").lower()
        if canon and SequenceMatcher(None, sv, canon).ratio() >= 0.85:
            return True
        return False

    def _bto_to_uberon(self, bto_id: str) -> List[str]:
        bto_map = self._load_bto_uberon_map()
        return bto_map.get(self._norm_id(bto_id), [])

    # =========================================================================
    # CLO map loading
    # =========================================================================

    def _get_clo_entry(self, clo_id: str) -> Optional[Dict]:
        clo_map = self._load_clo_map()
        return clo_map.get(self._norm_id(clo_id))

    def _lookup_clo_id_by_name(self, name: str) -> Optional[str]:
        self._load_clo_map()  # ensure label index built
        if self._clo_label_idx is None:
            return None
        return self._clo_label_idx.get(name.lower().strip())

    def _load_clo_map(self) -> Dict:
        if self._clo_map is not None:
            return self._clo_map

        cache_path = self._cache_dir / "clo_derives_from.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    self._clo_map = json.load(f)
                self._build_clo_label_idx()
                logger.info(
                    "Loaded CLO derives_from cache from %s (%d entries)",
                    cache_path, len(self._clo_map),
                )
                return self._clo_map
            except Exception as exc:
                logger.warning("Failed to read CLO cache %s: %s", cache_path, exc)

        clo_path = self._ontology_dir / "clo.owl"
        if not clo_path.exists():
            logger.warning(
                "clo.owl not found at %s — CLO consistency checks disabled", clo_path
            )
            self._clo_map = {}
            self._clo_label_idx = {}
            return self._clo_map

        logger.info("Parsing CLO OWL from %s …", clo_path)
        self._clo_map = self._parse_clo_owl(str(clo_path))

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self._clo_map, f, indent=2)
            logger.info("Wrote CLO cache to %s", cache_path)
        except Exception as exc:
            logger.warning("Could not write CLO cache: %s", exc)

        self._build_clo_label_idx()
        return self._clo_map

    def _build_clo_label_idx(self) -> None:
        if not self._clo_map:
            self._clo_label_idx = {}
            return
        self._clo_label_idx = {
            entry["label"].lower(): clo_id
            for clo_id, entry in self._clo_map.items()
            if entry.get("label")
        }

    @staticmethod
    def _parse_clo_owl(path: str) -> Dict:
        """
        Parse CLO OWL/XML to extract derives_from relationships.

        Returns
        -------
        dict mapping (normalised) CLO ID → {label, species_ncbi, tissue_bto,
        tissue_uberon}
        """
        result: Dict = {}
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            logger.error("Failed to parse CLO OWL %s: %s", path, exc)
            return result

        root = tree.getroot()

        for cls in root.iter(f"{{{_OWL}}}Class"):
            about = cls.get(f"{{{_RDF}}}about", "")
            if "CLO_" not in about:
                continue

            # e.g. "CLO_0000148"  →  "CLO:0000148"
            local  = about.rsplit("/", 1)[-1]
            clo_id = local.replace("_", ":", 1)  # only first underscore

            label_el = cls.find(f"{{{_RDFS}}}label")
            label    = (label_el.text or "").strip() if label_el is not None else ""

            entry: Dict = {
                "label":         label,
                "species_ncbi":  None,
                "tissue_bto":    None,
                "tissue_uberon": None,
            }

            # Walk all subClassOf/Restriction blocks within this Class element
            for restriction in cls.iter(f"{{{_OWL}}}Restriction"):
                prop_el = restriction.find(f"{{{_OWL}}}onProperty")
                if prop_el is None:
                    continue
                prop_iri = prop_el.get(f"{{{_RDF}}}resource", "")
                if "RO_0001000" not in prop_iri:   # derives_from
                    continue

                filler_el  = restriction.find(f"{{{_OWL}}}someValuesFrom")
                if filler_el is None:
                    continue
                filler_iri = filler_el.get(f"{{{_RDF}}}resource", "")
                if not filler_iri:
                    continue

                local_filler = filler_iri.rsplit("/", 1)[-1]
                if "NCBITaxon_" in local_filler:
                    entry["species_ncbi"] = local_filler          # keep underscore form
                elif "BTO_" in local_filler or local_filler.startswith("BTO:"):
                    entry["tissue_bto"] = local_filler.replace("_", ":", 1)
                elif "UBERON_" in local_filler:
                    entry["tissue_uberon"] = local_filler.replace("_", ":", 1)

            if label or entry["species_ncbi"] or entry["tissue_bto"] or entry["tissue_uberon"]:
                result[clo_id] = entry

        logger.info("Parsed %d CLO entries from OWL", len(result))
        return result

    # =========================================================================
    # BTO → UBERON map loading
    # =========================================================================

    def _load_bto_uberon_map(self) -> Dict:
        if self._bto_uberon_map is not None:
            return self._bto_uberon_map

        bto_path = self._ontology_dir / "bto.obo"
        if not bto_path.exists():
            logger.warning(
                "bto.obo not found at %s — BTO→UBERON mapping disabled", bto_path
            )
            self._bto_uberon_map = {}
            return self._bto_uberon_map

        self._bto_uberon_map = self._parse_obo_xrefs(str(bto_path), xref_prefix="UBERON:")
        logger.info("Built BTO→UBERON map: %d entries", len(self._bto_uberon_map))
        return self._bto_uberon_map

    # =========================================================================
    # DOID → tissue map loading
    # =========================================================================

    def _load_doid_tissue_map(self) -> Dict:
        if self._doid_tissue_map is not None:
            return self._doid_tissue_map

        cache_path = self._cache_dir / "doid_tissue.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    self._doid_tissue_map = json.load(f)
                return self._doid_tissue_map
            except Exception:
                pass

        doid_path = self._ontology_dir / "doid.obo"
        if not doid_path.exists():
            logger.warning(
                "doid.obo not found at %s — DOID tissue checks disabled", doid_path
            )
            self._doid_tissue_map = {}
            return self._doid_tissue_map

        # Parse both xref: UBERON: and relationship: located_in UBERON:
        xref_map = self._parse_obo_xrefs(str(doid_path), xref_prefix="UBERON:")
        rel_map  = self._parse_obo_relationship(str(doid_path), "located_in", "UBERON:")

        # Merge: union of IDs per disease
        merged: Dict[str, List[str]] = {}
        for did, ids in {**xref_map, **rel_map}.items():
            existing = set(merged.get(did, []))
            existing.update(ids)
            merged[did] = list(existing)

        self._doid_tissue_map = merged
        logger.info("Built DOID tissue map: %d entries", len(self._doid_tissue_map))

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self._doid_tissue_map, f, indent=2)
        except Exception:
            pass

        return self._doid_tissue_map

    # =========================================================================
    # CL → tissue map loading
    # =========================================================================

    def _load_cl_tissue_map(self) -> Dict:
        if self._cl_tissue_map is not None:
            return self._cl_tissue_map

        cache_path = self._cache_dir / "cl_tissue.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    self._cl_tissue_map = json.load(f)
                return self._cl_tissue_map
            except Exception:
                pass

        cl_path = self._ontology_dir / "cl.obo"
        if not cl_path.exists():
            logger.warning(
                "cl.obo not found at %s — CL tissue checks disabled", cl_path
            )
            self._cl_tissue_map = {}
            return self._cl_tissue_map

        self._cl_tissue_map = self._parse_obo_relationship(
            str(cl_path), "part_of", "UBERON:"
        )
        logger.info("Built CL tissue map: %d entries", len(self._cl_tissue_map))

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self._cl_tissue_map, f, indent=2)
        except Exception:
            pass

        return self._cl_tissue_map

    # =========================================================================
    # OBO file parsers (static)
    # =========================================================================

    @staticmethod
    def _parse_obo_xrefs(obo_path: str, xref_prefix: str) -> Dict[str, List[str]]:
        """
        Parse an OBO file and return {term_id: [xref_id, ...]} for every
        ``xref:`` line whose value starts with *xref_prefix*.
        """
        result: Dict[str, List[str]] = {}
        current_id: Optional[str] = None
        in_term = False

        with open(obo_path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if line.startswith("["):
                    in_term = line == "[Term]"
                    current_id = None
                elif in_term:
                    if line.startswith("id:") and current_id is None:
                        current_id = line[3:].strip()
                    elif line.startswith("xref:") and current_id:
                        # strip inline comments after the ID
                        xref_val = line[5:].strip().split()[0]
                        if xref_val.startswith(xref_prefix):
                            result.setdefault(current_id, []).append(xref_val)

        return result

    @staticmethod
    def _parse_obo_relationship(
        obo_path: str,
        relation: str,
        target_prefix: str,
    ) -> Dict[str, List[str]]:
        """
        Parse an OBO file and return {term_id: [target_id, ...]} for every
        ``relationship: <relation> <target_id>`` line where target_id starts
        with *target_prefix*.
        """
        result: Dict[str, List[str]] = {}
        current_id: Optional[str] = None
        in_term = False

        # Pre-compile a pattern like: relationship: part_of UBERON:\d+
        prefix_bare = target_prefix.rstrip(":")
        pat = re.compile(
            r"^relationship:\s+" + re.escape(relation)
            + r"\s+(" + re.escape(prefix_bare) + r"[:\d_]+)"
        )

        with open(obo_path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if line.startswith("["):
                    in_term = line == "[Term]"
                    current_id = None
                elif in_term:
                    if line.startswith("id:") and current_id is None:
                        current_id = line[3:].strip()
                    elif current_id:
                        m = pat.match(line)
                        if m:
                            # strip inline comments
                            target_id = m.group(1).split("!")[0].strip()
                            result.setdefault(current_id, []).append(target_id)

        return result
