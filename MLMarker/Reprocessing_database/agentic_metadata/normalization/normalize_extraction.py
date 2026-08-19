"""
Bridge between the extraction JSON output and the ontology normalizer.

Walks sample_groups → fields → evidence items, normalizing each value
against the appropriate ontology. Adds ontology_id, ontology_name, and
similarity fields alongside the original value when a match is found.

Cell lines use exact matching (case/hyphen insensitive) instead of SapBERT,
because cell line codes are arbitrary and semantic similarity is meaningless.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

from .normalizer import TermNormalizer

logger = logging.getLogger(__name__)

# Fields normalized via SapBERT embedding search
EVIDENCE_ARRAY_FIELDS: Dict[str, str] = {
    "organism": "organism",
    "tissue": "tissue",
    "disease": "disease",
    "cell_part": "cell_part",
    "instrument": "instrument",
    "fragmentation": "fragmentation",
    "acquisition": "acquisition",
    "ionization": "ionization",
    "labeling": "labeling",
    "modifications": "modifications",
    "enzymes": "enzymes",
    "lc_column": "lc_column",
    "treatment_class": "treatment_class",
}

# Fields using exact matching (not SapBERT)
EXACT_MATCH_FIELDS = {"cell_line"}

# Special object fields — not normalized
METHOD_FIELDS: Dict[str, str] = {}


def _normalize_key(s: str) -> str:
    """Normalize a cell line name for exact matching: lowercase, strip hyphens/spaces."""
    return re.sub(r"[\s\-_()]", "", s).lower()


class CellLineExactMatcher:
    """Exact matcher for cell line names, bypassing SapBERT."""

    def __init__(self, vocab_path: Path):
        self.canonical: Dict[str, str] = {}  # normalized_key -> canonical_name
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if not term or term.startswith("#"):
                    continue
                key = _normalize_key(term)
                # Keep the first occurrence as canonical
                if key not in self.canonical:
                    self.canonical[key] = term

    def match(self, value: str) -> Optional[str]:
        """Return canonical name if exact match found, else None."""
        key = _normalize_key(value)
        return self.canonical.get(key)

def _clean_instrument(value: str) -> str:
    """Normalize instrument name variants before lookup.

    Handles patterns like:
      "Orbitrap Q Exactive HF"  → "Q Exactive HF"
      "Q-Exactive Orbitrap"     → "Q Exactive"
      "Thermo Orbitrap Fusion"  → "Orbitrap Fusion"
      "OrbitrapQ Exactive HF"   → "Q Exactive HF"
    """
    s = value.strip()
    s = re.sub(r"^Thermo\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"Q[\s-]*Exactive", "Q Exactive", s, flags=re.IGNORECASE)
    s = re.sub(r"Orbitrap\s*(?=Q Exactive)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=Q Exactive)\s*Orbitrap", "", s, flags=re.IGNORECASE)
    return s.strip()


def _load_post_normalization_map(ontology_dir: str, config: dict = None) -> Dict[str, Dict[str, str]]:
    """Load post-normalization forced mappings.

    Reads from config.yaml (normalization.post_normalization_map) if a config
    dict is provided, otherwise falls back to the standalone YAML file in
    ontology_dir for backward compatibility.

    Returns a dict of {entity_type: {lowercase_source: target_term}}.
    """
    raw = None

    # Prefer config dict (from config.yaml normalization section)
    if config is not None:
        raw = config.get("normalization", {}).get("post_normalization_map")

    # Fallback: standalone YAML file
    if raw is None:
        path = Path(ontology_dir) / "post_normalization_map.yaml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

    if not raw:
        return {}

    # Build case-insensitive lookup
    return {
        entity_type: {src.lower(): tgt for src, tgt in mappings.items()}
        for entity_type, mappings in raw.items()
        if isinstance(mappings, dict)
    }


def _apply_post_normalization(item: dict, entity_type: str,
                              post_map: Dict[str, Dict[str, str]]) -> dict:
    """Override ontology_name if a forced mapping exists for this term."""
    mappings = post_map.get(entity_type)
    if not mappings:
        return item
    ontology_name = (item.get("ontology_name") or "").lower()
    if ontology_name in mappings:
        original = item["ontology_name"]
        item["ontology_name"] = mappings[ontology_name]
        logger.debug("Post-normalization override: '%s' -> '%s'", original, item["ontology_name"])
    return item


def _normalize_item(item: dict, entity_type: str, normalizer: TermNormalizer) -> dict:
    """Normalize a single evidence-array item in-place via SapBERT."""
    value = item.get("value")
    if not value or not isinstance(value, str):
        return item

    query = value
    if entity_type == "instrument":
        query = _clean_instrument(value)

    result = normalizer.normalize(query, entity_type=entity_type)
    if result.is_normalized:
        item["ontology_id"] = result.ontology_id
        item["ontology_name"] = result.ontology_name
        item["similarity"] = round(result.similarity, 4)
    return item


def _exact_match_item(item: dict, matcher: CellLineExactMatcher) -> dict:
    """Normalize a cell line item via exact matching."""
    value = item.get("value")
    if not value or not isinstance(value, str):
        return item

    canonical = matcher.match(value)
    if canonical:
        item["ontology_name"] = canonical
        item["similarity"] = 1.0
    return item


def normalize_extraction(
    data: dict,
    normalizer: TermNormalizer,
    cell_line_matcher: CellLineExactMatcher = None,
    config: dict = None,
    tracker=None,
) -> dict:
    """
    Normalize all sample groups in an extraction result.

    Args:
        data: Parsed extraction JSON (has "sample_groups" key)
        normalizer: Initialized TermNormalizer with ontologies loaded
        cell_line_matcher: Exact matcher for cell lines (optional)
        config: Full config dict (from config.yaml); used to read
                post_normalization_map. Falls back to standalone YAML if None.
        tracker: Optional NormalizationTracker to record all mappings for reporting.

    Returns:
        The same dict with ontology metadata added to matched items.
    """
    post_map = _load_post_normalization_map(normalizer.config.ontology_dir, config=config)

    for group in data.get("sample_groups", []):
        # SapBERT-normalized fields
        for field_name, entity_type in EVIDENCE_ARRAY_FIELDS.items():
            items = group.get(field_name)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    raw_value = item.get("value", "")
                    _normalize_item(item, entity_type, normalizer)
                    pre_post = item.get("ontology_name", "")
                    _apply_post_normalization(item, entity_type, post_map)
                    post_post = item.get("ontology_name", "")

                    if tracker and raw_value:
                        sim = item.get("similarity", 0.0)
                        if not pre_post:
                            tracker.record(field_name, raw_value, "", sim, "unmapped")
                        elif post_post != pre_post:
                            tracker.record(field_name, raw_value, post_post, sim, "post_mapped")
                        else:
                            tracker.record(field_name, raw_value, post_post, sim, "mapped")

        # Exact-match fields (cell lines)
        if cell_line_matcher:
            items = group.get("cell_line")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        raw_value = item.get("value", "")
                        _exact_match_item(item, cell_line_matcher)
                        if tracker and raw_value:
                            matched = item.get("ontology_name", "")
                            if matched:
                                tracker.record("cell_line", raw_value, matched, 1.0, "exact_match")
                            else:
                                tracker.record("cell_line", raw_value, "", 0.0, "unmapped")

        # Special method fields (fractionation, enrichment)
        for field_name, entity_type in METHOD_FIELDS.items():
            obj = group.get(field_name)
            if not isinstance(obj, dict):
                continue
            method = obj.get("method")
            if method and isinstance(method, str):
                result = normalizer.normalize(method, entity_type=entity_type)
                if result.is_normalized:
                    obj["method_ontology_id"] = result.ontology_id
                    obj["method_ontology_name"] = result.ontology_name
                    obj["method_similarity"] = round(result.similarity, 4)

    return data
