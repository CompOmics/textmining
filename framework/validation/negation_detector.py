"""
Negation Detector
=================
Uses negspacy (NegEx algorithm) to detect when an extracted value is negated
in its evidence sentence — e.g., "no TMT labeling was used" while the extractor
returned TMT as the labeling method.

The detector is designed to be:
- Lazy: the spaCy pipeline is only loaded on first use (heavy model load).
- Graceful: if spaCy/negspacy is not installed, the detector silently no-ops.
- Evidence-aware: skips ``inferred: ...`` evidence and ``unknown`` values.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy singleton — shared across the process lifetime so the model is
# only loaded once (en_core_web_sm + NegEx pipe).
_NLP: Any = None
_NLP_LOAD_FAILED: bool = False


def _load_nlp() -> Any:
    """Lazy-load the spaCy+negex pipeline (once per process)."""
    global _NLP, _NLP_LOAD_FAILED

    if _NLP is not None:
        return _NLP
    if _NLP_LOAD_FAILED:
        return None

    try:
        import spacy  # noqa: F401
        import negspacy.negation  # noqa: F401 — registers @Language.factory("negex")
        _NLP = spacy.load("en_core_web_sm")
        _NLP.add_pipe("negex")
        logger.info("NegationDetector: loaded spaCy + negex pipeline")
    except Exception as exc:
        logger.warning(
            "NegationDetector: could not load spaCy/negex — negation checks disabled. "
            "Install with: pip install spacy negspacy && python -m spacy download en_core_web_sm  "
            "(%s)", exc,
        )
        _NLP_LOAD_FAILED = True
        _NLP = None

    return _NLP



# ---------------------------------------------------------------------------
# Fields we should NOT run negation detection on.
# These are expected to describe absence / non-existence by design, so a
# negated evidence sentence is correct rather than suspicious.
# ---------------------------------------------------------------------------
_SKIP_FIELDS = frozenset({
    "unknown",          # sentinel value itself
    "label-free",       # "no labels used" IS the positive evidence
    "none",
    "n/a",
})

_INFERRED_PREFIX = "inferred: "


def _is_inferred(evidence: str) -> bool:
    return isinstance(evidence, str) and evidence.lower().startswith(_INFERRED_PREFIX)


def _extract_val_evidence(value: Any):
    """Return (value_str, evidence_str) from list or dict format."""
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, dict):
        v = value.get("value") or value.get("resolved", "")
        e = value.get("evidence", "")
        return v, e
    return None, None


class NegationDetector:
    """
    Detect negated evidence using NegEx via negspacy.

    Usage::

        detector = NegationDetector()
        flags = detector.check(record)   # record is a metadata dict

    Each flag is a dict::

        {
            "type":     "negated_evidence",
            "field":    "labeling",
            "value":    "TMT",
            "evidence": "No TMT labeling was used in this study."
        }
    """

    def check(self, record: dict) -> list[dict]:
        """
        Scan every field in *record* for negated evidence.

        Returns
        -------
        list[dict]
            Flags for any field where the evidence sentence negates the
            extracted value.  Empty list if nothing problematic found or
            if the detector is unavailable.
        """
        nlp = _load_nlp()
        if nlp is None:
            return []

        flags: list[dict] = []

        for field_name, field_value in record.items():
            if field_name.startswith("_"):
                continue

            val, evidence = _extract_val_evidence(field_value)
            if not val or not evidence:
                continue

            val_lower = val.strip().lower()

            # Skip sentinel values and inferred evidence
            if val_lower in _SKIP_FIELDS:
                continue
            if val_lower == "unknown":
                continue
            if _is_inferred(evidence):
                continue
            if not isinstance(evidence, str) or not isinstance(val, str):
                continue

            negated = self._is_negated_in_sentence(nlp, val, evidence)
            if negated:
                flags.append({
                    "type":     "negated_evidence",
                    "field":    field_name,
                    "value":    val,
                    "evidence": evidence,
                })

        return flags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_negated_in_sentence(nlp: Any, value: str, evidence: str) -> bool:
        """
        Return True if NegEx marks *value* as negated within *evidence*.

        Strategy
        --------
        1. Run the full spaCy pipeline on the evidence sentence — this gives
           the dependency parse (needed by NegEx for sentence boundaries).
        2. Inject a named-entity span over the value's character range.
        3. Re-run only the negex component on the already-parsed doc.
        4. Return True if the entity's ``_.negex`` attribute is True.

        If the value doesn't appear literally in the evidence sentence we
        return False — nothing to check.
        """
        evidence_lower = evidence.lower()
        val_lower = value.lower()

        start = evidence_lower.find(val_lower)
        if start == -1:
            return False  # value not literally in evidence sentence

        end = start + len(value)

        try:
            # Full parse (tokenisation + dep parser for sentence boundaries)
            doc = nlp(evidence)

            # Inject the entity span covering the value
            span = doc.char_span(start, end, label="ENTITY",
                                 alignment_mode="expand")
            if span is None:
                return False

            doc.set_ents([span])

            # Run only the negex component (dep parse already applied above)
            negex_pipe = nlp.get_pipe("negex")
            doc = negex_pipe(doc)

            for ent in doc.ents:
                return bool(ent._.negex)

        except Exception as exc:
            logger.debug("NegationDetector._is_negated_in_sentence error: %s", exc)

        return False
