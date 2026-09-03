"""
Numeric Mismatch Detector
=========================
Flags fields where a numeric value (concentration, energy, percentage, etc.)
does not appear verbatim in its evidence sentence.

Motivation
----------
Fuzzy string matching cannot distinguish "5 mM" from "50 mM" — their
SequenceMatcher ratio is ~0.8.  For numeric values there is no tolerance:
if the extracted number is not an exact substring of the evidence, the
extraction is wrong or the evidence is fabricated.

Output flag type: ``"numeric_mismatch"``
"""

from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Same pattern as ValidationAgent._NUMERIC_RE
_NUMERIC_RE = re.compile(
    r'^[+\-]?\d+\.?\d*\s*'
    r'(?:mm|\xb5m|\u03bcm|nm|mg/ml|\xb5g/ml|ng/ml|mg|\xb5g|ng'
    r'|%|nce|ev|kv|kva|amu|da|kda|rpm'
    r'|(?:ms|min|h|x|m|n|p|u|k|g)(?=$|\s|,|;|\.))',
    re.IGNORECASE,
)

_INFERRED_PREFIX = "inferred: "


def _is_numeric(val: str) -> bool:
    return bool(_NUMERIC_RE.match(val.strip()))


def _is_inferred(evidence: str) -> bool:
    return isinstance(evidence, str) and evidence.lower().startswith(_INFERRED_PREFIX)


def _extract_val_evidence(value: Any):
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, dict):
        v = value.get("value") or value.get("resolved", "")
        e = value.get("evidence", "")
        return v, e
    return None, None


class NumericMismatchDetector:
    """
    Flag fields where a numeric value is not verbatim in its evidence.

    Usage::

        detector = NumericMismatchDetector()
        flags = detector.check(record)

    Each flag::

        {
            "type":     "numeric_mismatch",
            "field":    "alkylation concentration",
            "value":    "50 mM",
            "evidence": "alkylated in 5 mM iodoacetamide for 45 min"
        }
    """

    def check(self, record: dict) -> list[dict]:
        flags: list[dict] = []

        for field_name, field_value in record.items():
            if field_name.startswith("_"):
                continue

            val, evidence = _extract_val_evidence(field_value)
            if not val or not evidence:
                continue
            if val.strip().lower() in ("unknown", ""):
                continue
            if _is_inferred(evidence):
                continue
            if not isinstance(val, str) or not isinstance(evidence, str):
                continue
            if not _is_numeric(val):
                continue

            # Exact substring check — no fuzzy tolerance for numerics
            if val.lower() not in evidence.lower():
                flags.append({
                    "type":     "numeric_mismatch",
                    "field":    field_name,
                    "value":    val,
                    "evidence": evidence,
                })

        return flags
