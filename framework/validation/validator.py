"""
Validation agent for extraction quality assurance.

Provides schema validation, evidence checking, and confidence scoring
to ensure extraction quality without always requiring LLM calls.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Tuple

from core.llm import LLMClient
from core.logging import get_logger
from .schema import (
    ValidationMode, FieldSchema, ValidationResult, ConfidenceMetrics,
    ALL_FIELDS, BIOLOGICAL_FIELDS, TECHNICAL_FIELDS
)

logger = get_logger(__name__)


# ============================================================================
# LLM Validation Prompt
# ============================================================================

VALIDATION_PROMPT = """You are a QA Critic for scientific metadata extraction.
Your job is to verify that the extracted JSON metadata matches the Source Text and follows the Strict Rules.

SOURCE TEXT:
{text}

EXTRACTED JSON:
{json_data}

STRICT RULES TO ENFORCE:
1. All values must be EXPLICITLY in the text (exact substring match). Do not allow abbreviation expansions.
2. If not in text, value must be "unknown".
3. No hallucinations or inferred values that aren't supported by text.
4. "unknown" values must have empty evidence string "".
5. Each value must be a list: [value_string, evidence_string]. If it is not a list, FIX IT.

TASK:
- Review each field in the EXTRACTED JSON.
- If a value violates the rules (e.g. it is not in the text, or should be "unknown"), CORRECT it.
- If the extraction is perfect, return it as is.
- Return ONLY the final corrected JSON.

CORRECTED JSON:"""


class ValidationAgent:
    """
    Validates and scores extraction quality.
    
    Supports three modes:
    - SCHEMA_ONLY: Fast rule-based validation (no LLM)
    - LLM_ONLY: LLM-based validation (original behavior)
    - HYBRID: Schema first, LLM for low-confidence cases
    """
    
    def __init__(self, mode: ValidationMode = ValidationMode.SCHEMA_ONLY):
        """
        Initialize the validation agent.
        
        Args:
            mode: Validation strategy to use
        """
        self.mode = mode
        self._llm: Optional[LLMClient] = None
    
    @property
    def llm(self) -> LLMClient:
        """Lazy-load LLM client only when needed."""
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm
    
    # ========================================================================
    # Main Validation Methods
    # ========================================================================
    
    def validate(self, text: str, metadata: dict, 
                 mode: Optional[ValidationMode] = None) -> dict:
        """
        Validate and optionally correct extracted metadata.
        
        Args:
            text: Source document text
            metadata: Extracted metadata dict
            mode: Override default validation mode
            
        Returns:
            Validated (and possibly corrected) metadata
        """
        mode = mode or self.mode
        
        logger.info("Validating extraction (mode=%s)", mode.value)
        
        if mode == ValidationMode.SCHEMA_ONLY:
            result = self.validate_schema(metadata)
            corrected = self._fix_format_errors(metadata, result)
            return self._add_confidence(corrected, text)
        
        elif mode == ValidationMode.LLM_ONLY:
            return self._validate_with_llm(text, metadata)
        
        elif mode == ValidationMode.HYBRID:
            # First do schema validation
            result = self.validate_schema(metadata)
            confidence = self.calculate_confidence(metadata, text)
            
            # If confidence is low, use LLM
            if confidence.overall < 0.7 or not result.is_valid:
                logger.info("Low confidence (%.2f), using LLM validation", 
                           confidence.overall)
                return self._validate_with_llm(text, metadata)
            
            corrected = self._fix_format_errors(metadata, result)
            return self._add_confidence(corrected, text)
        
        return metadata
    
    def validate_schema(self, metadata: dict) -> ValidationResult:
        """
        Validate metadata against format rules (dynamic - checks all fields).
        
        Args:
            metadata: Extracted metadata dict
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        field_scores = {}
        
        # Dynamically validate ALL fields in metadata
        for field_name, value in metadata.items():
            # Skip internal metadata fields
            if field_name.startswith('_'):
                continue
            
            # Check format
            is_valid, error = self._validate_field_format(field_name, value)
            
            if not is_valid:
                errors.append(error)
                field_scores[field_name] = 0.0
            else:
                field_scores[field_name] = 1.0
        
        # Also check for missing predefined fields (as warnings only)
        for field_name, schema in ALL_FIELDS.items():
            if field_name not in metadata and schema.required:
                warnings.append(f"Missing field: {field_name}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            field_scores=field_scores,
            confidence=sum(field_scores.values()) / max(len(field_scores), 1)
        )
    
    def _validate_field_format(self, field_name: str, value: Any) -> tuple:
        """
        Validate a single field's format. Accepts:
        - List format: [value, evidence]
        - Normalized object: {value, evidence, ontology_id, ...}
        - Resolved object: {resolved, confidence, status, sources, ...}
        
        Returns:
            (is_valid, error_message)
        """
        if value is None:
            return False, f"{field_name}: Value is None"
        
        # Accept normalized/enriched object format
        if isinstance(value, dict):
            # Normalized format: {value, evidence, ontology_id, ...}
            if 'value' in value or 'resolved' in value:
                val = value.get('value') or value.get('resolved')
                evidence = value.get('evidence', '')
                
                # Check for unknown with evidence
                if val == "unknown" and evidence:
                    return False, f"{field_name}: 'unknown' values must have empty evidence"
                
                return True, ""
            else:
                return False, f"{field_name}: Dict missing 'value' or 'resolved' key"
        
        # Accept list format: [value, evidence]
        if isinstance(value, list):
            if len(value) != 2:
                return False, f"{field_name}: Expected [value, evidence], got {len(value)} elements"
            
            val, evidence = value
            
            if not isinstance(val, str):
                return False, f"{field_name}: Value should be string"
            
            if not isinstance(evidence, str):
                return False, f"{field_name}: Evidence should be string"
            
            # Unknown with evidence is invalid
            if val == "unknown" and evidence:
                return False, f"{field_name}: 'unknown' values must have empty evidence"
            
            return True, ""
        
        # String alone is not valid format
        if isinstance(value, str):
            return False, f"{field_name}: Expected [value, evidence] or object, got string"
        
        return False, f"{field_name}: Unknown format {type(value).__name__}"
    
    def validate_evidence(self, metadata: dict, text: str) -> Dict[str, float]:
        """
        Check if evidence strings actually contain extracted values.
        Handles both list format and normalized object format.
        Uses fuzzy matching for abbreviations (e.g. 'P. falciparum' vs 'Plasmodium falciparum').
        
        Args:
            metadata: Extracted metadata
            text: Source document text
            
        Returns:
            Dict of field_name -> evidence_score (0-1)
        """
        scores = {}
        text_lower = text.lower()
        
        for field_name, value in metadata.items():
            if field_name.startswith('_'):  # Skip metadata fields
                continue
            
            # Extract val and evidence from different formats
            val, evidence = self._extract_value_evidence(value)
            
            # Also check for resolved value (for enriched data)
            resolved = None
            if isinstance(value, dict) and 'resolved' in value:
                resolved = value.get('resolved')
            
            if val is None:
                scores[field_name] = 0.0
                continue
            
            # Unknown values with empty evidence are valid
            if val == "unknown" and not evidence:
                scores[field_name] = 1.0
                continue
            
            score = 0.0
            
            # Check 1: Value appears in source text (exact or fuzzy)
            if self._text_contains(val, text_lower):
                score += 0.5
            elif self._fuzzy_match_text(val, text_lower) > 0.7:
                score += 0.4  # Slightly lower for fuzzy match
            
            # Check 2: Value appears in evidence (exact or fuzzy)
            if evidence and isinstance(val, str) and isinstance(evidence, str):
                if val.lower() in evidence.lower():
                    score += 0.3
                elif self._fuzzy_match(val, evidence) > 0.6:
                    score += 0.25
            
            # Check 3: Evidence appears in source text
            if evidence and isinstance(evidence, str) and evidence.lower() in text_lower:
                score += 0.2
            
            # Fallback: If evidence is empty but value is grounded in source text,
            # the extraction is objectively correct — don't penalize missing quote.
            # Source text grounding is a stronger signal than an LLM-generated quote.
            if not evidence and score >= 0.4:
                score = max(score, 0.7)
            
            # Bonus: If resolved matches LLM value (fuzzy)
            if resolved and val:
                match_score = self._fuzzy_match(val, resolved)
                if match_score > 0.8:
                    score += 0.1  # Bonus for agreement
            
            scores[field_name] = min(score, 1.0)
        
        return scores
    
    def get_critique(self, metadata: dict, text: str) -> dict:
        """
        Generate structured validation critique for re-extraction feedback.
        
        Returns a dict with:
          - overall_confidence: float
          - issues: list of human-readable issue strings
          - field_issues: dict of field_name -> list of issues
          
        Used by the extractor to build a refinement prompt when
        confidence is below threshold.
        """
        issues = []
        field_issues = {}
        text_lower = text.lower()
        
        # Schema validation
        schema_result = self.validate_schema(metadata)
        for error in schema_result.errors:
            issues.append(f"Format error: {error}")
        
        # Evidence checking per field
        for field_name, value in metadata.items():
            if field_name.startswith('_'):
                continue
            
            field_problems = []
            val, evidence = self._extract_value_evidence(value)
            
            if val is None or val == "unknown":
                continue
            
            # Check: empty evidence
            if not evidence:
                field_problems.append("no supporting evidence quote provided")
            
            # Check: value not grounded in source text
            if isinstance(val, str) and not self._text_contains(val, text_lower):
                if not (self._fuzzy_match_text(val, text_lower) > 0.7):
                    field_problems.append(
                        f"value '{val}' not found in source text"
                    )
            
            # Check: evidence not in source text (possible fabrication)
            if evidence and isinstance(evidence, str):
                if evidence.lower() not in text_lower:
                    field_problems.append(
                        "evidence quote not found in source text"
                    )
            
            if field_problems:
                field_issues[field_name] = field_problems
                for p in field_problems:
                    issues.append(f"Field '{field_name}': {p}")
        
        # Calculate confidence
        confidence = self.calculate_confidence(metadata, text)
        
        return {
            "overall_confidence": confidence.overall,
            "issues": issues,
            "field_issues": field_issues,
            "n_issues": len(issues),
        }
    
    def _text_contains(self, needle: str, haystack: str) -> bool:
        """Check if needle is in haystack (case-insensitive)."""
        if not isinstance(needle, str) or not isinstance(haystack, str):
            return False
        return needle.lower() in haystack.lower()
    
    def _fuzzy_match(self, s1: str, s2: str) -> float:
        """
        Fuzzy string similarity using SequenceMatcher.
        Handles abbreviations like 'P. falciparum' vs 'Plasmodium falciparum'.
        
        Returns:
            Similarity ratio (0-1)
        """
        if not s1 or not s2:
            return 0.0
        
        # Ensure both are strings
        if not isinstance(s1, str) or not isinstance(s2, str):
            return 0.0
        
        s1_lower = s1.lower().strip()
        s2_lower = s2.lower().strip()
        
        # Exact match
        if s1_lower == s2_lower:
            return 1.0
        
        # Check for abbreviation pattern: "X. name" vs "Xname name"
        abbrev_score = self._check_abbreviation(s1_lower, s2_lower)
        if abbrev_score > 0.8:
            return abbrev_score
        
        # Standard fuzzy match
        return SequenceMatcher(None, s1_lower, s2_lower).ratio()
    
    def _check_abbreviation(self, short: str, long: str) -> float:
        """
        Check if short is an abbreviation of long.
        E.g., 'P. falciparum' matches 'Plasmodium falciparum'
        """
        # Pattern: "X. rest" where X is first letter
        abbrev_pattern = re.match(r'^([a-z])\. (.+)$', short)
        if abbrev_pattern:
            first_letter = abbrev_pattern.group(1)
            rest = abbrev_pattern.group(2)
            
            # Check if long starts with that letter and contains rest
            parts = long.split()
            if parts and parts[0].startswith(first_letter):
                long_rest = ' '.join(parts[1:])
                if rest == long_rest or SequenceMatcher(None, rest, long_rest).ratio() > 0.9:
                    return 0.95
        
        # Also check reverse (long vs short)
        abbrev_pattern = re.match(r'^([a-z])\. (.+)$', long)
        if abbrev_pattern:
            first_letter = abbrev_pattern.group(1)
            rest = abbrev_pattern.group(2)
            
            parts = short.split()
            if parts and parts[0].startswith(first_letter):
                short_rest = ' '.join(parts[1:])
                if rest == short_rest or SequenceMatcher(None, rest, short_rest).ratio() > 0.9:
                    return 0.95
        
        return 0.0
    
    def _fuzzy_match_text(self, needle: str, text: str) -> float:
        """
        Find best fuzzy match of needle anywhere in text.
        Returns best match score.
        """
        if not needle or not text:
            return 0.0
        
        # Ensure both are strings
        if not isinstance(needle, str) or not isinstance(text, str):
            return 0.0
        
        needle_lower = needle.lower()
        text_lower = text.lower()
        
        # Quick exact check
        if needle_lower in text_lower:
            return 1.0
        
        # Check abbreviation patterns in text
        # Look for "X. word" pattern matching needle
        words = text_lower.split()
        for i, word in enumerate(words):
            if '.' in word and i + 1 < len(words):
                potential_abbrev = word + ' ' + words[i + 1]
                score = self._check_abbreviation(potential_abbrev, needle_lower)
                if score > 0.8:
                    return score
                score = self._check_abbreviation(needle_lower, potential_abbrev)
                if score > 0.8:
                    return score
        
        return 0.0
    
    def _extract_value_evidence(self, value: Any) -> tuple:
        """
        Extract (value, evidence) from various formats.
        
        Returns:
            (value_str, evidence_str) or (None, None) if invalid
        """
        if isinstance(value, list) and len(value) == 2:
            return value[0], value[1]
        
        if isinstance(value, dict):
            val = value.get('value') or value.get('resolved')
            evidence = value.get('evidence', '')
            return val, evidence
        
        return None, None
    
    def calculate_confidence(self, metadata: dict, text: str) -> ConfidenceMetrics:
        """
        Calculate confidence metrics for an extraction.
        
        Args:
            metadata: Extracted metadata
            text: Source document text
            
        Returns:
            ConfidenceMetrics object
        """
        # Schema validation
        schema_result = self.validate_schema(metadata)
        format_score = schema_result.confidence
        
        # Evidence checking
        evidence_scores = self.validate_evidence(metadata, text)
        evidence_score = (sum(evidence_scores.values()) / 
                         max(len(evidence_scores), 1))
        
        # Completeness (non-unknown fields)
        known_fields = 0
        total_fields = 0
        for field_name, value in metadata.items():
            if field_name.startswith('_'):
                continue
            total_fields += 1
            
            val, _ = self._extract_value_evidence(value)
            if val and val != "unknown":
                known_fields += 1
        
        completeness = known_fields / max(total_fields, 1)
        
        # Overall confidence (weighted average)
        overall = (
            format_score * 0.3 +
            evidence_score * 0.5 +
            completeness * 0.2
        )
        
        return ConfidenceMetrics(
            overall=overall,
            evidence_score=evidence_score,
            completeness=completeness,
            format_score=format_score,
        )
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _validate_with_llm(self, text: str, metadata: dict) -> dict:
        """Use LLM to validate and correct metadata."""
        json_str = json.dumps(metadata, indent=2)
        prompt = VALIDATION_PROMPT.format(text=text, json_data=json_str)
        
        logger.debug("Calling LLM for validation...")
        
        corrected_json_str = self.llm.get_completion(
            [{"role": "user", "content": prompt}], 
            temperature=0.0
        )
        
        try:
            # Extract JSON from response
            corrected = self._extract_json(corrected_json_str)
            return self._add_confidence(corrected, text)
        except json.JSONDecodeError:
            logger.warning("Validator returned invalid JSON, keeping original")
            return self._add_confidence(metadata, text)
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from LLM response."""
        # Find JSON object using brace matching
        start_idx = text.find('{')
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON found", text, 0)
        
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start=start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    
    def _fix_format_errors(self, metadata: dict, 
                           result: ValidationResult) -> dict:
        """Auto-fix common format errors."""
        fixed = {}
        
        for field_name, value in metadata.items():
            if field_name.startswith('_'):
                fixed[field_name] = value
                continue
            
            # Fix: Not a list → wrap in list with empty evidence
            if not isinstance(value, list):
                if isinstance(value, str):
                    fixed[field_name] = [value, ""]
                    logger.debug("Fixed %s: wrapped string in list", field_name)
                else:
                    fixed[field_name] = ["unknown", ""]
                continue
            
            # Fix: Single element list → add empty evidence
            if len(value) == 1:
                fixed[field_name] = [value[0], ""]
                logger.debug("Fixed %s: added empty evidence", field_name)
                continue
            
            # Fix: Nested list [[val, ev]] → [val, ev]
            if len(value) == 1 and isinstance(value[0], list):
                fixed[field_name] = value[0]
                logger.debug("Fixed %s: flattened nested list", field_name)
                continue
            
            fixed[field_name] = value
        
        return fixed
    
    def _add_confidence(self, metadata: dict, text: str) -> dict:
        """Add confidence metrics to metadata."""
        confidence = self.calculate_confidence(metadata, text)
        metadata['_confidence'] = confidence.to_dict()
        return metadata
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def get_confidence(self, metadata: dict, text: str) -> float:
        """Get overall confidence score for an extraction."""
        return self.calculate_confidence(metadata, text).overall
    
    def is_high_quality(self, metadata: dict, text: str, 
                        threshold: float = 0.7) -> bool:
        """Check if extraction meets quality threshold."""
        return self.get_confidence(metadata, text) >= threshold
