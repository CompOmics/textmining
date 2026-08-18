"""
Schema definitions for extraction validation.

Defines the expected structure and rules for extracted metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum


class ValidationMode(Enum):
    """Validation strategy modes."""
    SCHEMA_ONLY = "schema"      # Fast, rule-based validation only
    LLM_ONLY = "llm"            # LLM-based validation (original behavior)
    HYBRID = "hybrid"           # Schema first, LLM for uncertain cases


@dataclass
class FieldSchema:
    """Schema for a single extraction field."""
    name: str
    required: bool = True
    evidence_required: bool = True
    controlled_vocabulary: Optional[Set[str]] = None
    
    def validate_format(self, value: Any) -> tuple[bool, str]:
        """
        Validate value format.
        
        Expected format: [value_string, evidence_string]
        
        Returns:
            (is_valid, error_message)
        """
        if value is None:
            return False, f"{self.name}: Value is None"
        
        if not isinstance(value, list):
            return False, f"{self.name}: Expected list, got {type(value).__name__}"
        
        if len(value) != 2:
            return False, f"{self.name}: Expected [value, evidence], got {len(value)} elements"
        
        val, evidence = value
        
        if not isinstance(val, str):
            return False, f"{self.name}: Value should be string, got {type(val).__name__}"
        
        if not isinstance(evidence, str):
            return False, f"{self.name}: Evidence should be string, got {type(evidence).__name__}"
        
        # If value is "unknown", evidence must be empty
        if val == "unknown" and evidence:
            return False, f"{self.name}: 'unknown' values must have empty evidence"
        
        # If value is not "unknown", evidence is required
        if self.evidence_required and val != "unknown" and not evidence:
            return False, f"{self.name}: Non-unknown values require evidence"
        
        return True, ""


# ============================================================================
# Extraction Schema Definition
# ============================================================================

# Biological fields
BIOLOGICAL_FIELDS = {
    "species": FieldSchema("species"),
    "tissue": FieldSchema("tissue"),
    "cell_type": FieldSchema("cell_type"),
    "cell_line": FieldSchema("cell_line"),
    "disease": FieldSchema("disease"),
    "subcellular_location": FieldSchema("subcellular_location"),
}

# Technical/MS fields
TECHNICAL_FIELDS = {
    "instrument": FieldSchema("instrument"),
    "fragmentation": FieldSchema("fragmentation"),
    "labelling": FieldSchema("labelling"),
    "protease": FieldSchema("protease"),
    "enrichment": FieldSchema("enrichment"),
}

# Experimental fields
EXPERIMENTAL_FIELDS = {
    "experiment_type": FieldSchema("experiment_type"),
    "quantification": FieldSchema("quantification"),
    "sample_prep": FieldSchema("sample_prep"),
}

# All fields combined
ALL_FIELDS: Dict[str, FieldSchema] = {
    **BIOLOGICAL_FIELDS,
    **TECHNICAL_FIELDS,
    **EXPERIMENTAL_FIELDS,
}


@dataclass
class ValidationResult:
    """Result of validating an extraction."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    field_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "confidence": round(self.confidence, 3),
            "field_scores": {k: round(v, 3) for k, v in self.field_scores.items()},
        }


@dataclass 
class ConfidenceMetrics:
    """Confidence metrics for an extraction."""
    overall: float = 0.0
    evidence_score: float = 0.0
    completeness: float = 0.0
    format_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "overall": round(self.overall, 3),
            "evidence_score": round(self.evidence_score, 3),
            "completeness": round(self.completeness, 3),
            "format_score": round(self.format_score, 3),
        }
