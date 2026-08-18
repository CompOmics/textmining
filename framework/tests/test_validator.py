"""
Tests for the validation module.

Tests cover:
- Schema validation
- Evidence checking
- Confidence scoring
- Validation modes
"""

import pytest
from typing import Dict, Any


class TestFieldSchema:
    """Tests for FieldSchema validation."""
    
    def test_valid_format(self):
        """Test valid [value, evidence] format."""
        from validation.schema import FieldSchema
        
        schema = FieldSchema("species")
        is_valid, error = schema.validate_format(["human", "Human samples were used"])
        
        assert is_valid == True
        assert error == ""
    
    def test_invalid_not_list(self):
        """Test that non-list values are rejected."""
        from validation.schema import FieldSchema
        
        schema = FieldSchema("species")
        is_valid, error = schema.validate_format("human")
        
        assert is_valid == False
        assert "Expected list" in error
    
    def test_invalid_wrong_length(self):
        """Test that lists with wrong length are rejected."""
        from validation.schema import FieldSchema
        
        schema = FieldSchema("species")
        is_valid, error = schema.validate_format(["human"])
        
        assert is_valid == False
        assert "[value, evidence]" in error
    
    def test_unknown_with_empty_evidence(self):
        """Test that 'unknown' with empty evidence is valid."""
        from validation.schema import FieldSchema
        
        schema = FieldSchema("species")
        is_valid, error = schema.validate_format(["unknown", ""])
        
        assert is_valid == True
    
    def test_unknown_with_evidence_invalid(self):
        """Test that 'unknown' with evidence is invalid."""
        from validation.schema import FieldSchema
        
        schema = FieldSchema("species")
        is_valid, error = schema.validate_format(["unknown", "some evidence"])
        
        assert is_valid == False
        assert "empty evidence" in error


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from validation.schema import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Missing field: disease"],
            confidence=0.85,
            field_scores={"species": 1.0, "tissue": 0.5}
        )
        
        d = result.to_dict()
        
        assert d["is_valid"] == True
        assert d["confidence"] == 0.85
        assert "species" in d["field_scores"]


class TestConfidenceMetrics:
    """Tests for ConfidenceMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from validation.schema import ConfidenceMetrics
        
        metrics = ConfidenceMetrics(
            overall=0.82,
            evidence_score=0.9,
            completeness=0.7,
            format_score=1.0
        )
        
        d = metrics.to_dict()
        
        assert d["overall"] == 0.82
        assert d["evidence_score"] == 0.9


class TestValidationAgent:
    """Tests for ValidationAgent."""
    
    @pytest.fixture
    def sample_metadata(self) -> Dict[str, Any]:
        """Sample extraction metadata."""
        return {
            "species": ["human", "Human liver samples were collected"],
            "tissue": ["liver", "liver tissue analysis"],
            "cell_type": ["unknown", ""],
            "disease": ["cancer", "patients with cancer"],
        }
    
    @pytest.fixture
    def sample_text(self) -> str:
        """Sample source text."""
        return """
        Human liver samples were collected from patients with cancer.
        Liver tissue analysis revealed protein expression changes.
        MS analysis was performed on a Q Exactive instrument.
        """
    
    def test_validate_schema_valid(self, sample_metadata):
        """Test schema validation with valid metadata."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        result = agent.validate_schema(sample_metadata)
        
        assert result.is_valid == True
        assert len(result.errors) == 0
    
    def test_validate_schema_invalid_format(self):
        """Test schema catches format errors."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        bad_metadata = {
            "species": "human",  # Not a list!
            "tissue": ["liver"],  # Missing evidence
        }
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        result = agent.validate_schema(bad_metadata)
        
        assert result.is_valid == False
        assert len(result.errors) > 0
    
    def test_validate_evidence_present(self, sample_metadata, sample_text):
        """Test evidence validation when value is in text."""
        from validation.validator import ValidationAgent
        
        agent = ValidationAgent()
        scores = agent.validate_evidence(sample_metadata, sample_text)
        
        # "human" and "liver" should have high scores
        assert scores["species"] > 0.5
        assert scores["tissue"] > 0.5
    
    def test_calculate_confidence(self, sample_metadata, sample_text):
        """Test confidence calculation."""
        from validation.validator import ValidationAgent
        
        agent = ValidationAgent()
        confidence = agent.calculate_confidence(sample_metadata, sample_text)
        
        assert 0 <= confidence.overall <= 1
        assert 0 <= confidence.evidence_score <= 1
        assert 0 <= confidence.completeness <= 1
    
    def test_validate_adds_confidence(self, sample_metadata, sample_text):
        """Test that validate adds _confidence to output."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        result = agent.validate(sample_text, sample_metadata)
        
        assert "_confidence" in result
        assert "overall" in result["_confidence"]
    
    def test_fix_format_errors(self):
        """Test automatic format fixes."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        bad_metadata = {
            "species": "human",  # String instead of list
            "tissue": ["liver"],  # Missing evidence
        }
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        result = agent.validate("sample text with human and liver", bad_metadata)
        
        # Should be auto-fixed to proper format
        assert isinstance(result["species"], list)
        assert len(result["species"]) == 2
        assert isinstance(result["tissue"], list)
        assert len(result["tissue"]) == 2
    
    def test_is_high_quality(self, sample_metadata, sample_text):
        """Test high quality check."""
        from validation.validator import ValidationAgent
        
        agent = ValidationAgent()
        is_good = agent.is_high_quality(sample_metadata, sample_text, threshold=0.5)
        
        assert isinstance(is_good, bool)


class TestValidationModes:
    """Tests for different validation modes."""
    
    def test_schema_only_mode(self):
        """Test SCHEMA_ONLY mode doesn't call LLM."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        
        # LLM should not be loaded
        assert agent._llm is None
        
        result = agent.validate(
            "Sample text",
            {"species": ["human", "evidence"]}
        )
        
        # LLM should still not be loaded
        assert agent._llm is None
    
    def test_mode_override(self):
        """Test mode can be overridden per-call."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        agent = ValidationAgent(mode=ValidationMode.LLM_ONLY)
        
        # Override to schema only
        result = agent.validate(
            "Sample text",
            {"species": ["human", "evidence"]},
            mode=ValidationMode.SCHEMA_ONLY
        )
        
        # Should work without LLM
        assert "_confidence" in result


class TestDynamicValidation:
    """Tests for dynamic field validation."""
    
    def test_validates_dynamic_fields(self):
        """Test that all fields are validated, not just predefined ones."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        metadata = {
            "custom_field_1": ["value1", "evidence1"],
            "reduction concentration": ["5 mm", "concentration of 5 mm"],
            "another_field": ["value", "evidence"],
        }
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        result = agent.validate_schema(metadata)
        
        # All 3 fields should be validated
        assert len(result.field_scores) == 3
        assert result.is_valid == True
    
    def test_normalized_object_format(self):
        """Test validation of normalized object format."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        metadata = {
            "species": {
                "value": "Mus musculus",
                "evidence": "mouse samples",
                "ontology_id": "NCBITaxon:10090",
                "is_normalized": True
            },
            "cell_type": {
                "value": "myoblast",
                "evidence": "myoblast cells",
                "ontology_id": "CL:0000056",
            }
        }
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        result = agent.validate_schema(metadata)
        
        assert result.is_valid == True
        assert result.field_scores["species"] == 1.0
        assert result.field_scores["cell_type"] == 1.0
    
    def test_resolved_object_format(self):
        """Test validation of enriched/resolved format."""
        from validation.validator import ValidationAgent
        from validation.schema import ValidationMode
        
        metadata = {
            "instrument": {
                "resolved": "Q Exactive HF",
                "confidence": 1.0,
                "status": "AGREE",
            }
        }
        
        agent = ValidationAgent(mode=ValidationMode.SCHEMA_ONLY)
        result = agent.validate_schema(metadata)
        
        assert result.is_valid == True
        assert result.field_scores["instrument"] == 1.0
    
    def test_mixed_formats(self):
        """Test metadata with mixed list and object formats."""
        from validation.validator import ValidationAgent
        
        metadata = {
            "species": {"value": "human", "evidence": "human samples"},
            "tissue": ["liver", "liver tissue"],
            "cell_type": {"resolved": "epithelial cell", "confidence": 0.9},
        }
        
        agent = ValidationAgent()
        confidence = agent.calculate_confidence(metadata, "human liver tissue samples")
        
        assert confidence.format_score == 1.0  # All 3 fields valid
        assert confidence.completeness == 1.0  # All have non-unknown values
