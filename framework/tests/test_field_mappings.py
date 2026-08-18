"""
Tests for core.field_mappings module.

Validates that the canonical field name mappings are consistent
and cover all expected aliases.
"""

import pytest
from core.field_mappings import (
    LLM_TO_GOLDEN,
    resolve_field_name,
    SDRF_BIOLOGICAL,
    SDRF_TECHNICAL,
    SDRF_EXPERIMENTAL,
    SDRF_TO_GOLDEN,
    ANNOTATION_TO_GOLDEN,
    FIELD_TO_ENTITY_TYPE,
    FIELD_ONTOLOGY_MAP,
    AGENT_FIELDS,
    ALL_AGENT_FIELDS,
    METADATA_ONLY_FIELDS,
)


class TestResolveFieldName:
    """Tests for resolve_field_name()."""

    def test_exact_match(self):
        """Known LLM keys resolve to golden names."""
        assert resolve_field_name("tissue") == "organ"
        assert resolve_field_name("organism") == "species"
        assert resolve_field_name("disease_state") == "disease"

    def test_already_canonical(self):
        """Canonical golden names pass through unchanged."""
        assert resolve_field_name("species") == "species"
        assert resolve_field_name("organ") == "organ"
        assert resolve_field_name("disease") == "disease"

    def test_case_insensitive_fallback(self):
        """Case-insensitive matching works for unknown casing."""
        assert resolve_field_name("BMI") == "BMI"
        assert resolve_field_name("bmi") == "BMI"

    def test_unknown_name_passthrough(self):
        """Unknown field names pass through unchanged."""
        assert resolve_field_name("some_custom_field") == "some_custom_field"

    def test_technical_aliases(self):
        """Technical field aliases resolve correctly."""
        assert resolve_field_name("labeling") == "label"
        assert resolve_field_name("labelling") == "label"
        assert resolve_field_name("fragmentation_method") == "fragmentation"
        assert resolve_field_name("ms2massanalyzer") == "mass_analyzer"

    def test_experimental_aliases(self):
        """Experimental design aliases resolve correctly."""
        assert resolve_field_name("biological_replicate") == "replicates"
        assert resolve_field_name("number_of_biological_replicates") == "replicates"
        assert resolve_field_name("number_of_fractions") == "fractions"


class TestLLMToGolden:
    """Tests for LLM_TO_GOLDEN mapping."""

    def test_all_values_are_strings(self):
        """All mapped values should be strings."""
        for key, value in LLM_TO_GOLDEN.items():
            assert isinstance(value, str), f"Key {key} maps to non-string: {value}"

    def test_all_keys_are_strings(self):
        """All keys should be strings."""
        for key in LLM_TO_GOLDEN:
            assert isinstance(key, str), f"Non-string key: {key}"

    def test_bidirectional_mappings_consistent(self):
        """tissue->organ and organ->organ should be consistent."""
        assert LLM_TO_GOLDEN["tissue"] == LLM_TO_GOLDEN["organ"]
        assert LLM_TO_GOLDEN["organism"] == LLM_TO_GOLDEN["species"]
        assert LLM_TO_GOLDEN["disease_state"] == LLM_TO_GOLDEN["disease"]

    def test_minimum_size(self):
        """Should have a substantial number of mappings."""
        assert len(LLM_TO_GOLDEN) >= 50


class TestSDRFMappings:
    """Tests for SDRF column mappings."""

    def test_biological_mapping_has_species(self):
        assert "organism" in SDRF_BIOLOGICAL
        assert SDRF_BIOLOGICAL["organism"] == "species"

    def test_technical_mapping_has_instrument(self):
        assert "instrument" in SDRF_TECHNICAL
        assert SDRF_TECHNICAL["instrument"] == "instrument"

    def test_combined_is_superset(self):
        """SDRF_TO_GOLDEN should contain all individual mappings."""
        for key, value in SDRF_BIOLOGICAL.items():
            assert SDRF_TO_GOLDEN[key] == value
        for key, value in SDRF_TECHNICAL.items():
            assert SDRF_TO_GOLDEN[key] == value
        for key, value in SDRF_EXPERIMENTAL.items():
            assert SDRF_TO_GOLDEN[key] == value


class TestAnnotationMapping:
    """Tests for annotation characteristic mappings."""

    def test_has_all_agents(self):
        """All 3 agent types should be represented."""
        agents = {agent for _, (_, agent) in ANNOTATION_TO_GOLDEN.items()}
        assert "BiologicalAgent" in agents
        assert "TechnicalAgent" in agents
        assert "ExperimentalDesignAgent" in agents

    def test_values_are_tuples(self):
        """All values should be (field_name, agent_type) tuples."""
        for key, value in ANNOTATION_TO_GOLDEN.items():
            assert isinstance(value, tuple), f"Key {key}: expected tuple, got {type(value)}"
            assert len(value) == 2, f"Key {key}: expected 2-tuple, got {len(value)}"


class TestFieldToEntityType:
    """Tests for field-to-entity-type mapping."""

    def test_core_fields_covered(self):
        """Core normalizable fields should be mapped."""
        assert "species" in FIELD_TO_ENTITY_TYPE
        assert "cell_type" in FIELD_TO_ENTITY_TYPE
        assert "disease" in FIELD_TO_ENTITY_TYPE
        assert "organ" in FIELD_TO_ENTITY_TYPE
        assert "instrument" in FIELD_TO_ENTITY_TYPE

    def test_aliases_map_to_same_type(self):
        """Synonym fields should map to the same entity type."""
        assert FIELD_TO_ENTITY_TYPE["organ"] == FIELD_TO_ENTITY_TYPE["tissue"]
        assert FIELD_TO_ENTITY_TYPE["disease"] == FIELD_TO_ENTITY_TYPE["disease_state"]
        assert FIELD_TO_ENTITY_TYPE["species"] == FIELD_TO_ENTITY_TYPE["organism"]


class TestAgentFields:
    """Tests for agent field lists."""

    def test_all_3_agents_present(self):
        assert "BiologicalAgent" in AGENT_FIELDS
        assert "TechnicalAgent" in AGENT_FIELDS
        assert "ExperimentalDesignAgent" in AGENT_FIELDS

    def test_all_fields_is_concatenation(self):
        """ALL_AGENT_FIELDS should be the sum of all agent fields."""
        expected = (
            AGENT_FIELDS["BiologicalAgent"] +
            AGENT_FIELDS["TechnicalAgent"] +
            AGENT_FIELDS["ExperimentalDesignAgent"]
        )
        assert ALL_AGENT_FIELDS == expected

    def test_biological_has_species(self):
        assert "species" in AGENT_FIELDS["BiologicalAgent"]

    def test_technical_has_instrument(self):
        assert "instrument" in AGENT_FIELDS["TechnicalAgent"]


class TestMetadataOnlyFields:
    """Tests for metadata-only field set."""

    def test_is_set(self):
        assert isinstance(METADATA_ONLY_FIELDS, set)

    def test_contains_expected(self):
        assert "age" in METADATA_ONLY_FIELDS
        assert "sex" in METADATA_ONLY_FIELDS
        assert "precursor_tolerance" in METADATA_ONLY_FIELDS
