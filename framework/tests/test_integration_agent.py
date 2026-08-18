"""
Integration tests for the IntegrationAgent.

Tests the PRIDE descriptor priority logic and disagreement logging feature.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

from agents.integration_agent import IntegrationAgent


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_runassessor_data():
    """Sample RunAssessor data with PRIDE descriptors and tool inference."""
    return {
        "pxd_id": "PXD012345",
        "pipeline_version": "1.0",
        "pride_metadata": {
            "organisms": [
                {"name": "Homo sapiens", "accession": "9606", "cvLabel": "NCBITaxon"}
            ],
            "organismParts": [
                {"name": "liver", "accession": "UBERON:0002107"}
            ],
            "diseases": [
                {"name": "hepatocellular carcinoma", "accession": "DOID:684"}
            ],
            "instruments": [
                {"name": "Q Exactive HF", "accession": "MS:1002523"}
            ],
            "identifiedPTMStrings": [
                {"name": "Oxidation", "value": "Oxidation"}
            ],
            "experimentTypes": [
                {"name": "Shotgun proteomics", "accession": "PRIDE:0000311"}
            ],
            "quantificationMethods": [
                {"name": "Label-free quantification", "value": "Label-free"}
            ],
            "references": [
                {"pubmedID": 12345678}
            ]
        },
        "organism_identification": {
            "results": [
                {
                    "data": [
                        {"taxon_name": "Rattus norvegicus", "taxon_id": 10116, "score": 0.85},
                        {"taxon_name": "Mus musculus", "taxon_id": 10090, "score": 0.12}
                    ]
                }
            ]
        },
        "runAssessor": {
            "files": {
                "file1.raw": {
                    "instrument_model": {"name": "Orbitrap Fusion", "accession": "MS:1002416"},
                    "spectra_stats": {
                        "acquisition_type": "DDA",
                        "fragmentation_tag": "HCD",
                        "n_ms2_spectra": 50000,
                        "n_spectra": 55000
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_runassessor_data_no_pride_org():
    """RunAssessor data WITHOUT PRIDE organisms (should fallback to tool)."""
    return {
        "pxd_id": "PXD098765",
        "pipeline_version": "1.0",
        "pride_metadata": {
            "organisms": [],  # Empty - should fallback to tool
            "instruments": [
                {"name": "Q Exactive HF", "accession": "MS:1002523"}
            ],
            "references": [{"pubmedID": 87654321}]
        },
        "organism_identification": {
            "results": [
                {
                    "data": [
                        {"taxon_name": "Drosophila melanogaster", "taxon_id": 7227, "score": 0.95}
                    ]
                }
            ]
        },
        "runAssessor": {"files": {}}
    }


@pytest.fixture
def sample_llm_extraction():
    """Sample LLM extraction output."""
    return {
        "species": {
            "value": "Homo sapiens",
            "evidence": "Human liver tissue samples were analyzed"
        },
        "tissue": {
            "value": "liver",
            "evidence": "liver tissue samples"
        },
        "disease": {
            "value": "hepatocellular carcinoma",
            "evidence": "patients with hepatocellular carcinoma"
        },
        "_confidence": {
            "overall": 0.85
        }
    }


@pytest.fixture
def mock_runassessor_dir(tmp_path, sample_runassessor_data):
    """Create temporary RunAssessor directory with sample data."""
    ra_dir = tmp_path / "runassessor_data"
    ra_dir.mkdir()
    
    # Save sample data
    with open(ra_dir / "PXD012345_aggregated_results.json", 'w') as f:
        json.dump(sample_runassessor_data, f)
    
    return ra_dir


# ============================================================================
# PRIDE Priority Tests
# ============================================================================

class TestPRIDEPriority:
    """Tests for PRIDE descriptor priority logic."""
    
    def test_pride_organisms_prioritized_over_tool(
        self, mock_runassessor_dir, sample_runassessor_data, sample_llm_extraction
    ):
        """PRIDE organism should be used instead of tool inference."""
        agent = IntegrationAgent(str(mock_runassessor_dir))
        
        # Manually inject the data for testing
        result = agent.enrich("PXD012345", sample_llm_extraction, agent_type="BiologicalAgent")
        
        # PRIDE says "Homo sapiens" (accession 9606)
        # Tool says "Rattus norvegicus" (score 0.85)
        # Should use PRIDE value
        assert result["species"]["resolved"] == "Homo sapiens"
        assert result["species"]["sources"]["runassessor"]["value"] == "Homo sapiens"
        assert result["species"]["sources"]["runassessor"]["accession"] == "9606"
    
    def test_tool_used_when_no_pride_organisms(
        self, tmp_path, sample_runassessor_data_no_pride_org, sample_llm_extraction
    ):
        """Tool inference should be used when no PRIDE organisms exist."""
        ra_dir = tmp_path / "runassessor_data"
        ra_dir.mkdir()
        
        with open(ra_dir / "PXD098765_aggregated_results.json", 'w') as f:
            json.dump(sample_runassessor_data_no_pride_org, f)
        
        agent = IntegrationAgent(str(ra_dir))
        result = agent.enrich("PXD098765", sample_llm_extraction, agent_type="BiologicalAgent")
        
        # No PRIDE organisms, should fallback to tool
        assert result["species"]["resolved"] == "Drosophila melanogaster"
    
    def test_pride_instruments_prioritized(
        self, mock_runassessor_dir, sample_runassessor_data
    ):
        """PRIDE instruments should be prioritized over file analysis."""
        agent = IntegrationAgent(str(mock_runassessor_dir))
        
        tech_extraction = {
            "instrument": {"value": "unknown", "evidence": ""}
        }
        
        result = agent.enrich("PXD012345", tech_extraction, agent_type="TechnicalAgent")
        
        # PRIDE says "Q Exactive HF"
        # Files say "Orbitrap Fusion"
        # Should use PRIDE value
        assert result["instrument"]["resolved"] == "Q Exactive HF"


# ============================================================================
# Disagreement Detection Tests
# ============================================================================

class TestDisagreementDetection:
    """Tests for PRIDE vs Tool disagreement detection."""
    
    def test_disagreement_detected_for_species(
        self, mock_runassessor_dir, sample_llm_extraction
    ):
        """Should detect disagreement between PRIDE and tool for species."""
        agent = IntegrationAgent(str(mock_runassessor_dir))
        
        result = agent.enrich("PXD012345", sample_llm_extraction, agent_type="BiologicalAgent")
        
        # Should have detected disagreement (PRIDE: Homo sapiens, Tool: Rattus norvegicus)
        assert "_ra_disagreements" in result
        
        disagreements = result["_ra_disagreements"]
        species_disagreement = next(
            (d for d in disagreements if d["field"] == "species"), None
        )
        
        assert species_disagreement is not None
        assert species_disagreement["type"] == "PRIDE_VS_TOOL"
        assert species_disagreement["pride_value"] == "Homo sapiens"
        assert species_disagreement["tool_value"] == "Rattus norvegicus"
        assert species_disagreement["tool_name"] == "organism_identification (Peptonizer)"
    
    def test_no_disagreement_when_values_match(self, tmp_path, sample_llm_extraction):
        """No disagreement should be logged when PRIDE and tool agree."""
        ra_data = {
            "pxd_id": "PXD111111",
            "pride_metadata": {
                "organisms": [{"name": "Homo sapiens", "accession": "9606"}],
                "references": [{"pubmedID": 11111111}]
            },
            "organism_identification": {
                "results": [{"data": [{"taxon_name": "Homo sapiens", "taxon_id": 9606, "score": 0.99}]}]
            }
        }
        
        ra_dir = tmp_path / "runassessor_data"
        ra_dir.mkdir()
        with open(ra_dir / "PXD111111_aggregated_results.json", 'w') as f:
            json.dump(ra_data, f)
        
        agent = IntegrationAgent(str(ra_dir))
        result = agent.enrich("PXD111111", sample_llm_extraction, agent_type="BiologicalAgent")
        
        # No disagreement - values match
        assert "_ra_disagreements" not in result or len(result.get("_ra_disagreements", [])) == 0


# ============================================================================
# Disagreement Log File Tests
# ============================================================================

class TestDisagreementLogFile:
    """Tests for ra_disagreements.json log file creation."""
    
    def test_batch_creates_disagreement_log(self, mock_runassessor_dir, sample_llm_extraction, tmp_path):
        """enrich_batch should create ra_disagreements.json file."""
        agent = IntegrationAgent(str(mock_runassessor_dir))
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        results = {
            "test_ra_priority/PXD012345_test.txt": sample_llm_extraction
        }
        
        enriched = agent.enrich_batch(results, agent_name="BiologicalAgent", output_dir=str(output_dir))
        
        # Check log file was created
        log_path = output_dir / "ra_disagreements.json"
        assert log_path.exists()
        
        # Check log content
        with open(log_path) as f:
            log_data = json.load(f)
        
        assert len(log_data) > 0
        first_file = list(log_data.keys())[0]
        assert len(log_data[first_file]) > 0
        assert log_data[first_file][0]["type"] == "PRIDE_VS_TOOL"
    
    def test_batch_removes_disagreements_from_individual_outputs(
        self, mock_runassessor_dir, sample_llm_extraction, tmp_path
    ):
        """Disagreements should be in log file only, not in individual outputs."""
        agent = IntegrationAgent(str(mock_runassessor_dir))
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        results = {
            "test_ra_priority/PXD012345_test.txt": sample_llm_extraction
        }
        
        enriched = agent.enrich_batch(results, agent_name="BiologicalAgent", output_dir=str(output_dir))
        
        # Individual outputs should NOT have _ra_disagreements
        for filename, data in enriched.items():
            assert "_ra_disagreements" not in data
    
    def test_no_log_file_when_no_disagreements(self, tmp_path, sample_llm_extraction):
        """No log file should be created when there are no disagreements."""
        # Create data where PRIDE and tool agree
        ra_data = {
            "pxd_id": "PXD222222",
            "pride_metadata": {
                "organisms": [{"name": "Homo sapiens", "accession": "9606"}],
                "references": [{"pubmedID": 22222222}]
            },
            "organism_identification": {
                "results": [{"data": [{"taxon_name": "Homo sapiens", "taxon_id": 9606, "score": 0.99}]}]
            }
        }
        
        ra_dir = tmp_path / "runassessor_data"
        ra_dir.mkdir()
        with open(ra_dir / "PXD222222_aggregated_results.json", 'w') as f:
            json.dump(ra_data, f)
        
        agent = IntegrationAgent(str(ra_dir))
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        results = {"test/PXD222222_test.txt": sample_llm_extraction}
        agent.enrich_batch(results, agent_name="BiologicalAgent", output_dir=str(output_dir))
        
        # No log file since no disagreements
        log_path = output_dir / "ra_disagreements.json"
        assert not log_path.exists()


# ============================================================================
# PRIDE_TOOL_MAP Tests
# ============================================================================

class TestPRIDEToolMap:
    """Tests for PRIDE_TOOL_MAP configuration."""
    
    def test_all_mapped_fields_have_getters(self):
        """All fields in PRIDE_TOOL_MAP should have valid getter methods."""
        agent = IntegrationAgent("/tmp")  # Path doesn't matter for this test
        
        for field, (pride_getter, tool_getter, tool_name) in agent.PRIDE_TOOL_MAP.items():
            # Check PRIDE getter exists
            assert hasattr(agent, pride_getter), f"Missing PRIDE getter: {pride_getter}"
            
            # Check tool getter exists (if specified)
            if tool_getter:
                assert hasattr(agent, tool_getter), f"Missing tool getter: {tool_getter}"
                assert tool_name is not None, f"Tool name missing for {field}"
    
    def test_species_and_organism_share_getters(self):
        """species and organism should use the same PRIDE getter."""
        agent = IntegrationAgent("/tmp")
        
        species_pride, _, _ = agent.PRIDE_TOOL_MAP["species"]
        organism_pride, _, _ = agent.PRIDE_TOOL_MAP["organism"]
        
        assert species_pride == organism_pride == "_get_pride_organisms"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_runassessor_file(self, tmp_path, sample_llm_extraction):
        """Should handle missing RunAssessor file gracefully."""
        ra_dir = tmp_path / "empty_ra"
        ra_dir.mkdir()
        
        agent = IntegrationAgent(str(ra_dir))
        
        # Enrich with non-existent file - should fallback to LLM only
        result = agent.enrich("PXD999999", sample_llm_extraction, agent_type="BiologicalAgent")
        
        # Should still return valid result with LLM values
        assert "species" in result
        assert result["species"]["status"] == "LLM_ONLY"
    
    def test_empty_pride_metadata(self, tmp_path, sample_llm_extraction):
        """Should handle empty PRIDE metadata gracefully."""
        ra_data = {
            "pxd_id": "PXD333333",
            "pride_metadata": {},  # Empty
            "organism_identification": {"results": []}
        }
        
        ra_dir = tmp_path / "runassessor_data"
        ra_dir.mkdir()
        with open(ra_dir / "PXD333333_aggregated_results.json", 'w') as f:
            json.dump(ra_data, f)
        
        agent = IntegrationAgent(str(ra_dir))
        result = agent.enrich("PXD333333", sample_llm_extraction, agent_type="BiologicalAgent")
        
        # Should fallback to LLM values
        assert result["species"]["resolved"] == "Homo sapiens"
        assert result["species"]["status"] == "LLM_ONLY"
