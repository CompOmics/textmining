"""
Pytest configuration and shared fixtures for the extraction framework tests.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_ontology_path(fixtures_dir: Path) -> Path:
    """Path to the mini test ontology."""
    return fixtures_dir / "test_ontology.obo"


@pytest.fixture
def test_document_path(fixtures_dir: Path) -> Path:
    """Path to sample test document."""
    return fixtures_dir / "test_document.txt"


@pytest.fixture
def expected_output_path(fixtures_dir: Path) -> Path:
    """Path to expected extraction output."""
    return fixtures_dir / "expected_output.json"


@pytest.fixture
def test_document(test_document_path: Path) -> str:
    """Contents of the test document."""
    return test_document_path.read_text()


@pytest.fixture
def expected_output(expected_output_path: Path) -> Dict[str, Any]:
    """Expected extraction output as dictionary."""
    return json.loads(expected_output_path.read_text())


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

class MockLLMClient:
    """Mock LLM client that returns predefined responses."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.calls: List[Dict[str, Any]] = []
    
    def get_completion(self, messages: List[Dict], temperature: float = 0.0) -> str:
        """Return mock response based on prompt content."""
        self.calls.append({
            'messages': messages,
            'temperature': temperature,
        })
        
        # Default mock response
        user_content = messages[-1].get('content', '') if messages else ''
        
        # Check for predefined responses
        for key, response in self.responses.items():
            if key.lower() in user_content.lower():
                return response
        
        # Default extraction response
        return '''THOUGHT PROCESS:
Analyzing the document for metadata extraction.

FINAL JSON:
{
    "species": ["Homo sapiens", "Sample text evidence"],
    "tissue": ["liver", "liver tissue samples"],
    "cell_type": ["T cell", "T cells were isolated"],
    "disease": ["cancer", "cancer-associated markers"]
}'''


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_responses() -> Dict[str, str]:
    """Predefined LLM responses for specific prompts."""
    return {
        'biological': '''THOUGHT PROCESS:
Looking for species, tissue, and cell types.

FINAL JSON:
{
    "species": ["Homo sapiens", "quantitative proteomics on liver tissue samples from patients"],
    "tissue": ["liver", "human liver tissue samples"],
    "cell_type": ["T cell", "T cells and B cells were isolated"],
    "disease": ["hepatocellular carcinoma", "biomarkers for hepatocellular carcinoma"]
}''',
        'technical': '''THOUGHT PROCESS:
Looking for instruments and methods.

FINAL JSON:
{
    "instrument": ["Q Exactive HF", "LC-MS/MS analysis on a Q Exactive HF instrument"],
    "labelling": ["TMT", "analyzed using TMT labeling"]
}''',
    }


# ============================================================================
# Normalization Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory for ontology cache during tests."""
    cache_dir = tmp_path / "ontology_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_ontology_dir(tmp_path: Path, test_ontology_path: Path) -> Path:
    """Temporary ontology directory with test ontology."""
    ont_dir = tmp_path / "ontologies"
    ont_dir.mkdir()
    
    # Copy test ontology
    (ont_dir / "test.obo").write_text(test_ontology_path.read_text())
    
    return ont_dir


# ============================================================================
# Utility Functions
# ============================================================================

def assert_extraction_has_field(result: Dict, field: str, expected_value: str = None):
    """Assert that extraction result has a field with optional value check."""
    assert field in result, f"Missing field: {field}"
    
    value = result[field]
    
    if expected_value:
        if isinstance(value, list):
            assert value[0] == expected_value, f"Expected {expected_value}, got {value[0]}"
        elif isinstance(value, dict):
            assert value.get('value') == expected_value
        else:
            assert value == expected_value


def assert_normalization_result(result: Dict, field: str, 
                                 should_be_normalized: bool = True,
                                 expected_id_prefix: str = None):
    """Assert normalization result has expected properties."""
    assert field in result, f"Missing field: {field}"
    
    value = result[field]
    
    if isinstance(value, dict):
        assert 'is_normalized' in value
        
        if should_be_normalized:
            assert value['is_normalized'] == True
            assert value.get('ontology_id') is not None
            
            if expected_id_prefix:
                assert value['ontology_id'].startswith(expected_id_prefix)
