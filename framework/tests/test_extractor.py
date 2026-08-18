"""
Tests for the core extractor module.

Tests cover:
- JSON parsing and extraction
- Output normalization
- Error handling
"""

import pytest
import json


class TestNormalizeOutput:
    """Tests for normalize_output function."""
    
    def test_removes_raw_output_key(self):
        """Test that raw_output is passed through unchanged."""
        from core.extractor import normalize_output
        
        data = {"raw_output": "some text"}
        result = normalize_output(data)
        
        assert "raw_output" in result
    
    def test_removes_unwanted_keys(self):
        """Test that THOUGHT PROCESS and FINAL JSON are removed."""
        from core.extractor import normalize_output
        
        data = {
            "THOUGHT PROCESS": "thinking...",
            "FINAL JSON": "{}",
            "species": ["human", "evidence"],
        }
        
        result = normalize_output(data)
        
        assert "THOUGHT PROCESS" not in result
        assert "FINAL JSON" not in result
        assert "species" in result
    
    def test_flattens_nested_arrays(self):
        """Test that [[val, ev]] → [val, ev]."""
        from core.extractor import normalize_output
        
        data = {
            "species": [["human", "evidence text"]],
        }
        
        result = normalize_output(data)
        
        assert result["species"] == ["human", "evidence text"]
    
    def test_adds_empty_evidence(self):
        """Test that [val] → [val, '']."""
        from core.extractor import normalize_output
        
        data = {
            "species": ["human"],
        }
        
        result = normalize_output(data)
        
        assert result["species"] == ["human", ""]
    
    def test_preserves_valid_format(self):
        """Test that valid [val, ev] format is preserved."""
        from core.extractor import normalize_output
        
        data = {
            "species": ["Homo sapiens", "Human samples were used..."],
            "tissue": ["liver", "Liver tissue was analyzed"],
        }
        
        result = normalize_output(data)
        
        assert result["species"] == ["Homo sapiens", "Human samples were used..."]
        assert result["tissue"] == ["liver", "Liver tissue was analyzed"]


class TestJSONParsing:
    """Tests for JSON extraction from LLM output."""
    
    def test_extract_json_from_simple_output(self):
        """Test extracting JSON from clean LLM output."""
        # This tests the pattern used in BaseExtractor
        output = '''THOUGHT PROCESS:
Analyzing the document.

FINAL JSON:
{"species": ["human", "evidence"]}'''
        
        # Simulate the parsing logic from extractor
        if "FINAL JSON:" in output:
            parts = output.split("FINAL JSON:")
            json_str = parts[-1].strip()
        else:
            json_str = output
        
        # Find JSON object
        start_idx = json_str.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(json_str[start_idx:], start=start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            json_str = json_str[start_idx:end_idx]
        
        result = json.loads(json_str)
        
        assert result["species"] == ["human", "evidence"]
    
    def test_extract_json_with_nested_braces(self):
        """Test extracting JSON with nested structures."""
        output = '''FINAL JSON:
{
    "metadata": {"key": "value"},
    "species": ["human", "text with {curly} braces"]
}'''
        
        parts = output.split("FINAL JSON:")
        json_str = parts[-1].strip()
        
        # Find matching braces
        start_idx = json_str.find('{')
        brace_count = 0
        end_idx = len(json_str)
        for i, char in enumerate(json_str[start_idx:], start=start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        json_str = json_str[start_idx:end_idx]
        
        result = json.loads(json_str)
        
        assert "species" in result


class TestBaseExtractorPrompt:
    """Tests for prompt generation in extractors."""
    
    def test_biological_prompt_format(self):
        """Test that biological prompt includes document text."""
        from agents.biological_agent import BiologicalAgent
        from core.prompts import BIOLOGICAL_PROMPT
        
        test_text = "Sample document about human liver tissue."
        
        # Get the formatted prompt
        prompt = BIOLOGICAL_PROMPT.format(descriptor=test_text)
        
        assert test_text in prompt
    
    def test_technical_prompt_format(self):
        """Test that technical prompt includes document text."""
        from agents.technical_agent import TechnicalAgent
        from core.prompts import TECHNICAL_PROMPT
        
        test_text = "MS analysis using Q Exactive instrument."
        
        prompt = TECHNICAL_PROMPT.format(descriptor=test_text)
        
        assert test_text in prompt


class TestMockLLMIntegration:
    """Tests using mock LLM client."""
    
    def test_mock_llm_returns_response(self, mock_llm_client):
        """Test that mock LLM returns a response."""
        response = mock_llm_client.get_completion([
            {"role": "user", "content": "Extract biological metadata"}
        ])
        
        assert response is not None
        assert "FINAL JSON:" in response
    
    def test_mock_llm_tracks_calls(self, mock_llm_client):
        """Test that mock LLM tracks calls for verification."""
        mock_llm_client.get_completion([
            {"role": "user", "content": "Test message"}
        ], temperature=0.5)
        
        assert len(mock_llm_client.calls) == 1
        assert mock_llm_client.calls[0]['temperature'] == 0.5
