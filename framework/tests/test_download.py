"""
Tests for the download module.

Tests cover:
- Species ontology creation
- Ontology info retrieval
- Check functions
"""

import pytest
from pathlib import Path


class TestCuratedSpecies:
    """Tests for curated species list."""
    
    def test_curated_species_not_empty(self):
        """Ensure CURATED_SPECIES has entries."""
        from normalization.download import CURATED_SPECIES
        
        assert len(CURATED_SPECIES) > 0
    
    def test_curated_species_format(self):
        """Test that each species has correct format."""
        from normalization.download import CURATED_SPECIES
        
        for entry in CURATED_SPECIES:
            assert len(entry) == 3, f"Entry should have 3 elements: {entry}"
            term_id, name, synonyms = entry
            
            assert term_id.startswith("NCBITaxon:"), f"ID should start with NCBITaxon: {term_id}"
            assert isinstance(name, str) and len(name) > 0
            assert isinstance(synonyms, list)
    
    def test_common_species_present(self):
        """Ensure common model organisms are included."""
        from normalization.download import CURATED_SPECIES
        
        species_names = [entry[1] for entry in CURATED_SPECIES]
        
        assert "Homo sapiens" in species_names
        assert "Mus musculus" in species_names
        assert "Rattus norvegicus" in species_names


class TestCreateSpeciesOntology:
    """Tests for species.obo generation."""
    
    def test_creates_file(self, tmp_path: Path):
        """Test that species.obo file is created."""
        from normalization.download import create_species_ontology
        
        success, message = create_species_ontology(tmp_path)
        
        assert success == True
        assert (tmp_path / "species.obo").exists()
    
    def test_file_content_valid(self, tmp_path: Path):
        """Test that generated file has valid OBO content."""
        from normalization.download import create_species_ontology
        
        create_species_ontology(tmp_path)
        content = (tmp_path / "species.obo").read_text()
        
        # Check header
        assert "format-version: 1.2" in content
        assert "ontology: ncbitaxon_subset" in content
        
        # Check some terms
        assert "[Term]" in content
        assert "id: NCBITaxon:9606" in content
        assert 'name: Homo sapiens' in content
    
    def test_skips_existing_file(self, tmp_path: Path):
        """Test that existing file is not overwritten without force."""
        from normalization.download import create_species_ontology
        
        # Create file first
        output_path = tmp_path / "species.obo"
        output_path.write_text("existing content")
        
        success, message = create_species_ontology(tmp_path, force=False)
        
        assert success == True
        assert "Already exists" in message
        assert output_path.read_text() == "existing content"
    
    def test_force_overwrites(self, tmp_path: Path):
        """Test that force=True overwrites existing file."""
        from normalization.download import create_species_ontology
        
        # Create file first
        output_path = tmp_path / "species.obo"
        output_path.write_text("existing content")
        
        success, message = create_species_ontology(tmp_path, force=True)
        
        assert success == True
        assert "Created with" in message
        assert output_path.read_text() != "existing content"


class TestOntologyInfo:
    """Tests for ontology info functions."""
    
    def test_get_ontology_info_exists(self):
        """Test getting info for existing ontology."""
        from normalization.download import get_ontology_info
        
        info = get_ontology_info('cl')
        
        assert info is not None
        assert 'url' in info
        assert 'filename' in info
        assert 'description' in info
    
    def test_get_ontology_info_not_found(self):
        """Test getting info for nonexistent ontology."""
        from normalization.download import get_ontology_info
        
        info = get_ontology_info('nonexistent')
        
        assert info is None
    
    def test_get_species_info(self):
        """Test getting info for custom species ontology."""
        from normalization.download import get_ontology_info
        
        info = get_ontology_info('species')
        
        assert info is not None
        assert info['filename'] == 'species.obo'


class TestOntologySources:
    """Tests for ONTOLOGY_SOURCES constant."""
    
    def test_all_have_required_fields(self):
        """Test that all sources have required fields."""
        from normalization.download import ONTOLOGY_SOURCES
        
        for ont_id, info in ONTOLOGY_SOURCES.items():
            assert 'url' in info, f"{ont_id} missing 'url'"
            assert 'filename' in info, f"{ont_id} missing 'filename'"
            assert 'description' in info, f"{ont_id} missing 'description'"
    
    def test_urls_are_valid_format(self):
        """Test that URLs have valid format."""
        from normalization.download import ONTOLOGY_SOURCES
        
        for ont_id, info in ONTOLOGY_SOURCES.items():
            url = info['url']
            assert url.startswith('http'), f"{ont_id} URL should start with http: {url}"
    
    def test_filenames_have_extension(self):
        """Test that filenames have proper extensions."""
        from normalization.download import ONTOLOGY_SOURCES
        
        for ont_id, info in ONTOLOGY_SOURCES.items():
            filename = info['filename']
            assert filename.endswith('.obo') or filename.endswith('.owl'), \
                f"{ont_id} should have .obo or .owl extension: {filename}"


class TestCheckOntologies:
    """Tests for check_ontologies function."""
    
    def test_check_empty_dir(self, tmp_path: Path):
        """Test checking empty directory."""
        from normalization.download import check_ontologies
        
        status = check_ontologies(tmp_path)
        
        # All should be marked as not existing
        for ont_id, info in status.items():
            assert info['exists'] == False
    
    def test_check_with_file(self, tmp_path: Path):
        """Test checking directory with a file."""
        from normalization.download import check_ontologies
        
        # Create a fake cl.obo
        (tmp_path / "cl.obo").write_text("fake content")
        
        status = check_ontologies(tmp_path)
        
        assert status['cl']['exists'] == True
        assert status['cl']['size_mb'] > 0
