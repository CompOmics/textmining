"""
Tests for the normalization module.

Tests cover:
- OntologyLoader parsing
- TermNormalizer matching
- NormalizationResult properties
"""

import pytest
from pathlib import Path


class TestOntologyLoader:
    """Tests for OntologyLoader parsing."""
    
    def test_load_obo_file(self, test_ontology_path: Path):
        """Test loading an OBO format ontology."""
        from normalization.ontology import OntologyLoader
        
        loader = OntologyLoader()
        graph = loader.load(str(test_ontology_path))
        
        assert graph is not None
        assert len(graph) > 0
    
    def test_get_node_by_id(self, test_ontology_path: Path):
        """Test retrieving a node by ID."""
        from normalization.ontology import OntologyLoader
        
        loader = OntologyLoader()
        graph = loader.load(str(test_ontology_path))
        
        node = graph.get_node("CL:0000066")
        assert node is not None
        assert node.name == "epithelial cell"
    
    def test_node_synonyms(self, test_ontology_path: Path):
        """Test that synonyms are loaded correctly."""
        from normalization.ontology import OntologyLoader
        
        loader = OntologyLoader()
        graph = loader.load(str(test_ontology_path))
        
        node = graph.get_node("CL:0000738")
        assert node is not None
        assert "white blood cell" in node.synonyms or "WBC" in node.synonyms
    
    def test_node_all_names(self, test_ontology_path: Path):
        """Test all_names includes name and synonyms."""
        from normalization.ontology import OntologyLoader
        
        loader = OntologyLoader()
        graph = loader.load(str(test_ontology_path))
        
        node = graph.get_node("NCBITaxon:9606")
        all_names = node.all_names
        
        assert "Homo sapiens" in all_names
        assert "human" in all_names

    def test_synonym_scope_and_taxon_constraint_are_preserved(self, tmp_path):
        """OBO synonym scope and in_taxon metadata survive parsing."""
        from normalization.ontology import OntologyLoader

        path = tmp_path / "scope.obo"
        path.write_text(
            "format-version: 1.2\n\n"
            "[Term]\n"
            "id: TEST:1\n"
            "name: canonical label\n"
            "synonym: \"alias\" RELATED []\n"
            "relationship: in_taxon NCBITaxon:50557 ! Insecta\n"
        )

        node = OntologyLoader().load(str(path)).get_node("TEST:1")

        assert node.synonym_scopes["alias"] == "RELATED"
        assert node.taxon_ids == ["NCBITaxon:50557"]


class TestNormalizationResult:
    """Tests for NormalizationResult dataclass."""
    
    def test_confidence_high(self):
        """Test high confidence classification."""
        from normalization.normalizer import NormalizationResult
        
        result = NormalizationResult(
            original_term="human",
            ontology_id="NCBITaxon:9606",
            similarity=0.95,
            is_normalized=True,
        )
        
        assert result.confidence == 'high'
    
    def test_confidence_medium(self):
        """Test medium confidence classification."""
        from normalization.normalizer import NormalizationResult
        
        result = NormalizationResult(
            original_term="epithelial",
            similarity=0.75,
            is_normalized=True,
        )
        
        assert result.confidence == 'medium'
    
    def test_confidence_low(self):
        """Test low confidence classification."""
        from normalization.normalizer import NormalizationResult
        
        result = NormalizationResult(
            original_term="unknown term",
            similarity=0.5,
            is_normalized=False,
        )
        
        assert result.confidence == 'low'
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        from normalization.normalizer import NormalizationResult
        
        result = NormalizationResult(
            original_term="human",
            ontology_id="NCBITaxon:9606",
            ontology_name="Homo sapiens",
            similarity=0.98,
            is_normalized=True,
            entity_type="species",
        )
        
        d = result.to_dict()
        
        assert d['original_term'] == "human"
        assert d['ontology_id'] == "NCBITaxon:9606"
        assert d['is_normalized'] == True
        assert 'confidence' in d


class TestTermNormalizer:
    """Tests for TermNormalizer (without embeddings)."""
    
    def test_get_ontology_for_entity(self):
        """Test entity type to ontology mapping."""
        from normalization.normalizer import TermNormalizer
        
        normalizer = TermNormalizer()
        
        assert normalizer.get_ontology_for_entity('species') == 'species'
        assert normalizer.get_ontology_for_entity('cell_type') == 'cl'
        assert normalizer.get_ontology_for_entity('tissue') == 'uberon'
        assert normalizer.get_ontology_for_entity('disease') == 'doid'
    
    def test_normalize_unknown_ontology(self):
        """Test normalization with unknown ontology returns not normalized."""
        from normalization.normalizer import TermNormalizer
        
        normalizer = TermNormalizer()
        
        result = normalizer.normalize("some term", ontology_id="nonexistent")
        
        assert result.is_normalized == False
    
    def test_normalize_no_entity_type(self):
        """Test normalization without entity type."""
        from normalization.normalizer import TermNormalizer
        
        normalizer = TermNormalizer()
        
        result = normalizer.normalize("some term")
        
        # Without entity type or ontology_id, should return not normalized
        assert result.is_normalized == False

    def test_exact_primary_name_beats_identical_synonym(self):
        """A canonical label must beat the same text on another node."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyGraph, OntologyNode

        class WrongFirstIndex:
            def search(self, query, top_k=None):
                raise AssertionError("exact primary matches must bypass embeddings")

        graph = OntologyGraph("uberon")
        graph.add_node(OntologyNode(id="UBERON:0000955", name="brain"))
        graph.add_node(OntologyNode(
            id="UBERON:6110636",
            name="insect adult cerebral ganglion",
            synonyms=["brain"],
        ))
        normalizer = TermNormalizer()
        normalizer.graphs["uberon"] = graph
        normalizer.indices["uberon"] = WrongFirstIndex()

        result = normalizer.normalize(" Brain ", ontology_id="uberon")

        assert result.ontology_id == "UBERON:0000955"
        assert result.ontology_name == "brain"
        assert result.matched_text == "brain"
        assert result.similarity == 1.0
        assert result.is_normalized is True

    def test_batch_exact_primary_names_bypass_embeddings(self):
        """Batch normalization applies the same primary-name precedence."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyGraph, OntologyNode

        class NoSearchIndex:
            def search_many(self, queries, top_k=None):
                raise AssertionError("all exact primary matches should skip embeddings")

        graph = OntologyGraph("uberon")
        graph.add_node(OntologyNode(id="UBERON:0000955", name="brain"))
        graph.add_node(OntologyNode(
            id="UBERON:6110636",
            name="insect adult cerebral ganglion",
            synonyms=["brain"],
        ))
        graph.add_node(OntologyNode(id="UBERON:0000948", name="heart"))
        normalizer = TermNormalizer()
        normalizer.graphs["uberon"] = graph
        normalizer.indices["uberon"] = NoSearchIndex()

        results = normalizer.normalize_batch(
            ["brain", "HEART"], ontology_id="uberon"
        )

        assert [result.ontology_id for result in results] == [
            "UBERON:0000955",
            "UBERON:0000948",
        ]

    def test_imported_primary_name_does_not_bypass_target_ontology(self):
        """A primary label imported from HP must not override MONDO search."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyGraph, OntologyNode

        class MondoIndex:
            def search(self, query, top_k=None):
                return [("MONDO:0011122", "obesity", 1.0)]

        graph = OntologyGraph("mondo")
        graph.add_node(OntologyNode(id="HP:0001513", name="obesity"))
        graph.add_node(OntologyNode(
            id="MONDO:0011122",
            name="obsolete morbid obesity",
        ))
        normalizer = TermNormalizer()
        normalizer.graphs["mondo"] = graph
        normalizer.indices["mondo"] = MondoIndex()

        result = normalizer.normalize("obesity", ontology_id="mondo")

        assert result.ontology_id == "MONDO:0011122"

    def test_ambiguous_related_synonym_is_not_auto_normalized(self):
        """FAISS ordering must not resolve an exact-string synonym tie."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyGraph, OntologyNode

        class NoSearchIndex:
            def search(self, query, top_k=None):
                raise AssertionError("ambiguous synonym must bypass FAISS")

        graph = OntologyGraph("uberon")
        graph.add_node(OntologyNode(
            id="UBERON:1", name="first", synonyms=["skin"],
            synonym_scopes={"skin": "RELATED"},
        ))
        graph.add_node(OntologyNode(
            id="UBERON:2", name="second", synonyms=["skin"],
            synonym_scopes={"skin": "RELATED"},
        ))
        normalizer = TermNormalizer()
        normalizer.graphs["uberon"] = graph
        normalizer.indices["uberon"] = NoSearchIndex()

        result = normalizer.normalize("skin", ontology_id="uberon")

        assert result.is_normalized is False
        assert result.normalization_method == "ambiguous_exact_synonym"
        assert result.qc_flags == ["ambiguous_ontology_match"]
        assert {candidate[0] for candidate in result.candidates} == {
            "UBERON:1", "UBERON:2"
        }

    def test_unique_exact_synonym_is_accepted(self):
        """An unambiguous OBO EXACT synonym remains a deterministic match."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyGraph, OntologyNode

        class NoSearchIndex:
            def search(self, query, top_k=None):
                raise AssertionError("exact synonym should bypass embeddings")

        graph = OntologyGraph("species")
        graph.add_node(OntologyNode(
            id="NCBITaxon:9606", name="Homo sapiens",
            synonyms=["human"], synonym_scopes={"human": "EXACT"},
        ))
        normalizer = TermNormalizer()
        normalizer.graphs["species"] = graph
        normalizer.indices["species"] = NoSearchIndex()

        result = normalizer.normalize("human", ontology_id="species")

        assert result.ontology_id == "NCBITaxon:9606"
        assert result.normalization_method == "exact_synonym"
        assert result.synonym_scope == "EXACT"

    def test_generic_ptm_uses_unimod_primary_name(self):
        """Generic search-engine modification labels route to Unimod."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyGraph, OntologyNode

        class NoSearchIndex:
            def search(self, query, top_k=None):
                raise AssertionError("generic PTM should bypass PSI-MOD search")

        psimod = OntologyGraph("psimod")
        unimod = OntologyGraph("unimod")
        unimod.add_node(OntologyNode(id="UNIMOD:35", name="Oxidation"))
        normalizer = TermNormalizer()
        normalizer.graphs.update({"psimod": psimod, "unimod": unimod})
        normalizer.indices["psimod"] = NoSearchIndex()

        result = normalizer.normalize(
            "oxidation", entity_type="modification", ontology_id="psimod"
        )

        assert result.ontology_id == "UNIMOD:35"
        assert result.normalization_method == "exact_primary_unimod"

    def test_graph_primary_lookup_is_not_overwritten_by_synonym(self):
        """The graph-level exact lookup also prefers canonical labels."""
        from normalization.ontology import OntologyGraph, OntologyNode

        graph = OntologyGraph("uberon")
        canonical = OntologyNode(id="UBERON:0000955", name="brain")
        graph.add_node(canonical)
        graph.add_node(OntologyNode(
            id="UBERON:6110636",
            name="insect adult cerebral ganglion",
            synonyms=["brain"],
        ))

        assert graph.get_by_name("brain") is canonical


class TestTermExpansion:
    """Tests for TermNormalizer._expand_term() -- pure logic, no embeddings needed."""

    def _make_normalizer(self, aliases=None):
        from normalization.normalizer import TermNormalizer
        from normalization.config import NormalizationConfig
        config = NormalizationConfig()
        if aliases:
            config.term_aliases = aliases
        return TermNormalizer(config)

    # --- Genus-prefix heuristic (p.falciparum style) ---

    def test_expand_dotted_genus_falciparum(self):
        """p.falciparum: _expand_term no longer handles single-letter prefix.
        Graph injection (at index-build time) handles this form instead."""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("p.falciparum")
        assert result is None  # handled by graph injection, not query expansion

    def test_expand_dotted_genus_ecoli(self):
        """e.coli: single-letter form handled by graph injection, not _expand_term."""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("e.coli")
        assert result is None

    def test_expand_dotted_genus_hsapiens(self):
        """h.sapiens: single-letter form handled by graph injection, not _expand_term."""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("h.sapiens")
        assert result is None

    def test_expand_dotted_genus_musculus(self):
        """m.musculus: single-letter form handled by graph injection, not _expand_term."""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("m.musculus")
        assert result is None

    def test_expand_unknown_prefix_returns_none(self):
        """Unknown single-letter prefix with no alias should return None."""
        normalizer = self._make_normalizer()
        # 'y' is not in _GENUS_PREFIX_MAP
        result = normalizer._expand_term("y.species")
        assert result is None

    # --- Dot-binomial where genus is written in full ---

    def test_expand_dot_binomial_full_genus(self):
        """Plasmodium.falciparum → Plasmodium falciparum"""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("Plasmodium.falciparum")
        assert result == "Plasmodium falciparum"

    def test_expand_dot_binomial_homo_sapiens(self):
        """Homo.sapiens → Homo sapiens"""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("Homo.sapiens")
        assert result == "Homo sapiens"

    # --- Alias dict lookup ---

    def test_expand_alias_dict_exact(self):
        """Custom alias is expanded correctly."""
        normalizer = self._make_normalizer(aliases={"Pf": "Plasmodium falciparum"})
        result = normalizer._expand_term("Pf")
        assert result == "Plasmodium falciparum"

    def test_expand_alias_dict_case_insensitive(self):
        """Alias lookup is case-insensitive on the key."""
        normalizer = self._make_normalizer(aliases={"PF": "Plasmodium falciparum"})
        result = normalizer._expand_term("pf")
        assert result == "Plasmodium falciparum"

    # --- No-change cases ---

    def test_no_expansion_for_full_name(self):
        """Full binomial names should not be modified."""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("Homo sapiens")
        assert result is None

    def test_no_expansion_for_plain_word(self):
        """Plain single words with no dots should not be modified."""
        normalizer = self._make_normalizer()
        result = normalizer._expand_term("human")
        assert result is None

    # --- NormalizationResult fields ---

    def test_expanded_term_field_present_in_result(self):
        """When expansion fires, the expanded_term field is populated."""
        from normalization.normalizer import NormalizationResult
        result = NormalizationResult(
            original_term="p.falciparum",
            expanded_term="Plasmodium falciparum",
            ontology_id="NCBITaxon:5833",
            similarity=0.95,
            is_normalized=True,
        )
        d = result.to_dict()
        assert d.get('expanded_term') == "Plasmodium falciparum"

    def test_expanded_term_absent_when_none(self):
        """When no expansion, expanded_term key is absent from to_dict output."""
        from normalization.normalizer import NormalizationResult
        result = NormalizationResult(
            original_term="Homo sapiens",
            ontology_id="NCBITaxon:9606",
            similarity=0.98,
            is_normalized=True,
        )
        d = result.to_dict()
        assert 'expanded_term' not in d


class TestGraphInjection:
    """Tests for TermNormalizer._inject_abbreviated_synonyms()."""

    def _make_node(self, name, synonyms=None, obsolete=False):
        """Create a minimal OntologyNode for testing."""
        from normalization.ontology import OntologyNode
        return OntologyNode(id="FAKE:0001", name=name,
                            synonyms=list(synonyms or []),
                            is_obsolete=obsolete)

    def _make_normalizer_with_nodes(self, nodes):
        from normalization.normalizer import TermNormalizer
        from unittest.mock import MagicMock
        normalizer = TermNormalizer()
        # Build a minimal fake graph iterable
        fake_graph = nodes
        return normalizer, fake_graph

    def test_injects_single_letter_and_dot_binomial(self):
        """Plasmodium falciparum should get p.falciparum and Plasmodium.falciparum."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyNode
        normalizer = TermNormalizer()
        node = OntologyNode(id="NCBITaxon:5833", name="Plasmodium falciparum")
        normalizer._inject_abbreviated_synonyms([node])
        assert "p.falciparum" in node.synonyms
        assert "Plasmodium.falciparum" in node.synonyms

    def test_injects_for_homo_sapiens(self):
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyNode
        normalizer = TermNormalizer()
        node = OntologyNode(id="NCBITaxon:9606", name="Homo sapiens")
        normalizer._inject_abbreviated_synonyms([node])
        assert "h.sapiens" in node.synonyms
        assert "Homo.sapiens" in node.synonyms

    def test_skips_obsolete_nodes(self):
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyNode
        normalizer = TermNormalizer()
        node = OntologyNode(id="NCBITaxon:0", name="Homo sapiens", is_obsolete=True)
        normalizer._inject_abbreviated_synonyms([node])
        assert "h.sapiens" not in node.synonyms

    def test_no_injection_for_single_word_names(self):
        """Non-binomial names (single word) should not get abbreviated synonyms."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyNode
        normalizer = TermNormalizer()
        node = OntologyNode(id="CL:0000001", name="cell")
        normalizer._inject_abbreviated_synonyms([node])
        assert node.synonyms == []

    def test_no_duplicate_synonyms(self):
        """Injection should be idempotent (running twice doesn't duplicate)."""
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyNode
        normalizer = TermNormalizer()
        node = OntologyNode(id="NCBITaxon:5833", name="Plasmodium falciparum")
        normalizer._inject_abbreviated_synonyms([node])
        before = list(node.synonyms)
        normalizer._inject_abbreviated_synonyms([node])
        assert node.synonyms == before  # no duplicates

    def test_returns_count_of_modified_nodes(self):
        from normalization.normalizer import TermNormalizer
        from normalization.ontology import OntologyNode
        normalizer = TermNormalizer()
        nodes = [
            OntologyNode(id="A", name="Homo sapiens"),
            OntologyNode(id="B", name="Mus musculus"),
            OntologyNode(id="C", name="cell"),  # not binomial
        ]
        count = normalizer._inject_abbreviated_synonyms(nodes)
        assert count == 2  # only the two binomial nodes


class TestRegisterSynonym:
    """Tests for TermNormalizer.register_synonym() — graph + persistence."""

    def _make_normalizer_with_fake_ontology(self, tmp_path):
        """Set up a TermNormalizer with a minimal fake loaded ontology."""
        from normalization.normalizer import TermNormalizer
        from normalization.config import NormalizationConfig
        from normalization.ontology import OntologyNode

        config = NormalizationConfig(cache_dir=str(tmp_path))
        normalizer = TermNormalizer(config)

        # Inject a fake graph and a None index (no embeddings needed for graph tests)
        node = OntologyNode(id="NCBITaxon:5833", name="Plasmodium falciparum")
        fake_graph = [node]
        normalizer.graphs["species"] = fake_graph
        normalizer.indices["species"] = None  # index not needed for persistence tests

        return normalizer, node

    def test_register_adds_synonym_to_node(self, tmp_path):
        normalizer, node = self._make_normalizer_with_fake_ontology(tmp_path)
        result = normalizer.register_synonym("p.falciparum", "Plasmodium falciparum", "species")
        assert result is True
        assert "p.falciparum" in node.synonyms

    def test_register_persists_to_json(self, tmp_path):
        import json
        normalizer, node = self._make_normalizer_with_fake_ontology(tmp_path)
        normalizer.register_synonym("p.falciparum", "Plasmodium falciparum", "species")

        path = normalizer._custom_synonyms_path()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "p.falciparum" in data["species"]["Plasmodium falciparum"]

    def test_register_is_idempotent(self, tmp_path):
        """Registering the same synonym twice should not duplicate it."""
        import json
        normalizer, node = self._make_normalizer_with_fake_ontology(tmp_path)
        normalizer.register_synonym("p.falciparum", "Plasmodium falciparum", "species")
        normalizer.register_synonym("p.falciparum", "Plasmodium falciparum", "species")

        assert node.synonyms.count("p.falciparum") == 1
        data = json.loads(normalizer._custom_synonyms_path().read_text())
        assert data["species"]["Plasmodium falciparum"].count("p.falciparum") == 1

    def test_register_unknown_node_returns_false(self, tmp_path):
        normalizer, _ = self._make_normalizer_with_fake_ontology(tmp_path)
        result = normalizer.register_synonym("x.foo", "Nonexistent organism", "species")
        assert result is False

    def test_register_unknown_ontology_returns_false(self, tmp_path):
        normalizer, _ = self._make_normalizer_with_fake_ontology(tmp_path)
        result = normalizer.register_synonym("x.foo", "Plasmodium falciparum", "nonexistent_ont")
        assert result is False


class TestNormalizationConfig:
    """Tests for NormalizationConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        from normalization.config import NormalizationConfig
        
        config = NormalizationConfig()
        
        assert config.similarity_threshold == 0.7
        assert config.top_k == 5
        assert config.use_synonyms == True
    
    def test_get_cache_path(self):
        """Test cache path generation."""
        from normalization.config import NormalizationConfig
        
        config = NormalizationConfig(cache_dir="/tmp/cache")
        
        path = config.get_cache_path("cl")
        
        assert str(path) == "/tmp/cache/cl_index.pkl"
    
    def test_entity_ontology_map(self):
        """Test entity to ontology mapping exists."""
        from normalization.config import NormalizationConfig
        
        config = NormalizationConfig()
        
        assert 'species' in config.entity_ontology_map
        assert 'cell_type' in config.entity_ontology_map
        assert 'tissue' in config.entity_ontology_map
