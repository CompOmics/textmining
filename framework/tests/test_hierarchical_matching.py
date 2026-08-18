"""
Tests for the HIERARCHICAL matching tier and label normalization.

Tests cover:
- HierarchicalMatcher cell line fallback lookup
- HierarchicalMatcher._find_ancestor_distance() with mock graph
- HierarchicalMatcher._check_hierarchical_match() end-to-end
- _normalize_label_value() deterministic mapping
- CELL_LINE_TO_CELL_TYPE completeness
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════
#  Minimal mock classes so tests run without loading real ontologies/SapBERT
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MockOntologyNode:
    id: str
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


class MockOntologyGraph:
    """In-memory mini ontology for testing."""
    
    def __init__(self, nodes: dict):
        self._nodes = nodes  # id -> MockOntologyNode
    
    def get_node(self, node_id: str) -> Optional[MockOntologyNode]:
        return self._nodes.get(node_id)


@dataclass
class MockNormalizationResult:
    original_term: str
    ontology_id: Optional[str] = None
    ontology_name: Optional[str] = None
    similarity: float = 0.0
    is_normalized: bool = False


class MockTermNormalizer:
    """Normalizer that returns pre-configured results."""
    
    def __init__(self, term_map: dict, graphs: dict = None):
        """
        Args:
            term_map: Dict of (term, ontology_id) -> MockNormalizationResult
            graphs: Dict of ontology_id -> MockOntologyGraph
        """
        self._term_map = term_map
        self.graphs = graphs or {}
    
    def normalize(self, term: str, ontology_id: str = None, **kwargs):
        key = (term.lower().strip(), ontology_id)
        if key in self._term_map:
            return self._term_map[key]
        return MockNormalizationResult(original_term=term, is_normalized=False)


# ══════════════════════════════════════════════════════════════════════════
#  Build a miniature Cell Ontology hierarchy for testing
# ══════════════════════════════════════════════════════════════════════════

def _build_mini_cl_graph():
    """
    Mini CL hierarchy:
      CL:0000000 (cell)
        └── CL:0000066 (epithelial cell)
              └── CL:0002327 (mammary gland epithelial cell)
        └── CL:0000084 (T cell)
    """
    nodes = {
        "CL:0000000": MockOntologyNode(
            id="CL:0000000", name="cell",
            children=["CL:0000066", "CL:0000084"]
        ),
        "CL:0000066": MockOntologyNode(
            id="CL:0000066", name="epithelial cell",
            parents=["CL:0000000"], children=["CL:0002327"]
        ),
        "CL:0002327": MockOntologyNode(
            id="CL:0002327", name="mammary gland epithelial cell",
            parents=["CL:0000066"]
        ),
        "CL:0000084": MockOntologyNode(
            id="CL:0000084", name="T cell",
            parents=["CL:0000000"]
        ),
    }
    return MockOntologyGraph(nodes)


def _build_mini_uberon_graph():
    """
    Mini UBERON hierarchy:
      UBERON:0001062 (anatomical entity)
        └── UBERON:0000059 (large intestine)
              └── UBERON:0001155 (colon)
                    └── UBERON:0001159 (sigmoid colon)
    """
    nodes = {
        "UBERON:0001062": MockOntologyNode(
            id="UBERON:0001062", name="anatomical entity",
            children=["UBERON:0000059"]
        ),
        "UBERON:0000059": MockOntologyNode(
            id="UBERON:0000059", name="large intestine",
            parents=["UBERON:0001062"], children=["UBERON:0001155"]
        ),
        "UBERON:0001155": MockOntologyNode(
            id="UBERON:0001155", name="colon",
            parents=["UBERON:0000059"], children=["UBERON:0001159"]
        ),
        "UBERON:0001159": MockOntologyNode(
            id="UBERON:0001159", name="sigmoid colon",
            parents=["UBERON:0001155"]
        ),
    }
    return MockOntologyGraph(nodes)


# ══════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def cl_graph():
    return _build_mini_cl_graph()


@pytest.fixture
def uberon_graph():
    return _build_mini_uberon_graph()


@pytest.fixture
def matcher_with_hierarchy(cl_graph, uberon_graph):
    """HierarchicalMatcher with mock normalizer and ontology graphs."""
    from benchmark.semantic_matcher import HierarchicalMatcher
    
    term_map = {
        # CL terms
        ("epithelial cell", "cl"): MockNormalizationResult(
            original_term="epithelial cell", ontology_id="CL:0000066",
            ontology_name="epithelial cell", similarity=0.98, is_normalized=True
        ),
        ("mammary gland epithelial cell", "cl"): MockNormalizationResult(
            original_term="mammary gland epithelial cell", ontology_id="CL:0002327",
            ontology_name="mammary gland epithelial cell", similarity=0.95, is_normalized=True
        ),
        ("t cell", "cl"): MockNormalizationResult(
            original_term="T cell", ontology_id="CL:0000084",
            ontology_name="T cell", similarity=0.97, is_normalized=True
        ),
        # UBERON terms
        ("colon", "uberon"): MockNormalizationResult(
            original_term="colon", ontology_id="UBERON:0001155",
            ontology_name="colon", similarity=0.99, is_normalized=True
        ),
        ("sigmoid colon", "uberon"): MockNormalizationResult(
            original_term="sigmoid colon", ontology_id="UBERON:0001159",
            ontology_name="sigmoid colon", similarity=0.96, is_normalized=True
        ),
        ("large intestine", "uberon"): MockNormalizationResult(
            original_term="large intestine", ontology_id="UBERON:0000059",
            ontology_name="large intestine", similarity=0.97, is_normalized=True
        ),
    }
    
    normalizer = MockTermNormalizer(
        term_map=term_map,
        graphs={"cl": cl_graph, "uberon": uberon_graph}
    )
    
    matcher = HierarchicalMatcher(
        semantic_threshold=0.70,
        term_normalizer=normalizer,
        field_ontology_map={"cell_type": "cl", "organ": "uberon", "disease": "doid"},
    )
    return matcher


# ══════════════════════════════════════════════════════════════════════════
#  Tests: Cell Line Lookup
# ══════════════════════════════════════════════════════════════════════════

class TestCellLineLookup:
    """Tests for the CELL_LINE_TO_CELL_TYPE fallback dictionary."""
    
    def test_common_epithelial_lines(self):
        from benchmark.semantic_matcher import CELL_LINE_TO_CELL_TYPE
        assert CELL_LINE_TO_CELL_TYPE["hela"] == "epithelial cell"
        assert CELL_LINE_TO_CELL_TYPE["mcf-7"] == "epithelial cell"
        assert CELL_LINE_TO_CELL_TYPE["a549"] == "epithelial cell"
    
    def test_immune_lines(self):
        from benchmark.semantic_matcher import CELL_LINE_TO_CELL_TYPE
        assert CELL_LINE_TO_CELL_TYPE["jurkat"] == "T cell"
        assert CELL_LINE_TO_CELL_TYPE["thp-1"] == "monocyte"
        assert CELL_LINE_TO_CELL_TYPE["k562"] == "myeloid cell"
    
    def test_hepatocyte_lines(self):
        from benchmark.semantic_matcher import CELL_LINE_TO_CELL_TYPE
        assert CELL_LINE_TO_CELL_TYPE["hepg2"] == "hepatocyte"
        assert CELL_LINE_TO_CELL_TYPE["huh-7"] == "hepatocyte"
    
    def test_case_insensitive_convention(self):
        """All keys should be lowercase."""
        from benchmark.semantic_matcher import CELL_LINE_TO_CELL_TYPE
        for key in CELL_LINE_TO_CELL_TYPE:
            assert key == key.lower(), f"Key '{key}' is not lowercase"


# ══════════════════════════════════════════════════════════════════════════
#  Tests: _find_ancestor_distance
# ══════════════════════════════════════════════════════════════════════════

class TestFindAncestorDistance:
    """Tests for the ancestor distance calculation."""
    
    def test_direct_parent(self, matcher_with_hierarchy):
        """epithelial cell -> mammary gland epithelial cell = 1 hop."""
        hops = matcher_with_hierarchy._find_ancestor_distance(
            "CL:0000066", "CL:0002327", "cl"
        )
        assert hops == 1
    
    def test_grandparent(self, matcher_with_hierarchy):
        """cell -> mammary gland epithelial cell = 2 hops."""
        hops = matcher_with_hierarchy._find_ancestor_distance(
            "CL:0000000", "CL:0002327", "cl"
        )
        assert hops == 2
    
    def test_not_ancestor(self, matcher_with_hierarchy):
        """T cell is not ancestor of mammary gland epithelial cell."""
        hops = matcher_with_hierarchy._find_ancestor_distance(
            "CL:0000084", "CL:0002327", "cl"
        )
        assert hops is None
    
    def test_reversed_direction(self, matcher_with_hierarchy):
        """Descendant is not an ancestor."""
        hops = matcher_with_hierarchy._find_ancestor_distance(
            "CL:0002327", "CL:0000066", "cl"
        )
        assert hops is None
    
    def test_uberon_colon_hierarchy(self, matcher_with_hierarchy):
        """colon -> sigmoid colon = 1 hop."""
        hops = matcher_with_hierarchy._find_ancestor_distance(
            "UBERON:0001155", "UBERON:0001159", "uberon"
        )
        assert hops == 1
    
    def test_uberon_multi_hop(self, matcher_with_hierarchy):
        """large intestine -> sigmoid colon = 2 hops."""
        hops = matcher_with_hierarchy._find_ancestor_distance(
            "UBERON:0000059", "UBERON:0001159", "uberon"
        )
        assert hops == 2
    
    def test_missing_ontology(self, matcher_with_hierarchy):
        """Unknown ontology returns None."""
        hops = matcher_with_hierarchy._find_ancestor_distance(
            "X:001", "X:002", "nonexistent"
        )
        assert hops is None


# ══════════════════════════════════════════════════════════════════════════
#  Tests: _check_hierarchical_match
# ══════════════════════════════════════════════════════════════════════════

class TestCheckHierarchicalMatch:
    """Tests for the hierarchical matching method."""
    
    def test_parent_child_match(self, matcher_with_hierarchy):
        """epithelial cell vs mammary gland epithelial cell = HIERARCHICAL."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "epithelial cell", "mammary gland epithelial cell", "cell_type"
        )
        assert result is not None
        match_type, score = result
        assert match_type == "HIERARCHICAL"
        assert score == 0.90  # 1 hop
    
    def test_same_term_different_text(self, matcher_with_hierarchy):
        """Same ontology ID after normalization = HIERARCHICAL with score 0.90."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "colon", "colon", "organ"
        )
        assert result is not None
        assert result == ("HIERARCHICAL", 0.90)
    
    def test_uberon_parent_child(self, matcher_with_hierarchy):
        """colon vs sigmoid colon should match hierarchically."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "colon", "sigmoid colon", "organ"
        )
        assert result is not None
        assert result[0] == "HIERARCHICAL"
        assert result[1] == 0.90  # 1 hop
    
    def test_uberon_grandparent(self, matcher_with_hierarchy):
        """large intestine vs sigmoid colon = 2 hops."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "large intestine", "sigmoid colon", "organ"
        )
        assert result is not None
        assert result[0] == "HIERARCHICAL"
        assert result[1] == 0.87  # 2 hops
    
    def test_no_match_different_branches(self, matcher_with_hierarchy):
        """T cell vs epithelial cell = no hierarchical match."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "T cell", "epithelial cell", "cell_type"
        )
        # They share root "cell" but that's more than 4 hops for sibling relationship
        # Actually: T cell -> cell (1 hop) and epithelial cell -> cell (1 hop)
        # But _check_hierarchical_match checks if one is ancestor of the other,
        # not if they share a common ancestor
        assert result is None
    
    def test_unmapped_field_returns_none(self, matcher_with_hierarchy):
        """Field not in field_ontology_map returns None."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "some value", "other value", "instrument"
        )
        assert result is None
    
    def test_normalization_failure_returns_none(self, matcher_with_hierarchy):
        """Term that can't be normalized returns None."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "xyz_unknown_term", "abc_unknown_term", "cell_type"
        )
        assert result is None


# ══════════════════════════════════════════════════════════════════════════
#  Tests: Cell Line Fallback in Hierarchical Match
# ══════════════════════════════════════════════════════════════════════════

class TestCellLineFallback:
    """Tests for cell line → cell type fallback in hierarchical matching."""
    
    def test_mcf7_to_epithelial(self, matcher_with_hierarchy):
        """MCF-7 cell line should fall back to 'epithelial cell'."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "MCF-7", "mammary gland epithelial cell", "cell_type"
        )
        # MCF-7 isn't in the term_map, fallback should resolve to "epithelial cell"
        # Then epithelial cell -> mammary gland epithelial cell = 1 hop
        assert result is not None
        assert result[0] == "HIERARCHICAL"
    
    def test_jurkat_to_t_cell(self, matcher_with_hierarchy):
        """Jurkat cell line should fall back to 'T cell'."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "Jurkat", "T cell", "cell_type"
        )
        # Jurkat -> "T cell" via fallback, same ontology ID = 0.90
        assert result is not None
        assert result == ("HIERARCHICAL", 0.90)
    
    def test_hela_cells_suffix_stripping(self, matcher_with_hierarchy):
        """'HeLa cells' should strip suffix and look up 'hela'."""
        result = matcher_with_hierarchy._check_hierarchical_match(
            "HeLa cells", "epithelial cell", "cell_type"
        )
        # "HeLa cells" -> strip " cells" -> "hela" -> "epithelial cell"
        assert result is not None
        assert result == ("HIERARCHICAL", 0.90)


# ══════════════════════════════════════════════════════════════════════════
#  Tests: Label Normalization
# ══════════════════════════════════════════════════════════════════════════

class TestLabelNormalization:
    """Tests for _normalize_label_value()."""
    
    def test_tmt_6plex(self):
        from benchmark_data.run_sdrf_benchmark import _normalize_label_value
        result = _normalize_label_value("TMT126; TMT127; TMT128; TMT129; TMT130; TMT131")
        assert result == "TMT 6-plex"
    
    def test_tmt_10plex(self):
        from benchmark_data.run_sdrf_benchmark import _normalize_label_value
        channels = "; ".join([f"TMT{i}" for i in range(126, 136)])
        result = _normalize_label_value(channels)
        assert result == "TMT 10-plex"
    
    def test_itraq_4plex(self):
        from benchmark_data.run_sdrf_benchmark import _normalize_label_value
        result = _normalize_label_value("iTRAQ114; iTRAQ115; iTRAQ116; iTRAQ117")
        assert result == "iTRAQ 4-plex"
    
    def test_silac(self):
        from benchmark_data.run_sdrf_benchmark import _normalize_label_value
        result = _normalize_label_value("SILAC heavy; SILAC light")
        assert result == "SILAC"
    
    def test_label_free_unchanged(self):
        from benchmark_data.run_sdrf_benchmark import _normalize_label_value
        result = _normalize_label_value("label-free")
        assert result == "label-free"
    
    def test_none_passthrough(self):
        from benchmark_data.run_sdrf_benchmark import _normalize_label_value
        assert _normalize_label_value(None) is None
    
    def test_empty_passthrough(self):
        from benchmark_data.run_sdrf_benchmark import _normalize_label_value
        assert _normalize_label_value("") == ""


# ══════════════════════════════════════════════════════════════════════════
#  Tests: Full compare() with field_name
# ══════════════════════════════════════════════════════════════════════════

class TestCompareWithFieldName:
    """Tests for compare() with the field_name argument."""
    
    def test_exact_match_ignores_hierarchy(self, matcher_with_hierarchy):
        """Exact match should take precedence over hierarchical."""
        match_type, score = matcher_with_hierarchy.compare(
            "epithelial cell", "epithelial cell", field_name="cell_type"
        )
        assert match_type == "EXACT"
    
    def test_hierarchical_match_via_compare(self, matcher_with_hierarchy):
        """Parent-child should be caught by compare() when field_name is passed."""
        match_type, score = matcher_with_hierarchy.compare(
            "colon", "sigmoid colon", field_name="organ"
        )
        assert match_type == "HIERARCHICAL"
        assert score == 0.90
    
    def test_no_field_name_skips_hierarchy(self, matcher_with_hierarchy):
        """Without field_name, hierarchical matching is skipped."""
        match_type, score = matcher_with_hierarchy.compare(
            "colon", "sigmoid colon", field_name=None
        )
        # Should fall through to SEMANTIC or NO_MATCH
        assert match_type in ("SEMANTIC", "NO_MATCH")


# ══════════════════════════════════════════════════════════════════════════
#  Tests: Hierarchical Score Decrease with Hops
# ══════════════════════════════════════════════════════════════════════════

class TestHierarchicalScoring:
    """Verify scores decrease with hop distance."""
    
    def test_score_decreases_with_distance(self):
        from benchmark.semantic_matcher import HierarchicalMatcher
        scores = HierarchicalMatcher.HIERARCHICAL_SCORE_BY_HOPS
        assert scores[1] > scores[2] > scores[3] > scores[4]
    
    def test_all_scores_above_semantic_threshold(self):
        """Even at max distance, score should be above typical semantic threshold."""
        from benchmark.semantic_matcher import HierarchicalMatcher
        for hops, score in HierarchicalMatcher.HIERARCHICAL_SCORE_BY_HOPS.items():
            assert score >= 0.75, f"Score at {hops} hops ({score}) is below 0.75"
