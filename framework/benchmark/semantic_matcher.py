#!/usr/bin/env python3
"""
Semantic Matching Module using SapBERT embeddings.

Provides hierarchical matching strategy for comparing LLM output to golden annotations:
1. Exact match
2. Normalized match (case-insensitive)
3. Ontology match (via existing normalization)
4. Hierarchical match (parent-child in ontology hierarchy)
5. Semantic match (SapBERT embedding similarity)

The semantic tier used SciBERT (allenai/scibert_scivocab_uncased) until
2026-09-04. It was replaced because SciBERT scores topical relatedness rather
than synonymy on short domain terms: on 88,226 value pairs drawn from
genuinely unrelated annotation fields (Organism against CleavageAgent and
similar), where no true match is possible, SciBERT had a median cosine of
0.631 and 18.2% of pairs reached the 0.70 threshold. SapBERT, trained on UMLS
synonymy, never exceeded 0.49 on the same control. SapBERT is also what
framework/normalization already uses, so the whole pipeline now shares one
embedding model.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from functools import lru_cache

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Cell Line → Cell Type Fallback Lookup
#  Used when SapBERT normalization can't resolve a cell line name
#  to a cell type in the Cell Ontology.
# ═══════════════════════════════════════════════════════════════════════

CELL_LINE_TO_CELL_TYPE = {
    # Human epithelial
    "hela": "epithelial cell",
    "mcf-7": "epithelial cell",
    "mcf7": "epithelial cell",
    "mda-mb-231": "epithelial cell",
    "a549": "epithelial cell",
    "hct116": "epithelial cell",
    "caco-2": "epithelial cell",
    "caco2": "epithelial cell",
    "pc-3": "epithelial cell",
    "pc3": "epithelial cell",
    "lncap": "epithelial cell",
    "vero": "epithelial cell",
    "vero e6": "epithelial cell",
    "hek293": "epithelial cell",
    "hek-293": "epithelial cell",
    "293t": "epithelial cell",
    "hek293t": "epithelial cell",
    # Immune / blood
    "jurkat": "T cell",
    "k562": "myeloid cell",
    "thp-1": "monocyte",
    "thp1": "monocyte",
    "u937": "monocyte",
    "raw264.7": "macrophage",
    "raw 264.7": "macrophage",
    # Neuronal
    "sh-sy5y": "neuron",
    "shsy5y": "neuron",
    "neuro2a": "neuron",
    "n2a": "neuron",
    # Fibroblast
    "nih3t3": "fibroblast",
    "nih-3t3": "fibroblast",
    "mef": "fibroblast",
    "wi-38": "fibroblast",
    "3t3-l1": "fibroblast",
    # Muscle
    "c2c12": "myoblast",
    # Osteosarcoma
    "u2os": "osteoblast",
    "saos-2": "osteoblast",
    # Hepatocyte
    "hepg2": "hepatocyte",
    "hep-g2": "hepatocyte",
    "huh-7": "hepatocyte",
    "huh7": "hepatocyte",
    # Melanoma
    "a375": "melanocyte",
    "b16": "melanocyte",
    # Pancreatic
    "panc-1": "epithelial cell",
    "bxpc-3": "epithelial cell",
    # Stem cells
    "h9": "embryonic stem cell",
    "h1": "embryonic stem cell",
}


class SemanticMatcher:
    """SapBERT-based semantic similarity matcher."""
    
    def __init__(
        self, 
        model_name: str = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        threshold: float = 0.70,
        device: str = None
    ):
        """
        Initialize the semantic matcher.
        
        Args:
            model_name: HuggingFace model name (default: SapBERT)
            threshold: Similarity threshold for semantic match (0-1).
                0.70 is the project-wide "accepted as normalised" cutoff used
                in framework/normalization, reused here rather than tuned
                separately.
            device: 'cuda', 'cpu', or None for auto-detect
        """
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = device
        self._torch_device = None
        self._cache: Dict[str, np.ndarray] = {}
    
    @property
    def model(self):
        """Lazy load the tokenizer and model.

        Uses transformers directly with [CLS] pooling rather than
        SentenceTransformer mean pooling, so this tier embeds text exactly the
        way framework/normalization/index.py does. Mixing pooling strategies
        would put the same string at two different points in the space.
        """
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = self._device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._torch_device = torch.device(device)

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(self._torch_device)
            model.eval()
            self._model = model
            logger.info("Loaded semantic model: %s on %s", self.model_name, self._torch_device)
        return self._model
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string (cached).

        [CLS] token, L2-normalised, max 128 tokens: the same encoding as
        framework/normalization/index.py._embed_texts.
        """
        if not text:
            return np.zeros(768)  # SapBERT embedding dimension
        
        text_key = text.lower().strip()
        if text_key not in self._cache:
            import torch

            model = self.model  # triggers lazy load of tokenizer too
            inputs = self._tokenizer(
                [text_key],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt',
            )
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}
            with torch.no_grad():
                cls = model(**inputs).last_hidden_state[:, 0, :]
                cls = torch.nn.functional.normalize(cls, dim=1)
            self._cache[text_key] = cls[0].cpu().numpy()
        return self._cache[text_key]
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Embeddings are already normalized, so dot product = cosine similarity
        similarity = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def is_semantic_match(self, text1: str, text2: str) -> Tuple[bool, float]:
        """
        Check if two texts are semantically similar.
        
        Returns:
            Tuple of (is_match, similarity_score)
        """
        similarity = self.cosine_similarity(text1, text2)
        return similarity >= self.threshold, similarity


class HierarchicalMatcher:
    """
    Hierarchical matching strategy that tries multiple matching levels.
    
    Levels (in order of precedence):
    1. EXACT: Identical strings
    2. NORMALIZED: Same after normalization
    3. ONTOLOGY: Same ontology term resolution
    4. HIERARCHICAL: Parent-child relationship in ontology
    5. SEMANTIC: SapBERT similarity above threshold
    """
    
    MATCH_SCORES = {
        'EXACT': 1.0,
        'NORMALIZED': 0.95,
        'ONTOLOGY': 0.90,
        'HIERARCHICAL': None,  # Varies by hop distance
        'SEMANTIC': None,  # Uses actual similarity score
        'NO_MATCH': 0.0
    }
    
    # Score decreases with hop distance in the ontology hierarchy
    HIERARCHICAL_SCORE_BY_HOPS = {
        1: 0.90,
        2: 0.87,
        3: 0.83,
        4: 0.80,
    }
    
    # Field mapping: golden_field -> [llm_field alternatives]
    # Maps golden set field names to equivalent LLM output field names
    FIELD_MAPPING = {
        # ExperimentalDesignAgent mappings
        'replicates': ['number_of_biological_replicates', 'number_of_technical_replicates', 
                       'biological_replicate', 'technical_replicate'],
        'number_of_replicates': ['number_of_biological_replicates', 'number_of_technical_replicates'],
        
        # TechnicalAgent mappings  
        'missed_cleavages': ['missed_cleavage', 'max_missed_cleavages'],
        'cleavage_agent': ['cleavage_reagent', 'protease', 'enzyme'],
        'fragmentation': ['fragmentation_method', 'fragmentation_type'],
        'ionization': ['ionization_type', 'ionization_mode'],
        'instrument': ['mass_spectrometer', 'instrument_model'],
        'label': ['labeling', 'labeling_method', 'quantification_method'],
        'ptm': ['modification', 'variable_modification', 'fixed_modification'],
        'precursor_tolerance': ['precursor_mass_tolerance', 'ms1_tolerance'],
        'reduction_reagent': ['reducing_agent', 'reduction_agent'],
        
        # BiologicalAgent mappings
        'strain': ['cell_line', 'strain_name'],
        'material_type': ['sample_type', 'material'],
    }
    
    def __init__(
        self, 
        semantic_threshold: float = 0.75,
        use_ontology: bool = False,
        normalizer: Any = None,
        use_field_mapping: bool = True,
        term_normalizer: Any = None,
        field_ontology_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize hierarchical matcher.
        
        Args:
            semantic_threshold: Threshold for semantic matching
            use_ontology: Whether to use ontology normalization
            normalizer: Normalization module instance (optional)
            use_field_mapping: Whether to use field name mapping
            term_normalizer: TermNormalizer instance for hierarchical matching
            field_ontology_map: Maps field names to ontology IDs
                e.g. {"cell_type": "cl", "organ": "uberon", "disease": "doid"}
        """
        self.semantic_matcher = SemanticMatcher(threshold=semantic_threshold)
        self.use_ontology = use_ontology
        self.normalizer = normalizer
        self.use_field_mapping = use_field_mapping
        self.term_normalizer = term_normalizer
        self.field_ontology_map = field_ontology_map or {}
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        import re
        # Remove trademark/registered/copyright symbols common in instrument names
        normalized = re.sub(r'[™®©]', '', str(text))
        return normalized.lower().strip()
    
    def is_null_value(self, value: Any) -> bool:
        """Check if a value should be treated as null."""
        if value is None:
            return True
        if isinstance(value, str):
            cleaned = value.lower().strip()
            # Treat these as null
            null_values = [
                '', 'null', 'unknown', 'n/a', 'none', 'not available', 
                'not specified', 'not provided', 'na', 'n.a.', 'n.a',
                "['unknown', '']", '["unknown", ""]'  # Common list format
            ]
            return cleaned in null_values
        if isinstance(value, list):
            # Empty list or list of nulls
            non_null = [v for v in value if v and str(v).lower().strip() not in ['', 'unknown', 'none']]
            return len(non_null) == 0
        return False
    
    def clean_value(self, value: str) -> str:
        """Clean extracted value - normalize format, remove evidence strings."""
        if not value:
            return ""
        
        val = str(value).strip()
        
        # Handle list format like "['male', '']" or "['value', 'evidence...']"
        if val.startswith('[') and val.endswith(']'):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Take first non-empty, non-evidence element
                    for item in parsed:
                        if item and len(str(item)) < 100:  # Avoid evidence strings
                            cleaned = str(item).strip()
                            if cleaned.lower() not in ['unknown', 'none', '']:
                                return cleaned
                    return ""
            except:
                pass
        
        # Remove common list artifacts
        val = val.strip("[]'\"")
        
        # Truncate if it looks like evidence was appended
        if len(val) > 100 and ', ' in val:
            val = val.split(', ')[0]
        
        return val.strip()
    
    def extract_value(self, field_data: Any) -> Optional[str]:
        """Extract the actual value from a field (handles dict format)."""
        if field_data is None:
            return None
        if isinstance(field_data, dict):
            # Prefer 'resolved' (from ontology normalization), then 'value'
            resolved = field_data.get('resolved')
            if resolved and not self.is_null_value(resolved):
                return self.clean_value(str(resolved))
            value = field_data.get('value')
            if value:
                return self.clean_value(str(value))
            return None
        if isinstance(field_data, list):
            # Take first non-null item
            for item in field_data:
                if item and not self.is_null_value(item):
                    return self.clean_value(str(item))
            return None
        return self.clean_value(str(field_data))
    
    def get_mapped_value(self, llm_output: dict, golden_field: str) -> Optional[str]:
        """
        Get LLM value for a golden field, checking mapped field names.
        
        Args:
            llm_output: Full LLM output dict
            golden_field: Field name from golden set
            
        Returns:
            Extracted value from LLM output, checking mapped fields if needed
        """
        # First try exact field name
        if golden_field in llm_output:
            val = self.extract_value(llm_output[golden_field])
            if val and not self.is_null_value(val):
                return val
        
        # Try mapped field names
        if self.use_field_mapping and golden_field in self.FIELD_MAPPING:
            for alt_field in self.FIELD_MAPPING[golden_field]:
                if alt_field in llm_output:
                    val = self.extract_value(llm_output[alt_field])
                    if val and not self.is_null_value(val):
                        return val
        
        return None
    
    def compare_with_field(self, llm_output: dict, golden_field: str, golden_value: Any) -> Tuple[str, float]:
        """
        Compare LLM output against golden value, using field mapping.
        
        Args:
            llm_output: Full LLM output dict
            golden_field: Field name from golden set
            golden_value: Golden value for the field
            
        Returns:
            Tuple of (match_type, score)
        """
        # Get value from LLM, with field mapping
        llm_val = self.get_mapped_value(llm_output, golden_field)
        golden_val = str(golden_value) if golden_value is not None else None
        
        # Handle null cases
        llm_is_null = self.is_null_value(llm_val)
        golden_is_null = self.is_null_value(golden_val)
        
        if golden_is_null and llm_is_null:
            return 'EXACT', 1.0
        elif golden_is_null and not llm_is_null:
            return 'NO_MATCH', 0.0
        elif not golden_is_null and llm_is_null:
            return 'NO_MATCH', 0.0
        
        # Use normal comparison
        return self._compare_values(llm_val, golden_val)
    
    def _compare_values(self, llm_val: str, golden_val: str, field_name: str = None) -> Tuple[str, float]:
        """Compare two string values using hierarchical matching."""
        # Level 1: Exact match
        if llm_val == golden_val:
            return 'EXACT', 1.0
        
        # Level 2: Normalized match
        llm_norm = self.normalize_text(llm_val)
        golden_norm = self.normalize_text(golden_val)
        if llm_norm == golden_norm:
            return 'NORMALIZED', 0.95
        
        # Level 3: Ontology match (if enabled)
        if self.use_ontology and self.normalizer:
            try:
                llm_resolved = self.normalizer.normalize(llm_val)
                golden_resolved = self.normalizer.normalize(golden_val)
                if llm_resolved and golden_resolved:
                    if llm_resolved.get('term_id') == golden_resolved.get('term_id'):
                        return 'ONTOLOGY', 0.90
            except Exception:
                pass
        
        # Level 4: Hierarchical match (parent-child in ontology)
        if self.term_normalizer and field_name and field_name in self.field_ontology_map:
            match_result = self._check_hierarchical_match(
                llm_val, golden_val, field_name
            )
            if match_result is not None:
                return match_result
        
        # Level 5: Semantic match
        is_match, similarity = self.semantic_matcher.is_semantic_match(llm_val, golden_val)
        if is_match:
            return 'SEMANTIC', similarity
        
        return 'NO_MATCH', similarity
    
    @staticmethod
    def _split_multi_value(value: str) -> List[str]:
        """Split a semicolon-separated multi-value string into individual items.

        Semicolons inside parentheses are NOT treated as separators —
        e.g. 'SILAC (light: Lys/Arg; heavy: Lys/Arg)' stays as one item.
        """
        if not value:
            return []
        # Only split on semicolons that are outside parentheses
        import re
        items = []
        depth = 0
        current = []
        for ch in value:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ';' and depth == 0:
                part = ''.join(current).strip()
                if part:
                    items.append(part)
                current = []
            else:
                current.append(ch)
        part = ''.join(current).strip()
        if part:
            items.append(part)
        return items if items else [value]

    @staticmethod
    def _normalize_llm_separators(value: str) -> str:
        """Normalise common LLM multi-value separators to semicolons.

        LLMs often use ', ' or ' / ' where the golden standard uses '; '.
        Slash normalisation is limited to ' / ' (space-slash-space) to avoid
        splitting compound names like 'Lys/Arg' or file paths.
        """
        import re
        normalized = re.sub(r',\s+', '; ', value)          # 'A, B' -> 'A; B'
        normalized = re.sub(r'\s+/\s+', '; ', normalized)  # 'A / B' -> 'A; B'
        # Also handle 'A/B' when both sides look like reagent names (word chars only)
        normalized = re.sub(r'(?<=[A-Za-z0-9])/(?=[A-Za-z])', '; ', normalized)
        return normalized

    def _compare_multi_value(self, llm_items: List[str], golden_items: List[str], field_name: str = None) -> Tuple[str, float]:
        """
        Set-based recall scoring for multi-value fields.

        For each golden item, find the best match among all LLM items.
        Score = mean of per-golden best scores (recall-weighted).
        Match type reflects the best type achieved when recall > 0.
        """
        MATCH_RANK = {'EXACT': 5, 'NORMALIZED': 4, 'ONTOLOGY': 3, 'HIERARCHICAL': 2, 'SEMANTIC': 1, 'NO_MATCH': 0}

        best_scores = []
        best_types = []

        for g_item in golden_items:
            best_type, best_score = 'NO_MATCH', 0.0
            for l_item in llm_items:
                m_type, m_score = self._compare_values(l_item, g_item, field_name)
                if MATCH_RANK[m_type] > MATCH_RANK[best_type]:
                    best_type, best_score = m_type, m_score
                elif MATCH_RANK[m_type] == MATCH_RANK[best_type] and m_score > best_score:
                    best_score = m_score
            best_scores.append(best_score)
            best_types.append(best_type)

        mean_score = sum(best_scores) / len(best_scores)

        # For recall, only count precise matches (EXACT/NORMALIZED/ONTOLOGY/HIERARCHICAL).
        # SEMANTIC matches between distinct anatomy/biology terms are too imprecise
        # to say the golden item was actually "found" by the LLM.
        PRECISE_TYPES = {'EXACT', 'NORMALIZED', 'ONTOLOGY', 'HIERARCHICAL'}
        precise_matched = [t for t in best_types if t in PRECISE_TYPES]
        recall = len(precise_matched) / len(golden_items)

        if recall == 0.0:
            return 'NO_MATCH', round(mean_score, 6)

        # Full precise recall: use the best individual match type achieved
        if recall == 1.0:
            best_achieved = max(best_types, key=lambda t: MATCH_RANK[t])
            return best_achieved, round(mean_score, 6)

        # Partial recall: report as SEMANTIC with recall-weighted mean score
        return 'SEMANTIC', round(mean_score, 6)

    def compare(self, llm_value: Any, golden_value: Any, field_name: str = None) -> Tuple[str, float]:
        """
        Compare LLM output value against golden annotation.

        Supports multi-value fields (semicolon-separated): if either golden or
        predicted contains multiple values, scoring is recall-based — each golden
        item is matched against all predicted items and the mean best score is returned.

        Args:
            llm_value: Value from LLM output (may be dict with 'value'/'resolved')
            golden_value: Value from golden annotation
            field_name: Optional field name for hierarchical ontology matching

        Returns:
            Tuple of (match_type, score)
            match_type: 'EXACT', 'NORMALIZED', 'ONTOLOGY', 'HIERARCHICAL', 'SEMANTIC', 'NO_MATCH'
            score: Confidence score (0-1)
        """
        # Extract actual values
        llm_val = self.extract_value(llm_value)
        golden_val = str(golden_value) if golden_value is not None else None

        # Handle null cases
        llm_is_null = self.is_null_value(llm_val)
        golden_is_null = self.is_null_value(golden_val)

        if golden_is_null and llm_is_null:
            return 'EXACT', 1.0
        elif golden_is_null and not llm_is_null:
            return 'NO_MATCH', 0.0
        elif not golden_is_null and llm_is_null:
            return 'NO_MATCH', 0.0

        # Multi-value: if golden contains ";" delegate to set-based scoring.
        # Normalise common LLM separators (, ) to ; before splitting.
        golden_items = self._split_multi_value(golden_val)
        if len(golden_items) > 1:
            llm_normalised = self._normalize_llm_separators(llm_val)
            llm_items = self._split_multi_value(llm_normalised)
            return self._compare_multi_value(llm_items, golden_items, field_name)
        # Also handle LLM returning multiple values when golden is single
        llm_items = self._split_multi_value(llm_val)
        if len(llm_items) > 1:
            return self._compare_multi_value(llm_items, golden_items, field_name)

        # Single-value path
        return self._compare_values(llm_val, golden_val, field_name)
    
    def _check_hierarchical_match(
        self, llm_val: str, golden_val: str, field_name: str
    ) -> Optional[Tuple[str, float]]:
        """
        Check if two values have a parent-child relationship in an ontology.
        
        Uses SapBERT-based TermNormalizer to resolve free-text values to
        ontology IDs, then checks ancestor/descendant relationships.
        
        For cell_type fields, falls back to a hardcoded cell line → cell type
        lookup when SapBERT normalization doesn't resolve the LLM value.
        
        Args:
            llm_val: Value from LLM output
            golden_val: Value from golden annotation
            field_name: Benchmark field name (used to select ontology)
            
        Returns:
            Tuple of ('HIERARCHICAL', score) if match found, None otherwise
        """
        ontology_id = self.field_ontology_map.get(field_name)
        if not ontology_id:
            return None
        
        try:
            # Normalize both values to ontology IDs via SapBERT
            llm_result = self.term_normalizer.normalize(
                llm_val, ontology_id=ontology_id
            )
            golden_result = self.term_normalizer.normalize(
                golden_val, ontology_id=ontology_id
            )
            
            llm_id = llm_result.ontology_id if llm_result.is_normalized else None
            golden_id = golden_result.ontology_id if golden_result.is_normalized else None
            
            # Cell type fallback: if LLM value looks like a cell line name,
            # use the hardcoded lookup to get the cell type, then re-normalize
            if field_name == "cell_type" and not llm_id:
                cell_type_name = CELL_LINE_TO_CELL_TYPE.get(llm_val.lower().strip())
                # Also try without trailing "cells" / "cell"
                if not cell_type_name:
                    cleaned = llm_val.lower().strip()
                    for suffix in [" cells", " cell"]:
                        if cleaned.endswith(suffix):
                            cleaned = cleaned[:-len(suffix)].strip()
                            break
                    cell_type_name = CELL_LINE_TO_CELL_TYPE.get(cleaned)
                
                if cell_type_name:
                    logger.debug(
                        f"Cell line fallback: '{llm_val}' → '{cell_type_name}'"
                    )
                    llm_result = self.term_normalizer.normalize(
                        cell_type_name, ontology_id=ontology_id
                    )
                    llm_id = llm_result.ontology_id if llm_result.is_normalized else None
            
            if not llm_id or not golden_id:
                return None
            
            # If same ID after normalization, it's an ONTOLOGY-level match
            if llm_id == golden_id:
                return ('HIERARCHICAL', 0.90)
            
            # Check if LLM value is an ancestor of golden value
            hops = self._find_ancestor_distance(llm_id, golden_id, ontology_id)
            if hops is not None:
                score = self.HIERARCHICAL_SCORE_BY_HOPS.get(hops, 0.80)
                logger.debug(
                    f"Hierarchical match: '{llm_val}'({llm_id}) is ancestor of "
                    f"'{golden_val}'({golden_id}) at {hops} hops → score={score}"
                )
                return ('HIERARCHICAL', score)
            
            # Check if golden value is an ancestor of LLM value
            hops = self._find_ancestor_distance(golden_id, llm_id, ontology_id)
            if hops is not None:
                score = self.HIERARCHICAL_SCORE_BY_HOPS.get(hops, 0.80)
                logger.debug(
                    f"Hierarchical match: '{golden_val}'({golden_id}) is ancestor of "
                    f"'{llm_val}'({llm_id}) at {hops} hops → score={score}"
                )
                return ('HIERARCHICAL', score)
            
        except Exception as e:
            logger.debug(f"Hierarchical matching error for '{llm_val}' vs '{golden_val}': {e}")
        
        return None
    
    def _find_ancestor_distance(
        self, potential_ancestor_id: str, descendant_id: str, ontology_id: str
    ) -> Optional[int]:
        """
        Check if potential_ancestor_id is an ancestor of descendant_id.
        
        Returns hop count if ancestor is found (1-4), None otherwise.
        """
        graph = self.term_normalizer.graphs.get(ontology_id)
        if not graph:
            return None
        
        # Walk up from descendant, checking each level
        visited = set()
        current_ids = [descendant_id]
        
        for hop in range(1, 5):  # Max 4 hops
            next_ids = []
            for node_id in current_ids:
                if node_id in visited:
                    continue
                visited.add(node_id)
                node = graph.get_node(node_id)
                if node and node.parents:
                    for parent_id in node.parents:
                        if parent_id == potential_ancestor_id:
                            return hop
                        next_ids.append(parent_id)
            current_ids = next_ids
            if not current_ids:
                break
        
        return None


def calculate_weighted_metrics(results: list) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 from comparison results.
    
    Uses proper TP/FP/FN counting:
    - TP: LLM produced a value AND it matches (score > 0.5 or match_type != NO_MATCH)
    - FP: LLM produced a value but it doesn't match
    - FN: Golden has a value but LLM didn't produce a match
    
    Args:
        results: List of dicts with 'score', 'golden_has_value', 'llm_has_value'
        
    Returns:
        Dict with weighted_precision, weighted_recall, weighted_f1
    """
    if not results:
        return {'weighted_precision': 0.0, 'weighted_recall': 0.0, 'weighted_f1': 0.0}
    
    # Use a threshold to determine if a match is "correct"
    MATCH_THRESHOLD = 0.5
    
    tp = 0  # True Positives: LLM has value AND score >= threshold
    fp = 0  # False Positives: LLM has value BUT score < threshold (or no golden)
    fn = 0  # False Negatives: Golden has value BUT no match from LLM
    
    for r in results:
        llm_has_value = r.get('llm_has_value', False)
        golden_has_value = r.get('golden_has_value', False)
        score = r.get('score', 0.0)
        
        is_match = score >= MATCH_THRESHOLD
        
        if llm_has_value and golden_has_value:
            if is_match:
                tp += 1
            else: 
                fp += 1
                fn += 1
        elif llm_has_value and not golden_has_value:
            # LLM produced value but golden is empty - this could be counted as FP
            # but often we ignore these for bio extraction
            pass
        elif not llm_has_value and golden_has_value:
            fn += 1
        # If neither has value, it's a true negative (not counted in P/R)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1
    }


# Convenience function for quick comparison
def semantic_compare(llm_value: str, golden_value: str, threshold: float = 0.75) -> Tuple[str, float]:
    """Quick semantic comparison without full matcher initialization."""
    matcher = HierarchicalMatcher(semantic_threshold=threshold)
    return matcher.compare(llm_value, golden_value)


if __name__ == '__main__':
    # Quick test
    matcher = HierarchicalMatcher(semantic_threshold=0.70)
    
    test_pairs = [
        ("mouse", "Mus musculus"),
        ("brain", "prefrontal cortex"),
        ("human", "Homo sapiens"),
        ("LC-MS/MS", "liquid chromatography tandem mass spectrometry"),
        ("liver", "kidney"),
        ("male", "Male"),
        ("HeLa cells", "HeLa"),
    ]
    
    print("Testing semantic matching:\n")
    for llm_val, golden_val in test_pairs:
        match_type, score = matcher.compare(llm_val, golden_val)
        print(f"  '{llm_val}' vs '{golden_val}'")
        print(f"    → {match_type}: {score:.3f}")
        print()
