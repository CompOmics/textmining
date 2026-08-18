"""
Normalization Agent for the extraction pipeline.

This agent wraps the TermNormalizer to provide ontology-based
normalization of extracted metadata terms.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from normalization.normalizer import TermNormalizer, NormalizationResult
from normalization.config import NormalizationConfig
from core.field_mappings import FIELD_TO_ENTITY_TYPE

logger = logging.getLogger(__name__)


class NormalizationAgent:
    """
    Agent for normalizing extracted terms against ontologies.
    
    Integrates with the extraction pipeline to add ontology IDs
    and confidence scores to extracted metadata.
    
    Example:
        >>> agent = NormalizationAgent()
        >>> normalized = agent.normalize_batch({'file.txt': {'species': ['Homo sapiens', 'evidence']}})
    """
    
    # Fields that should be normalized and their entity types
    NORMALIZABLE_FIELDS = FIELD_TO_ENTITY_TYPE
    
    def __init__(self, 
                 ontology_dir: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 auto_load: bool = True):
        """
        Initialize the normalization agent.
        
        Args:
            ontology_dir: Directory containing ontology files
            cache_dir: Directory for cached indices
            auto_load: Automatically load ontologies on first use
        """
        # Set up paths
        base_dir = Path(__file__).parent.parent
        
        if ontology_dir:
            ont_dir = Path(ontology_dir)
        else:
            ont_dir = base_dir / "ontologies"
        
        if cache_dir:
            cch_dir = Path(cache_dir)
        else:
            cch_dir = base_dir / "ontology_cache"
        
        # Create config
        self.config = NormalizationConfig(
            ontology_dir=str(ont_dir),
            cache_dir=str(cch_dir),
            similarity_threshold=0.7,
            use_gpu=True,
        )
        
        self.normalizer: Optional[TermNormalizer] = None
        self._loaded = False
        self._auto_load = auto_load
        
        logger.info(f"NormalizationAgent initialized")
        logger.info(f"  Ontology dir: {ont_dir}")
        logger.info(f"  Cache dir: {cch_dir}")
    
    def _ensure_loaded(self) -> None:
        """Ensure ontologies are loaded."""
        if self._loaded:
            return
        
        if not self._auto_load:
            raise RuntimeError("Ontologies not loaded. Call load_ontologies() first.")
        
        self.load_ontologies()
    
    def load_ontologies(self) -> None:
        """Load all configured ontologies."""
        if self._loaded:
            logger.info("Ontologies already loaded")
            return
        
        logger.info("Loading ontologies for normalization...")
        
        self.normalizer = TermNormalizer(self.config)
        self.normalizer.load_all_ontologies()
        
        self._loaded = True
        
        # Log what was loaded
        loaded_count = len(self.normalizer.indices)
        logger.info(f"Loaded {loaded_count} ontology indices")
    
    def normalize_term(self, 
                       term: str, 
                       entity_type: str) -> Dict[str, Any]:
        """
        Normalize a single term.
        
        Args:
            term: Term to normalize
            entity_type: Type of entity (species, cell_type, etc.)
            
        Returns:
            Dictionary with normalization result
        """
        self._ensure_loaded()
        
        result = self.normalizer.normalize(term, entity_type=entity_type)
        
        return {
            'value': term,
            'ontology_id': result.ontology_id,
            'ontology_name': result.ontology_name,
            'similarity': result.similarity,
            'is_normalized': result.is_normalized,
            'confidence': result.confidence,
        }
    
    def normalize_extraction(self, 
                            extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all normalizable fields in an extraction result.
        
        Args:
            extraction: Dictionary with extracted fields
            
        Returns:
            Dictionary with normalized fields
        """
        self._ensure_loaded()
        
        normalized = {}
        
        for field, value in extraction.items():
            if field not in self.NORMALIZABLE_FIELDS:
                # Keep non-normalizable fields as-is
                normalized[field] = value
                continue
            
            entity_type = self.NORMALIZABLE_FIELDS[field]
            
            # Handle different value formats
            if isinstance(value, list) and len(value) >= 1:
                # Format: [value, evidence] or [value]
                term = value[0] if value else ""
                evidence = value[1] if len(value) > 1 else ""
                
                if term and term.lower() != "unknown":
                    norm_result = self.normalizer.normalize(term, entity_type=entity_type)
                    
                    normalized[field] = {
                        'value': term,
                        'evidence': evidence,
                        'ontology_id': norm_result.ontology_id,
                        'ontology_name': norm_result.ontology_name,
                        'similarity': round(norm_result.similarity, 3),
                        'is_normalized': norm_result.is_normalized,
                    }
                else:
                    normalized[field] = {
                        'value': term,
                        'evidence': evidence,
                        'is_normalized': False,
                    }
            elif isinstance(value, str):
                # Simple string value
                if value and value.lower() != "unknown":
                    norm_result = self.normalizer.normalize(value, entity_type=entity_type)
                    
                    normalized[field] = {
                        'value': value,
                        'ontology_id': norm_result.ontology_id,
                        'ontology_name': norm_result.ontology_name,
                        'similarity': round(norm_result.similarity, 3),
                        'is_normalized': norm_result.is_normalized,
                    }
                else:
                    normalized[field] = {
                        'value': value,
                        'is_normalized': False,
                    }
            else:
                # Keep other formats as-is
                normalized[field] = value
        
        return normalized
    
    def normalize_batch(self, 
                        file_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Normalize a batch of extraction results.
        
        This is the main method called by the pipeline.
        
        Args:
            file_results: Dictionary mapping filename to extraction result
            
        Returns:
            Dictionary mapping filename to normalized result
        """
        self._ensure_loaded()
        
        normalized_results = {}
        
        for filename, extraction in file_results.items():
            logger.info(f"Normalizing: {filename}")
            
            try:
                normalized = self.normalize_extraction(extraction)
                normalized_results[filename] = normalized
            except Exception as e:
                logger.error(f"Error normalizing {filename}: {e}")
                # Return original on error
                normalized_results[filename] = extraction
        
        return normalized_results
    
    def register_synonym(self,
                         synonym: str,
                         node_name: str,
                         entity_type: str) -> bool:
        """
        Register a new synonym for an ontology term at runtime.

        Convenience wrapper around ``TermNormalizer.register_synonym``.
        Determines the ontology from ``entity_type`` automatically.

        Args:
            synonym:     The abbreviated / variant name to register
                         (e.g. ``"p.falciparum"``).
            node_name:   The primary name of the ontology node
                         (e.g. ``"Plasmodium falciparum"``).
            entity_type: Entity type string used to resolve the ontology
                         (e.g. ``"species"``, ``"cell_type"``).

        Returns:
            ``True`` on success, ``False`` if the node was not found.

        Example::

            agent.register_synonym(
                synonym="p.falciparum",
                node_name="Plasmodium falciparum",
                entity_type="species",
            )
        """
        self._ensure_loaded()
        ontology_id = self.normalizer.get_ontology_for_entity(entity_type)
        if not ontology_id:
            logger.warning(
                f"register_synonym: no ontology mapped for entity_type '{entity_type}'"
            )
            return False
        return self.normalizer.register_synonym(synonym, node_name, ontology_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded ontologies."""
        if not self._loaded or not self.normalizer:
            return {'loaded': False}
        
        stats = {
            'loaded': True,
            'num_ontologies': len(self.normalizer.indices),
            'ontologies': {},
        }
        
        for ont_id, index in self.normalizer.indices.items():
            stats['ontologies'][ont_id] = {
                'num_terms': len(index.term_ids) if hasattr(index, 'term_ids') else 0,
            }
        
        return stats
