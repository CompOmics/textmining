"""
Normalization Agent for the extraction pipeline.

This agent wraps the TermNormalizer to provide ontology-based
normalization of extracted metadata terms.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

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

    _ENTITY_ALLOWED_PREFIXES = {
        'species': ('NCBITaxon:',),
        'organism': ('NCBITaxon:',),
        'cell_type': ('CL:',),
        'cell_line': ('CLO:',),
        'tissue': ('UBERON:',),
        'organ': ('UBERON:',),
        'sample_source': ('BTO:',),
        'disease': ('MONDO:',),
        'disease_state': ('MONDO:',),
        'instrument': ('MS:',),
        'labelling': ('MS:',),
        'fractionation': ('MS:',),
        'modification': ('MOD:', 'UNIMOD:'),
        'ptm': ('MOD:', 'UNIMOD:'),
    }

    # Ancestors needed to evaluate the explicit in_taxon constraints commonly
    # present in UBERON/CL. This is deliberately conservative and covers only
    # species in the curated species ontology.
    _SPECIES_TAXON_ANCESTORS = {
        'NCBITaxon:9606': {'NCBITaxon:9606', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:10090': {'NCBITaxon:10090', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:10116': {'NCBITaxon:10116', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9913': {'NCBITaxon:9913', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9823': {'NCBITaxon:9823', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9031': {'NCBITaxon:9031', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:7955': {'NCBITaxon:7955', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:7227': {'NCBITaxon:7227', 'NCBITaxon:50557', 'NCBITaxon:33208'},
        'NCBITaxon:6239': {'NCBITaxon:6239', 'NCBITaxon:33208'},
        'NCBITaxon:9544': {'NCBITaxon:9544', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9598': {'NCBITaxon:9598', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9541': {'NCBITaxon:9541', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9615': {'NCBITaxon:9615', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9685': {'NCBITaxon:9685', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9986': {'NCBITaxon:9986', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9940': {'NCBITaxon:9940', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9796': {'NCBITaxon:9796', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:9825': {'NCBITaxon:9825', 'NCBITaxon:7742', 'NCBITaxon:33208'},
        'NCBITaxon:8355': {'NCBITaxon:8355', 'NCBITaxon:7742', 'NCBITaxon:33208'},
    }

    @classmethod
    def _taxon_ancestors(
        cls,
        species_id: str,
        graphs: Dict[str, Any],
    ) -> Optional[Set[str]]:
        """Resolve taxon ancestors from loaded ontology imports when possible.

        CL imports an NCBI Taxonomy subset with the intermediate nodes needed
        for constraints such as Gnathostomata and Chordata.  That hierarchy is
        more reliable than a short hand-maintained list.  The curated fallback
        remains useful for lightweight deployments and unit tests.
        """
        for graph in graphs.values():
            if graph.get_node(species_id) is not None:
                ancestors = set(graph.get_ancestors(species_id, max_depth=100))
                if ancestors:
                    return {species_id, *ancestors}
        curated = cls._SPECIES_TAXON_ANCESTORS.get(species_id)
        return set(curated) if curated else None
    
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

    @staticmethod
    def _entries(value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
        return []

    @staticmethod
    def _append_qc_flag(entry: Dict[str, Any], flag: str) -> None:
        flags = entry.setdefault('normalization_qc_flags', [])
        if flag not in flags:
            flags.append(flag)

    def _apply_result_metadata(
        self,
        entry: Dict[str, Any],
        result: NormalizationResult,
        entity_type: str,
    ) -> None:
        """Attach normalization result and auditable QC metadata to an entry."""
        entry['is_normalized'] = result.is_normalized
        if result.is_normalized:
            entry['ontology_id'] = result.ontology_id
            entry['ontology_name'] = result.ontology_name
            entry['similarity'] = round(result.similarity, 3)
        normalization_method = getattr(result, 'normalization_method', None)
        matched_text = getattr(result, 'matched_text', None)
        synonym_scope = getattr(result, 'synonym_scope', None)
        candidates = getattr(result, 'candidates', [])
        if normalization_method:
            entry['normalization_method'] = normalization_method
        if matched_text:
            entry['normalization_matched_text'] = matched_text
        if synonym_scope:
            entry['synonym_scope'] = synonym_scope
        if candidates and not result.is_normalized:
            entry['normalization_candidate_ids'] = list(dict.fromkeys(
                candidate[0] for candidate in candidates
            ))
        for flag in getattr(result, 'qc_flags', []):
            self._append_qc_flag(entry, flag)

        ontology_id = result.ontology_id
        allowed = self._ENTITY_ALLOWED_PREFIXES.get(entity_type)
        if ontology_id and allowed and not ontology_id.startswith(allowed):
            value_key = str(entry.get('value', '')).casefold().strip()
            # PATO normal is the conventional controlled value for a normal or
            # healthy disease-state entry, so it is an explicit exception.
            is_normal_state = (
                entity_type in {'disease', 'disease_state'}
                and ontology_id == 'PATO:0000461'
                and value_key in {'normal', 'healthy', 'control'}
            )
            if not is_normal_state:
                self._append_qc_flag(entry, 'cross_namespace_mapping')

    def _add_taxon_qc(self, record: Dict[str, Any]) -> None:
        """Flag anatomy mappings incompatible with a known extracted species."""
        if not self.normalizer:
            return
        species_ids = {
            entry.get('ontology_id')
            for field in ('species', 'organism')
            for entry in self._entries(record.get(field))
            if str(entry.get('ontology_id', '')).startswith('NCBITaxon:')
        }
        if not species_ids:
            return
        graphs = getattr(self.normalizer, 'graphs', {})
        uberon = graphs.get('uberon')
        if uberon is None:
            return
        ancestors_by_species = {
            species_id: self._taxon_ancestors(species_id, graphs)
            for species_id in species_ids
        }
        # Absence from the locally imported taxonomic subset is not evidence
        # of incompatibility. Only issue this QC flag when every extracted
        # species has a known lineage.
        if any(ancestors is None for ancestors in ancestors_by_species.values()):
            return
        for field in ('tissue', 'organ'):
            for entry in self._entries(record.get(field)):
                node = uberon.get_node(str(entry.get('ontology_id', '')))
                if node is None or not node.taxon_ids:
                    continue
                compatible = any(
                    constraint in ancestors_by_species[species_id]
                    for species_id in species_ids
                    for constraint in node.taxon_ids
                )
                if not compatible:
                    self._append_qc_flag(entry, 'taxon_incompatible_mapping')
                    entry['ontology_taxon_constraints'] = node.taxon_ids
    
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
        entry = {
            'value': term,
            'confidence': result.confidence,
        }
        self._apply_result_metadata(entry, result, entity_type)
        return entry
    
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
            if (isinstance(value, list) and value
                    and all(isinstance(item, dict) and 'value' in item for item in value)):
                entries = []
                for item in value:
                    term = str(item.get('value', ''))
                    evidence = str(item.get('evidence', ''))
                    entry = {
                        'value': term,
                        'evidence': evidence,
                        'is_normalized': False,
                    }
                    if term and term.lower() != 'unknown':
                        norm_result = self.normalizer.normalize(
                            term, entity_type=entity_type
                        )
                        self._apply_result_metadata(
                            entry, norm_result, entity_type
                        )
                    entries.append(entry)
                normalized[field] = entries
            elif isinstance(value, list) and len(value) >= 1:
                # Format: [value, evidence] or [value]
                term = value[0] if value else ""
                evidence = value[1] if len(value) > 1 else ""
                
                if term and term.lower() != "unknown":
                    norm_result = self.normalizer.normalize(term, entity_type=entity_type)

                    entry = {
                        'value': term,
                        'evidence': evidence,
                    }
                    self._apply_result_metadata(entry, norm_result, entity_type)
                    normalized[field] = entry
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

                    entry = {
                        'value': value,
                    }
                    self._apply_result_metadata(entry, norm_result, entity_type)
                    normalized[field] = entry
                else:
                    normalized[field] = {
                        'value': value,
                        'is_normalized': False,
                    }
            else:
                # Keep other formats as-is
                normalized[field] = value
        
        self._add_taxon_qc(normalized)
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
        
        normalized_results: Dict[str, Dict[str, Any]] = {
            filename: {} for filename in file_results
        }

        # Group normalizable values across the entire document batch by entity
        # type. Each group becomes one SapBERT embedding call instead of one
        # model invocation per field per document.
        grouped: Dict[str, List[tuple]] = {}

        for filename, extraction in file_results.items():
            logger.info(f"Queueing normalization: {filename}")
            destination = normalized_results[filename]

            for field, value in extraction.items():
                if field not in self.NORMALIZABLE_FIELDS:
                    destination[field] = value
                    continue

                entity_type = self.NORMALIZABLE_FIELDS[field]
                if (isinstance(value, list) and value
                        and all(isinstance(item, dict) and 'value' in item for item in value)):
                    destination[field] = [None] * len(value)
                    for index, item in enumerate(value):
                        term = str(item.get('value', ''))
                        evidence = str(item.get('evidence', ''))
                        if term and term.lower() != 'unknown':
                            grouped.setdefault(entity_type, []).append(
                                (filename, field, index, term, evidence, 'multi', value)
                            )
                        else:
                            destination[field][index] = {
                                'value': term,
                                'evidence': evidence,
                                'is_normalized': False,
                            }
                elif isinstance(value, list) and value:
                    term = value[0] if value else ""
                    evidence = value[1] if len(value) > 1 else ""
                    if term and str(term).lower() != "unknown":
                        grouped.setdefault(entity_type, []).append(
                            (filename, field, None, str(term), evidence, 'legacy', value)
                        )
                    else:
                        destination[field] = {
                            'value': term,
                            'evidence': evidence,
                            'is_normalized': False,
                        }
                elif isinstance(value, str):
                    if value and value.lower() != "unknown":
                        grouped.setdefault(entity_type, []).append(
                            (filename, field, None, value, None, 'string', value)
                        )
                    else:
                        destination[field] = {
                            'value': value,
                            'is_normalized': False,
                        }
                else:
                    destination[field] = value

        for entity_type, requests in grouped.items():
            terms = [request[3] for request in requests]
            logger.info(
                "Normalizing %d values as %s in one batch",
                len(terms),
                entity_type,
            )
            try:
                results = self.normalizer.normalize_batch(
                    terms,
                    entity_type=entity_type,
                )
            except Exception as exc:
                logger.error(
                    "Batch normalization failed for %s: %s",
                    entity_type,
                    exc,
                )
                for filename, field, _index, _term, _evidence, _kind, original in requests:
                    normalized_results[filename][field] = original
                continue

            for request, norm_result in zip(requests, results):
                filename, field, index, term, evidence, kind, _original = request
                entry: Dict[str, Any] = {
                    'value': term,
                }
                if kind in ('multi', 'legacy'):
                    entry['evidence'] = evidence
                self._apply_result_metadata(entry, norm_result, entity_type)
                if kind == 'multi':
                    normalized_results[filename][field][index] = entry
                else:
                    normalized_results[filename][field] = entry

        for record in normalized_results.values():
            self._add_taxon_qc(record)
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
