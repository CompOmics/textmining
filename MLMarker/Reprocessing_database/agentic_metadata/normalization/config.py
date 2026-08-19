"""
Configuration for normalization module.

All vocabularies are custom curated text files — no external ontology downloads needed.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for the normalization pipeline."""

    ontology_dir: str = "ontologies/"
    cache_dir: str = "ontology_cache/"
    embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    similarity_threshold: float = 0.7
    top_k: int = 5
    use_synonyms: bool = True
    use_gpu: bool = True
    index_backend: str = "faiss"
    use_quantization: bool = True
    batch_size: int = 64
    term_aliases: Dict[str, str] = field(default_factory=dict)

    # Vocabulary files — all custom curated, one canonical term per line
    ontology_files: Dict[str, str] = field(default_factory=lambda: {
        'organisms': 'organisms.txt',
        'tissues': 'tissues.txt',
        'diseases': 'diseases.txt',
        'cell_parts': 'cell_parts.txt',
        'cell_lines': 'cell_lines.txt',
        'instruments': 'instruments.txt',
        'fragmentation': 'fragmentation.txt',
        'acquisition': 'acquisition.txt',
        'ionization': 'ionization.txt',
        'labeling': 'labeling.txt',
        'modifications': 'modifications.txt',
        'enzymes': 'enzymes.txt',
        'lc_column': 'lc_column.txt',
        'treatments': 'treatments.txt',
    })

    # Field name → vocabulary mapping (matches schemas.py field names)
    entity_ontology_map: Dict[str, str] = field(default_factory=lambda: {
        # Biology
        'organism': 'organisms',
        'tissue': 'tissues',
        'disease': 'diseases',
        'cell_part': 'cell_parts',
        # cell_line uses exact matching, not SapBERT (see normalize_extraction.py)
        # MS configuration
        'instrument': 'instruments',
        'fragmentation': 'fragmentation',
        'acquisition': 'acquisition',
        'ionization': 'ionization',
        'labeling': 'labeling',
        # Sample prep
        'modifications': 'modifications',
        'enzymes': 'enzymes',
        'lc_column': 'lc_column',
        # Treatments
        'treatment_class': 'treatments',
    })

    # Human-readable names for logging
    ontology_display_names: Dict[str, str] = field(default_factory=lambda: {
        'organisms': 'Organisms',
        'tissues': 'Tissues',
        'diseases': 'Diseases',
        'cell_parts': 'Cell Parts',
        'cell_lines': 'Cell Lines',
        'instruments': 'Instruments',
        'fragmentation': 'Fragmentation',
        'acquisition': 'Acquisition',
        'ionization': 'Ionization',
        'labeling': 'Labeling',
        'modifications': 'Modifications',
        'enzymes': 'Enzymes',
        'lc_column': 'LC Column',
        'treatments': 'Treatments',
    })

    def get_display_name(self, ontology_id: str) -> str:
        """Get human-readable name for an ontology."""
        return self.ontology_display_names.get(ontology_id, ontology_id)

    def get_ontology_path(self, ontology_id: str) -> Optional[Path]:
        """Get full path to ontology file."""
        if ontology_id not in self.ontology_files:
            return None
        return Path(self.ontology_dir) / self.ontology_files[ontology_id]

    def get_cache_path(self, ontology_id: str) -> Path:
        """Get cache path for ontology index."""
        return Path(self.cache_dir) / f"{ontology_id}_index.pkl"

    @classmethod
    def from_dict(cls, cfg: dict) -> "NormalizationConfig":
        """Create config from a normalization config dict (e.g. from config.yaml)."""
        return cls(
            ontology_dir=cfg.get("ontology_dir", "ontologies/"),
            cache_dir=cfg.get("cache_dir", "ontology_cache/"),
            similarity_threshold=cfg.get("similarity_threshold", 0.7),
            use_gpu=cfg.get("use_gpu", True),
            index_backend=cfg.get("backend", "faiss"),
            use_quantization=cfg.get("use_quantization", True),
            term_aliases=cfg.get("term_aliases", {}),
        )
