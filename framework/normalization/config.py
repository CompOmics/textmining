"""
Configuration for normalization module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """
    Configuration for the normalization pipeline.
    
    Attributes:
        ontology_dir: Directory containing ontology files
        cache_dir: Directory for caching embeddings/indices
        embedding_model: SapBERT model name
        similarity_threshold: Minimum similarity for match
        top_k: Number of candidates to retrieve
        use_synonyms: Whether to include synonyms in index
        use_gpu: Whether to use GPU for embeddings
        index_backend: Backend for nearest-neighbor search
        batch_size: Batch size for embedding computation
    """
    
    ontology_dir: str = "ontologies/"
    cache_dir: str = "ontology_cache/"
    embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    similarity_threshold: float = 0.7
    top_k: int = 5
    use_synonyms: bool = True
    use_gpu: bool = True
    index_backend: str = "faiss"  # faiss (recommended), sklearn, or annoy
    use_quantization: bool = True  # Use compression for large indices
    batch_size: int = 64
    # User-supplied abbreviation→full-name mappings applied before embedding lookup.
    # Can be set programmatically or loaded from config.yaml under
    # normalization.term_aliases (key/value pairs).
    term_aliases: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Load overrides from config.yaml if present."""
        self._load_from_yaml()
        
    def _load_from_yaml(self):
        """Load configuration from config.yaml."""
        config_path = Path("config.yaml")
        if not config_path.exists():
            return
            
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
                
            if not config_data:
                return

            # Apply paths overrides
            if "paths" in config_data:
                paths = config_data["paths"]
                if "ontology_dir" in paths: self.ontology_dir = paths["ontology_dir"]
                if "cache_dir" in paths: self.cache_dir = paths["cache_dir"]
                
            # Apply normalization overrides
            if "normalization" in config_data:
                norm = config_data["normalization"]
                if "backend" in norm: self.index_backend = norm["backend"]
                if "use_gpu" in norm: self.use_gpu = norm["use_gpu"]
                if "use_quantization" in norm: self.use_quantization = norm["use_quantization"]
                if "similarity_threshold" in norm: self.similarity_threshold = norm["similarity_threshold"]
                if "top_k" in norm: self.top_k = norm["top_k"]
                if "term_aliases" in norm and isinstance(norm["term_aliases"], dict):
                    self.term_aliases.update(norm["term_aliases"])

            logger.info(f"Loaded configuration overrides from {config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    
    # Ontology file mappings - all available ontologies
    ontology_files: Dict[str, str] = field(default_factory=lambda: {
        # Core ontologies
        'cl': 'cl.obo',           # Cell Ontology
        'uberon': 'uberon.obo',   # Anatomy Ontology
        'species': 'species.obo', # Curated species subset (not full NCBITaxon)
        'doid': 'doid.obo',       # Disease Ontology
        'psi-ms': 'psi-ms.obo',   # Mass Spectrometry Ontology
        'unimod': 'unimod.obo',   # Unimod (PTMs)
        'bto': 'bto.obo',         # BRENDA Tissue Ontology
        # Additional ontologies
        'clo': 'clo.owl',         # Cell Line Ontology (OWL format)
        'pride-cv': 'pride-cv.obo',  # PRIDE Controlled Vocabulary
        'mondo': 'mondo.obo',     # Mondo Disease Ontology
        'psimod': 'psimod.obo',   # PSI-Mod (modifications)
        'experimentalfactor': 'experimentalfactor.obo',  # Experimental Factor Ontology (EFO)
        'drosophilaanatomy': 'drosophilaanatomy.obo',    # Drosophila anatomy
        'plantontology': 'plantontology.obo',            # Plant ontology
        'zebrafishanatomydevelopment': 'zebrafishanatomydevelopment.obo',  # Zebrafish anatomy
        'flybase': 'flybase.obo',        # FlyBase controlled vocabulary
        'ratstrains': 'ratstrains.obo',  # Rat strains
        'chebi': 'chebi.obo',            # Chemical entities (ChEBI)
        'phenotypeandtrait': 'phenotypeandtrait.obo',    # PATO (Phenotype and Trait)
    })
    
    # Entity type to ontology mapping
    entity_ontology_map: Dict[str, str] = field(default_factory=lambda: {
        # Species/Organism
        'species': 'species',
        'organism': 'species',
        'rat_strain': 'ratstrains',
        # Cell types and lines
        'cell_type': 'cl',
        'cell_line': 'clo',
        # Anatomy/Tissue
        'tissue': 'uberon',
        'organ': 'uberon',
        'sample_source': 'bto',
        'drosophila_anatomy': 'drosophilaanatomy',
        'plant_anatomy': 'plantontology',
        'zebrafish_anatomy': 'zebrafishanatomydevelopment',
        # Disease
        'disease': 'mondo',
        'disease_state': 'mondo',
        'mondo_disease': 'mondo',
        # Mass spectrometry
        'instrument': 'psi-ms',
        'labelling': 'psi-ms',
        'fractionation': 'psi-ms',
        'pride_cv': 'pride-cv',
        # Modifications
        'modification': 'psimod',
        'ptm': 'psimod',
        'psimod': 'psimod',
        # Other
        'experimental_factor': 'experimentalfactor',
        'chemical': 'chebi',
        'phenotype': 'phenotypeandtrait',
        'flybase': 'flybase',
    })
    
    def get_ontology_path(self, ontology_id: str) -> Optional[Path]:
        """Get full path to ontology file."""
        if ontology_id not in self.ontology_files:
            return None
        return Path(self.ontology_dir) / self.ontology_files[ontology_id]
    
    def get_cache_path(self, ontology_id: str) -> Path:
        """Get cache path for ontology index."""
        return Path(self.cache_dir) / f"{ontology_id}_index.pkl"


# Default ontology URLs for downloading
ONTOLOGY_URLS = {
    'cl': 'http://purl.obolibrary.org/obo/cl.obo',
    'uberon': 'http://purl.obolibrary.org/obo/uberon.obo',
    'ncbitaxon': 'http://purl.obolibrary.org/obo/ncbitaxon.obo',
    'doid': 'http://purl.obolibrary.org/obo/doid.obo',
    'ms': 'https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo',
}
