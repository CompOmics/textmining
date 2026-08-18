#!/usr/bin/env python3
"""
Pre-build ontology indices for faster normalization.

This script loads all ontologies and builds SapBERT embedding indices,
caching them to disk for faster startup when running the pipeline.

Usage:
    python -m normalization.build_index
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_indices():
    """Build and cache all ontology indices."""
    from .config import NormalizationConfig
    from .normalizer import TermNormalizer
    
    # Configure paths
    ontology_dir = Path(__file__).parent.parent / "ontologies"
    cache_dir = Path(__file__).parent.parent / "ontology_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Ontology directory: {ontology_dir}")
    logger.info(f"Cache directory: {cache_dir}")
    
    # Check what ontologies we have
    obo_files = list(ontology_dir.glob("*.obo"))
    owl_files = list(ontology_dir.glob("*.owl"))
    all_files = obo_files + owl_files
    
    logger.info(f"Found {len(all_files)} ontology files")
    
    # All ontologies to index
    core_ontologies = {
        # Core ontologies (prioritized for extraction)
        'cl': 'cl.obo',           # Cell types
        'uberon': 'uberon.obo',   # Tissues
        'species': 'species.obo', # Species
        'doid': 'doid.obo',       # Diseases
        'psi-ms': 'psi-ms.obo',   # MS terms
        'unimod': 'unimod.obo',   # PTMs
        'bto': 'bto.obo',         # Sample sources
        # Additional ontologies
        'clo': 'clo.owl',         # Cell lines (OWL format)
        'pride-cv': 'pride-cv.obo',  # PRIDE Controlled Vocabulary
        'mondo': 'mondo.obo',     # Mondo Disease Ontology
        'psimod': 'psimod.obo',   # PSI-Mod
        'experimentalfactor': 'experimentalfactor.obo',  # EFO
        'drosophilaanatomy': 'drosophilaanatomy.obo',    # Drosophila anatomy
        'plantontology': 'plantontology.obo',            # Plant ontology
        'zebrafishanatomydevelopment': 'zebrafishanatomydevelopment.obo',  # Zebrafish
        'flybase': 'flybase.obo',        # FlyBase
        'ratstrains': 'ratstrains.obo',  # Rat strains
        'chebi': 'chebi.obo',            # Chemical entities
        'phenotypeandtrait': 'phenotypeandtrait.obo',    # PATO
    }
    
    # Create config
    config = NormalizationConfig(
        ontology_dir=str(ontology_dir),
        cache_dir=str(cache_dir),
        similarity_threshold=0.7,
        use_gpu=True,
    )
    
    # Update ontology file mappings
    config.ontology_files = core_ontologies
    
    # Build indices
    logger.info("Initializing TermNormalizer...")
    normalizer = TermNormalizer(config)
    
    logger.info("Loading and indexing ontologies...")
    normalizer.load_all_ontologies()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INDEXING COMPLETE")
    logger.info("="*60)
    
    for ont_id in normalizer.indices:
        index = normalizer.indices[ont_id]
        n_terms = len(index.term_ids) if hasattr(index, 'term_ids') else 'unknown'
        logger.info(f"  {ont_id}: {n_terms} terms indexed")
    
    logger.info(f"\nCache saved to: {cache_dir}")
    logger.info("The normalization agent will now load faster!")


if __name__ == "__main__":
    print("="*60)
    print("ONTOLOGY INDEX BUILDER")
    print("="*60)
    print("\nThis will build SapBERT embedding indices for all ontologies.")
    print("First run may take 5-10 minutes. Subsequent runs use cache.\n")
    
    build_indices()
