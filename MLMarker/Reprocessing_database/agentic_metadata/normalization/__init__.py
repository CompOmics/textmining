"""
Normalization module for ontology-based entity normalization.

Components:
    - OntologyLoader: Load ontologies from OBO/OWL files
    - OntologyIndex: Build searchable index with embeddings
    - TermNormalizer: Normalize terms against ontologies
    - download_all_ontologies: Download required ontology files
"""

from .ontology import OntologyLoader, OntologyGraph, OntologyNode
from .index import OntologyIndex
from .normalizer import TermNormalizer, NormalizationResult
from .config import NormalizationConfig

# Optional download module (may not exist)
try:
    from .download import (
        download_all_ontologies,
        download_ontology,
        check_ontologies,
        get_ontology_info,
        ONTOLOGY_SOURCES,
    )
    _HAS_DOWNLOAD = True
except ImportError:
    _HAS_DOWNLOAD = False
    download_all_ontologies = None
    download_ontology = None
    check_ontologies = None
    get_ontology_info = None
    ONTOLOGY_SOURCES = None

__all__ = [
    'OntologyLoader',
    'OntologyGraph',
    'OntologyNode',
    'OntologyIndex',
    'TermNormalizer',
    'NormalizationResult',
    'NormalizationConfig',
]

# Add download utilities only if available
if _HAS_DOWNLOAD:
    __all__.extend([
        'download_all_ontologies',
        'download_ontology',
        'check_ontologies',
        'get_ontology_info',
        'ONTOLOGY_SOURCES',
    ])
