#!/usr/bin/env python3
"""
Download ontology files from their official sources.

This script downloads all required ontology files for the normalization module.
Files are saved to the ontologies/ directory.

Usage:
    python -m normalization.download           # Download all ontologies
    python -m normalization.download --check   # Check which ontologies exist
    python -m normalization.download cl uberon # Download specific ontologies
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ontology sources - official download URLs
ONTOLOGY_SOURCES: Dict[str, Dict[str, str]] = {
    # Core ontologies
    'cl': {
        'url': 'http://purl.obolibrary.org/obo/cl.obo',
        'filename': 'cl.obo',
        'description': 'Cell Ontology',
    },
    'uberon': {
        'url': 'http://purl.obolibrary.org/obo/uberon.obo',
        'filename': 'uberon.obo',
        'description': 'Anatomy Ontology (UBERON)',
    },
    'doid': {
        'url': 'http://purl.obolibrary.org/obo/doid.obo',
        'filename': 'doid.obo',
        'description': 'Disease Ontology',
    },
    'bto': {
        'url': 'http://purl.obolibrary.org/obo/bto.obo',
        'filename': 'bto.obo',
        'description': 'BRENDA Tissue Ontology',
    },
    'psi-ms': {
        'url': 'https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo',
        'filename': 'psi-ms.obo',
        'description': 'Mass Spectrometry Ontology',
    },
    'unimod': {
        'url': 'https://www.unimod.org/obo/unimod.obo',
        'filename': 'unimod.obo',
        'description': 'Unimod PTM Ontology',
    },
    'pride-cv': {
        'url': 'https://raw.githubusercontent.com/PRIDE-Utilities/pride-ontology/master/pride_cv.obo',
        'filename': 'pride-cv.obo',
        'description': 'PRIDE Controlled Vocabulary',
    },
    # Additional ontologies
    'clo': {
        'url': 'http://purl.obolibrary.org/obo/clo.owl',
        'filename': 'clo.owl',
        'description': 'Cell Line Ontology',
    },
    'mondo': {
        'url': 'http://purl.obolibrary.org/obo/mondo.obo',
        'filename': 'mondo.obo',
        'description': 'Mondo Disease Ontology',
    },
    'psimod': {
        'url': 'http://purl.obolibrary.org/obo/mod.obo',
        'filename': 'psimod.obo',
        'description': 'PSI-Mod Ontology',
    },
    'experimentalfactor': {
        'url': 'https://github.com/EBISPOT/efo/releases/download/current/efo.obo',
        'filename': 'experimentalfactor.obo',
        'description': 'Experimental Factor Ontology (EFO)',
    },
    'chebi': {
        'url': 'http://purl.obolibrary.org/obo/chebi.obo',
        'filename': 'chebi.obo',
        'description': 'Chemical Entities of Biological Interest (ChEBI)',
    },
    'hancestro': {
        'url': 'https://raw.githubusercontent.com/EBISPOT/hancestro/main/hancestro.obo',
        'filename': 'hancestro.obo',
        'description': 'Human Ancestry Ontology (HANCESTRO)',
    },
    'phenotypeandtrait': {
        'url': 'http://purl.obolibrary.org/obo/pato.obo',
        'filename': 'phenotypeandtrait.obo',
        'description': 'Phenotype and Trait Ontology (PATO)',
    },
    'plantontology': {
        'url': 'http://purl.obolibrary.org/obo/po.obo',
        'filename': 'plantontology.obo',
        'description': 'Plant Ontology',
    },
    'drosophilaanatomy': {
        'url': 'http://purl.obolibrary.org/obo/fbbt.obo',
        'filename': 'drosophilaanatomy.obo',
        'description': 'Drosophila Anatomy Ontology (FBbt)',
    },
    'zebrafishanatomydevelopment': {
        'url': 'http://purl.obolibrary.org/obo/zfa.obo',
        'filename': 'zebrafishanatomydevelopment.obo',
        'description': 'Zebrafish Anatomy Ontology (ZFA)',
    },
    'flybase': {
        'url': 'http://purl.obolibrary.org/obo/fbcv.obo',
        'filename': 'flybase.obo',
        'description': 'FlyBase Controlled Vocabulary',
    },
    'ratstrains': {
        'url': 'http://purl.obolibrary.org/obo/rs.obo',
        'filename': 'ratstrains.obo',
        'description': 'Rat Strain Ontology',
    },
}

# Species is a custom curated subset, not available from OBO
# We auto-generate it with common model organisms
CUSTOM_ONTOLOGIES = {
    'species': {
        'filename': 'species.obo',
        'description': 'Curated species subset (NCBITaxon)',
        'note': 'Auto-generated with common model organisms',
    },
}

# Curated species list for proteomics/life sciences
CURATED_SPECIES = [
    # Common model organisms
    ("NCBITaxon:9606", "Homo sapiens", ["human", "Homo sapiens (human)"]),
    ("NCBITaxon:10090", "Mus musculus", ["mouse", "house mouse"]),
    ("NCBITaxon:10116", "Rattus norvegicus", ["rat", "Norway rat"]),
    ("NCBITaxon:9913", "Bos taurus", ["cow", "cattle", "bovine"]),
    ("NCBITaxon:9823", "Sus scrofa", ["pig", "wild boar", "porcine"]),
    ("NCBITaxon:9031", "Gallus gallus", ["chicken"]),
    ("NCBITaxon:7955", "Danio rerio", ["zebrafish"]),
    ("NCBITaxon:7227", "Drosophila melanogaster", ["fruit fly", "Drosophila"]),
    ("NCBITaxon:6239", "Caenorhabditis elegans", ["C. elegans", "nematode"]),
    ("NCBITaxon:4932", "Saccharomyces cerevisiae", ["yeast", "baker's yeast", "budding yeast"]),
    ("NCBITaxon:562", "Escherichia coli", ["E. coli"]),
    # Primates
    ("NCBITaxon:9544", "Macaca mulatta", ["rhesus monkey", "rhesus macaque"]),
    ("NCBITaxon:9598", "Pan troglodytes", ["chimpanzee", "chimp"]),
    ("NCBITaxon:9541", "Macaca fascicularis", ["crab-eating macaque", "cynomolgus monkey"]),
    # Other mammals
    ("NCBITaxon:9615", "Canis lupus familiaris", ["dog", "domestic dog", "canine"]),
    ("NCBITaxon:9685", "Felis catus", ["cat", "domestic cat", "feline"]),
    ("NCBITaxon:9986", "Oryctolagus cuniculus", ["rabbit", "European rabbit"]),
    ("NCBITaxon:9940", "Ovis aries", ["sheep", "domestic sheep", "ovine"]),
    ("NCBITaxon:9796", "Equus caballus", ["horse", "domestic horse", "equine"]),
    ("NCBITaxon:9825", "Sus scrofa domesticus", ["domestic pig"]),
    # Plants
    ("NCBITaxon:3702", "Arabidopsis thaliana", ["thale cress", "mouse-ear cress", "Arabidopsis"]),
    ("NCBITaxon:4530", "Oryza sativa", ["rice"]),
    ("NCBITaxon:4577", "Zea mays", ["maize", "corn"]),
    # Amphibians
    ("NCBITaxon:8355", "Xenopus laevis", ["African clawed frog", "Xenopus"]),
    # Bacteria
    ("NCBITaxon:83333", "Escherichia coli K-12", ["E. coli K-12", "E. coli K12"]),
    ("NCBITaxon:287", "Pseudomonas aeruginosa", ["P. aeruginosa"]),
    ("NCBITaxon:1423", "Bacillus subtilis", ["B. subtilis"]),
    ("NCBITaxon:1280", "Staphylococcus aureus", ["S. aureus", "Staph aureus"]),
    ("NCBITaxon:2104", "Mycoplasma", ["mycoplasma"]),
    # Yeast strains
    ("NCBITaxon:559292", "Saccharomyces cerevisiae S288C", ["S. cerevisiae S288C"]),
    # Viruses
    ("NCBITaxon:2697049", "SARS-CoV-2", ["Severe acute respiratory syndrome coronavirus 2", "COVID-19 virus"]),
    ("NCBITaxon:11676", "Human immunodeficiency virus 1", ["HIV-1", "HIV"]),
]


def get_ontology_dir() -> Path:
    """Get the ontologies directory path."""
    return Path(__file__).parent.parent / "ontologies"


def download_ontology(ontology_id: str, 
                      output_dir: Optional[Path] = None,
                      force: bool = False) -> Tuple[bool, str]:
    """
    Download a single ontology file.
    
    Args:
        ontology_id: Ontology identifier (e.g., 'cl', 'uberon')
        output_dir: Directory to save file (default: ontologies/)
        force: Re-download even if file exists
        
    Returns:
        Tuple of (success, message)
    """
    if ontology_id not in ONTOLOGY_SOURCES:
        if ontology_id in CUSTOM_ONTOLOGIES:
            return False, f"{ontology_id}: Custom ontology - not available for download"
        return False, f"Unknown ontology: {ontology_id}"
    
    source = ONTOLOGY_SOURCES[ontology_id]
    output_dir = output_dir or get_ontology_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / source['filename']
    
    if output_path.exists() and not force:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        return True, f"{ontology_id}: Already exists ({size_mb:.1f} MB)"
    
    logger.info(f"Downloading {source['description']}...")
    logger.info(f"  URL: {source['url']}")
    
    try:
        # Setup session with retries
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Stream download
        response = session.get(source['url'], stream=True, timeout=(10, 60))
        response.raise_for_status()
        
        # Check size if available
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            logger.info(f"  Size: {total_size / (1024*1024):.1f} MB")
            
        # Download and count bytes
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0 and downloaded % (1024*1024) == 0:
                        percent = int(downloaded * 100 / total_size)
                        if percent % 10 == 0:
                            logger.info(f"  Progress: {percent}%")

        # Validation: Check strictly for file size
        actual_size = output_path.stat().st_size
        if actual_size < 10 * 1024:  # < 10KB is suspicious (likely error page)
            logger.warning(f"  WARNING: File too small ({actual_size} bytes). Possible error page.")
            output_path.unlink() # Delete bad file
            return False, f"{ontology_id}: Failed - Downloaded file too small (<10KB)"
            
        size_mb = actual_size / (1024 * 1024)
        return True, f"{ontology_id}: Downloaded successfully ({size_mb:.1f} MB)"
        
    except Exception as e:
        if output_path.exists():
            output_path.unlink() # Clean up partial file
        return False, f"{ontology_id}: Error - {str(e)}"


def create_species_ontology(output_dir: Optional[Path] = None,
                            force: bool = False) -> Tuple[bool, str]:
    """
    Create the curated species.obo file with common model organisms.
    
    Args:
        output_dir: Directory to save file (default: ontologies/)
        force: Overwrite even if file exists
        
    Returns:
        Tuple of (success, message)
    """
    output_dir = output_dir or get_ontology_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "species.obo"
    
    if output_path.exists() and not force:
        size_kb = output_path.stat().st_size / 1024
        return True, f"species: Already exists ({size_kb:.1f} KB)"
    
    logger.info("Creating curated species.obo...")
    
    # Build OBO content
    lines = [
        "format-version: 1.2",
        "ontology: ncbitaxon_subset",
        "data-version: curated-2025",
        "default-namespace: NCBITaxon",
        "",
    ]
    
    for term_id, name, synonyms in CURATED_SPECIES:
        lines.append("[Term]")
        lines.append(f"id: {term_id}")
        lines.append(f"name: {name}")
        for syn in synonyms:
            lines.append(f'synonym: "{syn}" EXACT []')
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return True, f"species: Created with {len(CURATED_SPECIES)} organisms"


def download_all_ontologies(output_dir: Optional[Path] = None,
                           force: bool = False,
                           skip_large: bool = False) -> Dict[str, Tuple[bool, str]]:
    """
    Download all ontology files including curated species.
    
    Args:
        output_dir: Directory to save files
        force: Re-download even if files exist
        skip_large: Skip very large ontologies (chebi, efo)
        
    Returns:
        Dictionary of ontology_id -> (success, message)
    """
    results = {}
    
    # Large ontologies that can be skipped for faster setup
    large_ontologies = {'chebi', 'experimentalfactor', 'drosophilaanatomy'}
    
    # Download from URLs
    for ontology_id in ONTOLOGY_SOURCES:
        if skip_large and ontology_id in large_ontologies:
            results[ontology_id] = (False, f"{ontology_id}: Skipped (large file)")
            continue
            
        success, message = download_ontology(ontology_id, output_dir, force)
        results[ontology_id] = (success, message)
        logger.info(message)
    
    # Create curated species.obo
    success, message = create_species_ontology(output_dir, force)
    results['species'] = (success, message)
    logger.info(message)
    
    return results


def check_ontologies(output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Check which ontologies exist and their status.
    
    Args:
        output_dir: Directory to check
        
    Returns:
        Dictionary with ontology status information
    """
    output_dir = output_dir or get_ontology_dir()
    status = {}
    
    all_ontologies = {**ONTOLOGY_SOURCES, **CUSTOM_ONTOLOGIES}
    
    for ontology_id, info in all_ontologies.items():
        file_path = output_dir / info['filename']
        
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            status[ontology_id] = {
                'exists': True,
                'path': str(file_path),
                'size_mb': size_mb,
                'description': info['description'],
            }
        else:
            status[ontology_id] = {
                'exists': False,
                'path': str(file_path),
                'size_mb': 0,
                'description': info['description'],
                'downloadable': ontology_id in ONTOLOGY_SOURCES,
            }
    
    return status


def get_ontology_info(ontology_id: str) -> Optional[Dict]:
    """Get information about an ontology."""
    if ontology_id in ONTOLOGY_SOURCES:
        return ONTOLOGY_SOURCES[ontology_id]
    elif ontology_id in CUSTOM_ONTOLOGIES:
        return CUSTOM_ONTOLOGIES[ontology_id]
    return None


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Download ontology files for normalization'
    )
    parser.add_argument(
        'ontologies',
        nargs='*',
        help='Specific ontologies to download (default: all)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check which ontologies exist'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download even if files exist'
    )
    parser.add_argument(
        '--skip-large',
        action='store_true',
        help='Skip large ontologies (chebi, efo, fbbt)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available ontologies'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: ontologies/)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.list:
        print("\nAvailable Ontologies:")
        print("=" * 60)
        for ont_id, info in ONTOLOGY_SOURCES.items():
            print(f"  {ont_id:25} - {info['description']}")
        print("\nCustom Ontologies (not auto-downloaded):")
        for ont_id, info in CUSTOM_ONTOLOGIES.items():
            print(f"  {ont_id:25} - {info['description']}")
        return
    
    if args.check:
        print("\nOntology Status:")
        print("=" * 60)
        status = check_ontologies(output_dir)
        
        existing = [k for k, v in status.items() if v['exists']]
        missing = [k for k, v in status.items() if not v['exists']]
        
        total_size = sum(v['size_mb'] for v in status.values())
        
        print(f"\nFound: {len(existing)}/{len(status)} ontologies")
        print(f"Total size: {total_size:.1f} MB")
        
        if missing:
            print(f"\nMissing ontologies:")
            for ont_id in missing:
                downloadable = status[ont_id].get('downloadable', False)
                mark = "✓" if downloadable else "✗"
                print(f"  [{mark}] {ont_id} - {status[ont_id]['description']}")
        return
    
    print("\n" + "=" * 60)
    print("ONTOLOGY DOWNLOADER")
    print("=" * 60)
    
    if args.ontologies:
        # Download specific ontologies
        for ont_id in args.ontologies:
            success, message = download_ontology(ont_id, output_dir, args.force)
            print(message)
    else:
        # Download all
        print("\nDownloading all ontologies...")
        print("This may take a while depending on your connection.\n")
        
        results = download_all_ontologies(output_dir, args.force, args.skip_large)
        
        successful = sum(1 for s, _ in results.values() if s)
        print(f"\n{'=' * 60}")
        print(f"Downloaded: {successful}/{len(results)} ontologies")
        
        if successful < len(results):
            print("\nFailed downloads:")
            for ont_id, (success, msg) in results.items():
                if not success:
                    print(f"  - {msg}")


if __name__ == "__main__":
    main()
