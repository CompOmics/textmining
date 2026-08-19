"""
Ontology loading and graph management.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Iterator, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OntologyNode:
    """
    Represents a node in the ontology.
    
    Attributes:
        id: Ontology ID (e.g., CL:0000001)
        name: Primary name of the term
        synonyms: Alternative names
        definition: Term definition
        parents: Parent term IDs
        children: Child term IDs
        is_obsolete: Whether term is obsolete
    """
    id: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    definition: str = ""
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    is_obsolete: bool = False
    
    @property
    def all_names(self) -> List[str]:
        """Get name and all synonyms."""
        return [self.name] + self.synonyms


class OntologyGraph:
    """
    Graph representation of an ontology.
    
    Provides traversal methods for parent/child relationships.
    
    Example:
        >>> graph = OntologyGraph()
        >>> graph.add_node(OntologyNode(id='CL:0000001', name='cell'))
        >>> ancestors = graph.get_ancestors('CL:0000001')
    """
    
    def __init__(self, ontology_id: str = ""):
        self.ontology_id = ontology_id
        self.nodes: Dict[str, OntologyNode] = {}
        self._name_to_id: Dict[str, str] = {}
        
    def add_node(self, node: OntologyNode) -> None:
        """Add node to graph."""
        self.nodes[node.id] = node
        # Index by name (lowercase)
        self._name_to_id[node.name.lower()] = node.id
        for syn in node.synonyms:
            self._name_to_id[syn.lower()] = node.id
    
    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_by_name(self, name: str) -> Optional[OntologyNode]:
        """Get node by name or synonym (case insensitive)."""
        node_id = self._name_to_id.get(name.lower())
        if node_id:
            return self.nodes.get(node_id)
        return None
    
    def get_ancestors(self, node_id: str, max_depth: int = 10) -> List[str]:
        """
        Get all ancestors of a node.
        
        Args:
            node_id: ID of the node
            max_depth: Maximum depth to traverse
            
        Returns:
            List of ancestor IDs (nearest first)
        """
        ancestors = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth or current_id in visited:
                continue
            visited.add(current_id)
            
            node = self.get_node(current_id)
            if node and node.parents:
                for parent_id in node.parents:
                    if parent_id not in visited:
                        ancestors.append(parent_id)
                        queue.append((parent_id, depth + 1))
        
        return ancestors
    
    def get_descendants(self, node_id: str, max_depth: int = 10) -> List[str]:
        """
        Get all descendants of a node.
        
        Args:
            node_id: ID of the node
            max_depth: Maximum depth to traverse
            
        Returns:
            List of descendant IDs (nearest first)
        """
        descendants = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth or current_id in visited:
                continue
            visited.add(current_id)
            
            node = self.get_node(current_id)
            if node and node.children:
                for child_id in node.children:
                    if child_id not in visited:
                        descendants.append(child_id)
                        queue.append((child_id, depth + 1))
        
        return descendants
    
    def get_common_ancestor(self, id1: str, id2: str) -> Optional[str]:
        """Find lowest common ancestor of two nodes."""
        ancestors1 = set([id1] + self.get_ancestors(id1))
        
        queue = [id2]
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in ancestors1:
                return current
            if current in visited:
                continue
            visited.add(current)
            
            node = self.get_node(current)
            if node and node.parents:
                queue.extend(node.parents)
        
        return None
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get path from node to root."""
        path = [node_id]
        current = self.get_node(node_id)
        
        while current and current.parents:
            parent_id = current.parents[0]  # Take first parent
            path.append(parent_id)
            current = self.get_node(parent_id)
        
        return path
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self) -> Iterator[OntologyNode]:
        return iter(self.nodes.values())


class OntologyLoader:
    """
    Load ontologies from OBO or OWL files.
    
    Supports OBO format (recommended) and basic OWL/XML.
    
    Example:
        >>> loader = OntologyLoader()
        >>> graph = loader.load_obo('ontologies/cl.obo')
    """
    
    def load(self, file_path: str) -> OntologyGraph:
        """
        Load ontology from file.

        Args:
            file_path: Path to ontology file

        Returns:
            OntologyGraph instance
        """
        path = Path(file_path)

        if path.suffix == '.obo':
            return self.load_obo(file_path)
        elif path.suffix in ['.owl', '.xml']:
            return self.load_owl(file_path)
        elif path.suffix == '.txt':
            return self.load_text_vocab(file_path)
        else:
            raise ValueError(f"Unknown ontology format: {path.suffix}")

    def load_text_vocab(self, file_path: str) -> OntologyGraph:
        """
        Load a plain-text vocabulary file (one canonical term per line).

        Used for fields without a standard ontology (e.g. enzymes, lc_column).
        SapBERT embedding distance handles synonym matching — only canonical
        terms are needed.
        """
        path = Path(file_path)
        ontology_id = path.stem.lower()
        graph = OntologyGraph(ontology_id)

        logger.debug(f"Loading text vocabulary from {file_path}")

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                term = line.strip()
                if not term or term.startswith('#'):
                    continue
                count += 1
                node = OntologyNode(
                    id=f"{ontology_id}:{count:04d}",
                    name=term,
                )
                graph.add_node(node)

        logger.debug(f"Loaded {len(graph)} terms from {ontology_id}")
        return graph
    
    def load_obo(self, file_path: str) -> OntologyGraph:
        """
        Load ontology from OBO file.
        
        Args:
            file_path: Path to OBO file
            
        Returns:
            OntologyGraph instance
        """
        path = Path(file_path)
        ontology_id = path.stem.lower()
        graph = OntologyGraph(ontology_id)
        
        logger.debug(f"Loading ontology from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse [Term] blocks
        term_pattern = re.compile(r'\[Term\](.*?)(?=\n\[|\Z)', re.DOTALL)
        terms = term_pattern.findall(content)
        
        for term_block in terms:
            node = self._parse_obo_term(term_block)
            if node and not node.is_obsolete:
                graph.add_node(node)
        
        # Build child relationships
        self._build_children(graph)
        
        logger.debug(f"Loaded {len(graph)} terms from {ontology_id}")
        return graph
    
    def _parse_obo_term(self, block: str) -> Optional[OntologyNode]:
        """Parse a single OBO term block."""
        lines = block.strip().split('\n')
        
        node_id = None
        name = None
        synonyms = []
        definition = ""
        parents = []
        is_obsolete = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('id:'):
                node_id = line[3:].strip()
            elif line.startswith('name:'):
                name = line[5:].strip()
            elif line.startswith('synonym:'):
                # Extract synonym text from quotes
                match = re.search(r'"([^"]+)"', line)
                if match:
                    synonyms.append(match.group(1))
            elif line.startswith('def:'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    definition = match.group(1)
            elif line.startswith('is_a:'):
                parent_id = line[5:].split('!')[0].strip()
                # Strip OBO qualifiers like {source="FMA"} from parent IDs
                if '{' in parent_id:
                    parent_id = parent_id.split('{')[0].strip()
                parents.append(parent_id)
            elif line.startswith('is_obsolete:') and 'true' in line.lower():
                is_obsolete = True
        
        if node_id and name:
            return OntologyNode(
                id=node_id,
                name=name,
                synonyms=synonyms,
                definition=definition,
                parents=parents,
                is_obsolete=is_obsolete
            )
        return None
    
    def _build_children(self, graph: OntologyGraph) -> None:
        """Build child relationships from parent relationships."""
        for node in graph:
            for parent_id in node.parents:
                parent = graph.get_node(parent_id)
                if parent:
                    parent.children.append(node.id)
    
    def load_owl(self, file_path: str) -> OntologyGraph:
        """
        Load ontology from OWL/XML file.
        
        Basic parser for OWL format. For complex OWL files,
        consider using owlready2 or rdflib.
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("xml.etree required for OWL parsing")
        
        path = Path(file_path)
        ontology_id = path.stem.lower()
        graph = OntologyGraph(ontology_id)
        
        logger.debug(f"Loading OWL ontology from {file_path}")
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Handle OWL namespaces
        ns = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'obo': 'http://purl.obolibrary.org/obo/',
        }
        
        # Find all Class elements
        for cls in root.findall('.//owl:Class', ns):
            about = cls.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '')
            
            if not about:
                continue
            
            # Extract ID from URI
            node_id = about.split('/')[-1].replace('_', ':')
            
            # Get label
            label = cls.find('rdfs:label', ns)
            name = label.text if label is not None else node_id
            
            # Get synonyms (various properties)
            synonyms = []
            for syn_elem in cls.findall('.//obo:hasExactSynonym', ns):
                if syn_elem.text:
                    synonyms.append(syn_elem.text)
            
            node = OntologyNode(
                id=node_id,
                name=name,
                synonyms=synonyms
            )
            graph.add_node(node)
        
        logger.debug(f"Loaded {len(graph)} terms from OWL")
        return graph


def download_ontology(url: str, output_path: str) -> None:
    """
    Download ontology file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save file
    """
    import urllib.request
    
    logger.info(f"Downloading ontology from {url}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    
    logger.info(f"Saved to {output_path}")
