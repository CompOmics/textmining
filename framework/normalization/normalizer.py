"""
Term normalization against ontologies.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .ontology import OntologyGraph, OntologyNode, OntologyLoader
from .index import OntologyIndex
from .config import NormalizationConfig

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """
    Result of normalizing a term.
    
    Attributes:
        original_term: Original input term
        ontology_id: Matched ontology term ID
        ontology_name: Matched ontology term name
        similarity: Similarity score
        matched_text: Which text was matched (name/synonym)
        candidates: Alternative matches
        entity_type: Type of entity (cell_type, tissue, etc.)
        is_normalized: Whether normalization succeeded
    """
    original_term: str
    ontology_id: Optional[str] = None
    ontology_name: Optional[str] = None
    similarity: float = 0.0
    matched_text: Optional[str] = None
    candidates: List[Tuple[str, str, float]] = field(default_factory=list)
    entity_type: str = ""
    is_normalized: bool = False
    expanded_term: Optional[str] = None  # Non-None when abbreviation was expanded
    normalization_method: Optional[str] = None
    synonym_scope: Optional[str] = None
    qc_flags: List[str] = field(default_factory=list)
    
    @property
    def confidence(self) -> str:
        """Categorize confidence level."""
        if self.similarity >= 0.9:
            return 'high'
        elif self.similarity >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            'original_term': self.original_term,
            'ontology_id': self.ontology_id,
            'ontology_name': self.ontology_name,
            'similarity': self.similarity,
            'matched_text': self.matched_text,
            'entity_type': self.entity_type,
            'is_normalized': self.is_normalized,
            'confidence': self.confidence,
            'num_candidates': len(self.candidates),
        }
        if self.expanded_term:
            d['expanded_term'] = self.expanded_term
        if self.normalization_method:
            d['normalization_method'] = self.normalization_method
        if self.synonym_scope:
            d['synonym_scope'] = self.synonym_scope
        if self.qc_flags:
            d['normalization_qc_flags'] = self.qc_flags
        return d


class TermNormalizer:
    """
    Normalize terms against ontologies.
    
    Uses embedding-based search and disambiguation to find
    the best matching ontology terms.
    
    Example:
        >>> normalizer = TermNormalizer()
        >>> normalizer.load_ontology('cl', 'ontologies/cl.obo')
        >>> result = normalizer.normalize('epithelial cell', entity_type='cell_type')
    """
    
    # Regex for standard scientific binomial names (e.g. "Plasmodium falciparum")
    _BINOMIAL_RE = re.compile(r'^([A-Z][a-z]+) ([a-z][a-z0-9_-]+)$')

    # Several source ontologies include imported nodes from other namespaces.
    # Exact-name precedence must stay within the requested ontology namespace;
    # e.g. a MONDO disease lookup must not resolve to an imported HP term.
    _ONTOLOGY_ID_PREFIXES: Dict[str, Tuple[str, ...]] = {
        'bto': ('BTO:',),
        'cl': ('CL:',),
        'clo': ('CLO:',),
        'doid': ('DOID:',),
        'mondo': ('MONDO:',),
        'pride-cv': ('PRIDE:',),
        'psi-ms': ('MS:',),
        'psimod': ('MOD:',),
        'species': ('NCBITaxon:',),
        'uberon': ('UBERON:',),
        'unimod': ('UNIMOD:',),
    }

    _PTM_UNIMOD_ALIASES: Dict[str, str] = {
        'acetylation': 'acetyl',
        'carbamidomethylation': 'carbamidomethyl',
        'oxidized': 'oxidation',
        'phosphorylation': 'phospho',
    }

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize normalizer.
        
        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        self.loader = OntologyLoader()
        self.graphs: Dict[str, OntologyGraph] = {}
        self.indices: Dict[str, OntologyIndex] = {}

    def _exact_primary_candidates(
        self,
        graph: OntologyGraph,
        term: str,
        ontology_id: str,
    ) -> List[OntologyNode]:
        """Return exact primary matches belonging to the target namespace."""
        candidates = graph.get_primary_candidates(term)
        prefixes = self._ONTOLOGY_ID_PREFIXES.get(ontology_id.lower())
        if not prefixes:
            return candidates
        return [node for node in candidates if node.id.startswith(prefixes)]

    @staticmethod
    def _add_graph_synonym(
        graph: Any,
        node: OntologyNode,
        synonym: str,
        scope: str,
    ) -> None:
        """Add a synonym, retaining compatibility with lightweight test graphs."""
        if hasattr(graph, 'add_synonym'):
            graph.add_synonym(node.id, synonym, scope=scope)
            return
        if synonym not in node.synonyms:
            node.synonyms.append(synonym)
        node.synonym_scopes[synonym] = scope.upper()

    def _exact_lexical_result(
        self,
        graph: OntologyGraph,
        term: str,
        ontology_id: str,
        entity_type: str,
    ) -> Optional[NormalizationResult]:
        """Resolve deterministic primary/synonym matches without embeddings."""
        primary_matches = self._exact_primary_candidates(
            graph, term, ontology_id
        )
        if len(primary_matches) == 1:
            node = primary_matches[0]
            return NormalizationResult(
                original_term=term,
                ontology_id=node.id,
                ontology_name=node.name,
                similarity=1.0,
                matched_text=node.name,
                candidates=[(node.id, node.name, 1.0)],
                entity_type=entity_type,
                is_normalized=True,
                normalization_method='exact_primary',
            )
        if len(primary_matches) > 1:
            candidates = [
                (node.id, node.name, 1.0) for node in primary_matches
            ]
            logger.warning(
                "Ambiguous exact primary label %r in %s: %s",
                term,
                ontology_id,
                [node.id for node in primary_matches],
            )
            return NormalizationResult(
                original_term=term,
                matched_text=term,
                candidates=candidates,
                entity_type=entity_type,
                is_normalized=False,
                normalization_method='ambiguous_exact_primary',
                qc_flags=['ambiguous_ontology_match'],
            )

        prefixes = self._ONTOLOGY_ID_PREFIXES.get(ontology_id.lower())
        raw_matches = graph.get_synonym_matches(term)
        if prefixes:
            raw_matches = [
                (node, scope) for node, scope in raw_matches
                if node.id.startswith(prefixes)
            ]
        if not raw_matches:
            return None

        by_id: Dict[str, Tuple[OntologyNode, set]] = {}
        for node, scope in raw_matches:
            if node.id not in by_id:
                by_id[node.id] = (node, set())
            by_id[node.id][1].add(scope.upper())
        exact_ids = [
            node_id for node_id, (_node, scopes) in by_id.items()
            if scopes & {'EXACT', 'CUSTOM'}
        ]
        if len(exact_ids) == 1:
            node, scopes = by_id[exact_ids[0]]
            scope = 'CUSTOM' if 'CUSTOM' in scopes else 'EXACT'
            return NormalizationResult(
                original_term=term,
                ontology_id=node.id,
                ontology_name=node.name,
                similarity=1.0,
                matched_text=term,
                candidates=[(node.id, term, 1.0)],
                entity_type=entity_type,
                is_normalized=True,
                normalization_method='exact_synonym',
                synonym_scope=scope,
            )

        candidates = [
            (node.id, node.name, 1.0) for node, _scopes in by_id.values()
        ]
        if len(exact_ids) > 1 or len(by_id) > 1:
            method = 'ambiguous_exact_synonym'
            flag = 'ambiguous_ontology_match'
        else:
            method = 'related_synonym_requires_context'
            flag = 'related_synonym_requires_context'
        logger.warning(
            "Unresolved exact synonym %r in %s (%s): %s",
            term,
            ontology_id,
            method,
            list(by_id),
        )
        return NormalizationResult(
            original_term=term,
            matched_text=term,
            candidates=candidates,
            entity_type=entity_type,
            is_normalized=False,
            normalization_method=method,
            qc_flags=[flag],
        )

    def _ptm_unimod_result(
        self, term: str, entity_type: str
    ) -> Optional[NormalizationResult]:
        """Resolve generic modification labels against canonical Unimod names."""
        if entity_type.lower() not in {'modification', 'ptm', 'psimod'}:
            return None
        graph = self.graphs.get('unimod')
        if graph is None:
            return None
        lookup_term = self._PTM_UNIMOD_ALIASES.get(
            " ".join(term.casefold().split()), term
        )
        matches = self._exact_primary_candidates(graph, lookup_term, 'unimod')
        if len(matches) != 1:
            return None
        node = matches[0]
        method = (
            'exact_primary_unimod'
            if lookup_term == term else 'alias_to_unimod'
        )
        return NormalizationResult(
            original_term=term,
            ontology_id=node.id,
            ontology_name=node.name,
            similarity=1.0,
            matched_text=node.name,
            candidates=[(node.id, node.name, 1.0)],
            entity_type=entity_type,
            is_normalized=True,
            normalization_method=method,
        )
    
    def _inject_abbreviated_synonyms(self, graph: OntologyGraph) -> int:
        """
        Inject abbreviated scientific name forms as synonyms into graph nodes.

        For every non-obsolete node whose primary name is a standard binomial
        (e.g. ``Plasmodium falciparum``), two abbreviated forms are appended to
        the node's synonym list **before** the embedding index is built:

            1. Single-letter genus prefix  →  ``p.falciparum``
            2. Full genus dot-separated    →  ``Plasmodium.falciparum``

        These abbreviated forms are common in scientific writing but are absent
        from ontology synonym fields.  Injecting them here means SapBERT embeds
        them as recognised variants of the correct ontology entry, so a query of
        ``p.falciparum`` matches directly without any query-rewriting.

        .. note::
            This modifies in-memory graph nodes only; the source ``.obo`` files
            are not changed.  Existing cached indices do **not** contain these
            synonyms — delete ``ontology_cache/`` and rebuild if you want the
            index to include them (see ``_expand_term`` fallback for existing
            caches).

        Returns:
            Number of nodes that received new synonyms.
        """
        count = 0
        for node in graph:
            if node.is_obsolete:
                continue
            m = self._BINOMIAL_RE.match(node.name)
            if not m:
                continue

            genus, epithet = m.group(1), m.group(2)
            existing_lower = {s.lower() for s in node.synonyms} | {node.name.lower()}

            added = False
            for form in (f"{genus[0].lower()}.{epithet}", f"{genus}.{epithet}"):
                if form.lower() not in existing_lower:
                    self._add_graph_synonym(
                        graph, node, form, scope="EXACT"
                    )
                    existing_lower.add(form.lower())
                    added = True
            if added:
                count += 1

        logger.info(
            f"Injected abbreviated binomial synonyms into {count} ontology nodes "
            f"(delete ontology_cache/ to rebuild index with these synonyms)"
        )
        return count

    def load_ontology(self,
                      ontology_id: str,
                      file_path: str,
                      use_cache: bool = True) -> None:
        """
        Load an ontology and build index.

        Abbreviated scientific name synonyms (e.g. ``p.falciparum``) are
        injected into the graph nodes before the index is built, so that
        freshly built indices match abbreviated queries directly via SapBERT.
        Existing cached indices can fall back to ``_expand_term`` in
        ``normalize()``; rebuild by deleting ``ontology_cache/``.

        Args:
            ontology_id: Identifier for ontology (e.g., 'cl')
            file_path: Path to ontology file
            use_cache: Whether to use cached index
        """
        logger.info(f"Loading ontology: {ontology_id}")

        # Load graph and inject abbreviated synonyms before indexing
        graph = self.loader.load(file_path)
        self._inject_abbreviated_synonyms(graph)
        self._load_custom_synonyms(graph, ontology_id)  # reload persisted custom synonyms
        self.graphs[ontology_id] = graph

        # Build or load index
        cache_path = self.config.get_cache_path(ontology_id)

        if use_cache:
            self.indices[ontology_id] = OntologyIndex.load_or_build(
                graph, str(cache_path), self.config
            )
        else:
            index = OntologyIndex(self.config)
            index.build(graph, include_synonyms=self.config.use_synonyms)
            self.indices[ontology_id] = index
    
    def load_all_ontologies(self) -> None:
        """Load all configured ontologies."""
        for ontology_id, filename in self.config.ontology_files.items():
            path = self.config.get_ontology_path(ontology_id)
            if path and path.exists():
                self.load_ontology(ontology_id, str(path))
            else:
                logger.warning(f"Ontology file not found: {path}")
    
    def get_ontology_for_entity(self, entity_type: str) -> Optional[str]:
        """Get ontology ID for entity type."""
        return self.config.entity_ontology_map.get(entity_type.lower())
    
    def _expand_term(self, term: str) -> Optional[str]:
        """
        Fallback query-side expansion for abbreviated terms.

        Used when searching a **cached index** that was built before
        ``_inject_abbreviated_synonyms`` was added (i.e. the abbreviated form
        is not yet embedded in the index).  For freshly-built or rebuilt
        indices the abbreviated forms are already present as synonyms and this
        method is never needed.

        Rules applied in order:

        1. ``Genus.species``  →  ``Genus species``
           (e.g. ``Plasmodium.falciparum`` → ``Plasmodium falciparum``)
           Pure text transform, no lookup table needed.
        2. Alias dict lookup from ``config.term_aliases`` (case-insensitive).

        The single-letter-prefix rule (``p.falciparum`` → ``Plasmodium
        falciparum``) is intentionally absent here: that form is handled by
        graph injection at index-build time.  If you have an old cache and need
        the rule, delete ``ontology_cache/`` to trigger a fresh build.

        Returns:
            Expanded string if a rule matched and the result differs from the
            input, otherwise ``None``.
        """
        term_stripped = term.strip()
        term_lower = term_stripped.lower()

        # 1. Dot-separated binomial where genus is already written in full
        #    e.g. "Plasmodium.falciparum" → "Plasmodium falciparum"
        m = re.fullmatch(r'([A-Z][a-z]+)\.([a-z][a-z0-9_-]+)', term_stripped)
        if m:
            expanded = f"{m.group(1)} {m.group(2)}"
            logger.debug(f"Dot-binomial fallback expansion: '{term_stripped}' → '{expanded}'")
            return expanded

        # 2. Alias dict lookup (case-insensitive key)
        aliases = getattr(self.config, 'term_aliases', {})
        if term_lower in {k.lower(): k for k in aliases}:
            for alias_key, alias_val in aliases.items():
                if alias_key.lower() == term_lower:
                    expanded = alias_val
                    if expanded.lower() != term_lower:
                        logger.debug(f"Alias expansion: '{term_stripped}' → '{expanded}'")
                        return expanded

        return None

    # ------------------------------------------------------------------
    # Custom synonym persistence & runtime registration
    # ------------------------------------------------------------------

    def _custom_synonyms_path(self) -> 'Path':
        """Return path to the custom-synonyms JSON file."""
        from pathlib import Path
        return Path(self.config.cache_dir) / "custom_synonyms.json"

    def _load_custom_synonyms(self, graph: OntologyGraph, ontology_id: str) -> int:
        """
        Load previously registered custom synonyms from disk and inject them
        into the graph nodes.

        Called inside ``load_ontology()`` alongside
        ``_inject_abbreviated_synonyms()``, so that any synonym registered at
        runtime in a previous session is automatically available when the
        ontology is loaded next time.

        The file format is::

            {
              "species": {
                "Plasmodium falciparum": ["p.falciparum", "pf 3d7"],
                ...
              },
              ...
            }

        Returns:
            Number of synonyms injected.
        """
        import json
        from pathlib import Path

        path = self._custom_synonyms_path()
        if not path.exists():
            return 0

        try:
            with open(path) as f:
                all_data: Dict[str, Dict[str, list]] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load custom synonyms from {path}: {e}")
            return 0

        ont_data = all_data.get(ontology_id, {})
        if not ont_data:
            return 0

        # Build node lookup by name (lowercase)
        node_by_name: Dict[str, OntologyNode] = {
            n.name.lower(): n for n in graph
        }

        injected = 0
        for node_name, synonyms in ont_data.items():
            node = node_by_name.get(node_name.lower())
            if node is None:
                logger.warning(
                    f"Custom synonym: node '{node_name}' not found in {ontology_id}"
                )
                continue
            existing_lower = {s.lower() for s in node.synonyms} | {node.name.lower()}
            for syn in synonyms:
                if syn.lower() not in existing_lower:
                    self._add_graph_synonym(
                        graph, node, syn, scope="CUSTOM"
                    )
                    existing_lower.add(syn.lower())
                    injected += 1

        if injected:
            logger.info(
                f"Loaded {injected} custom synonyms into {ontology_id} from {path}"
            )
        return injected

    def register_synonym(self,
                         synonym: str,
                         node_name: str,
                         ontology_id: str) -> bool:
        """
        Register a new synonym for an ontology term at runtime.

        This method does three things atomically:

        1. **Graph** — appends the synonym to the in-memory node so subsequent
           calls within the same session see it immediately.
        2. **Index** — embeds the synonym text and appends the vector to the
           live index (FAISS only; sklearn/annoy log a warning and require a
           full rebuild).
        3. **Disk** — persists the synonym to ``ontology_cache/custom_synonyms.json``
           so it is automatically loaded by ``_load_custom_synonyms`` in every
           future session.

        Args:
            synonym:    The new abbreviated / variant name  (e.g. ``"p.falciparum"``).
            node_name:  The primary name of the ontology node to attach it to
                        (e.g. ``"Plasmodium falciparum"``).
            ontology_id: Ontology identifier (e.g. ``'species'``, ``'cl'``).

        Returns:
            ``True`` on success, ``False`` if the node could not be found or
            the ontology was not loaded.

        Example::

            normalizer.register_synonym(
                synonym="p.falciparum",
                node_name="Plasmodium falciparum",
                ontology_id="species",
            )
        """
        import json
        from pathlib import Path

        if ontology_id not in self.graphs:
            logger.warning(f"register_synonym: ontology '{ontology_id}' not loaded")
            return False

        graph = self.graphs[ontology_id]
        index = self.indices.get(ontology_id)

        # Find the target node (match by primary name or existing synonyms)
        target_node: Optional[OntologyNode] = None
        for node in graph:
            if node.name.lower() == node_name.lower():
                target_node = node
                break
            if any(s.lower() == node_name.lower() for s in node.synonyms):
                target_node = node
                break

        if target_node is None:
            logger.warning(
                f"register_synonym: no node matching '{node_name}' in {ontology_id}"
            )
            return False

        existing_lower = {s.lower() for s in target_node.synonyms} | {target_node.name.lower()}
        if synonym.lower() in existing_lower:
            logger.debug(
                f"register_synonym: '{synonym}' already present for '{target_node.name}'"
            )
            return True  # Already registered — idempotent

        # 1. Update graph node
        self._add_graph_synonym(
            graph, target_node, synonym, scope="CUSTOM"
        )
        logger.info(
            f"Registered synonym '{synonym}' → '{target_node.name}' "
            f"[{ontology_id}] (node id: {target_node.id})"
        )

        # 2. Embed and append to live index
        if index is not None:
            try:
                new_emb = index._embed_texts([synonym])   # (1 x dim)
                new_idx = len(index.term_ids)
                index.term_ids.append(target_node.id)
                index.term_texts.append(synonym)
                if target_node.id in index.id_to_indices:
                    index.id_to_indices[target_node.id].append(new_idx)
                else:
                    index.id_to_indices[target_node.id] = [new_idx]

                # Append to backend index
                backend = self.config.index_backend
                if backend == 'faiss' and index.index is not None:
                    import faiss
                    import numpy as np
                    norm = np.linalg.norm(new_emb, axis=1, keepdims=True)
                    new_emb_norm = new_emb / np.maximum(norm, 1e-10)
                    index.index.add(new_emb_norm.astype('float32'))
                    logger.debug(f"Appended '{synonym}' to FAISS index")
                else:
                    logger.warning(
                        f"Incremental index update not supported for backend "
                        f"'{backend}'. Run 'python -m normalization.build_index' "
                        f"to rebuild the index with the new synonym."
                    )
            except Exception as e:
                logger.error(f"Failed to update index for synonym '{synonym}': {e}")

        # 3. Persist to disk
        path = self._custom_synonyms_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            existing_data: Dict[str, Dict[str, list]] = {}
            if path.exists():
                with open(path) as f:
                    existing_data = json.load(f)

            ont_block = existing_data.setdefault(ontology_id, {})
            node_syns = ont_block.setdefault(target_node.name, [])
            if synonym not in node_syns:
                node_syns.append(synonym)

            with open(path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            logger.info(f"Persisted custom synonym to {path}")

        except Exception as e:
            logger.error(f"Failed to persist synonym to {path}: {e}")

        return True

    def normalize(self,
                  term: str,
                  entity_type: Optional[str] = None,
                  ontology_id: Optional[str] = None,
                  top_k: Optional[int] = None) -> NormalizationResult:
        """
        Normalize a term against ontology.

        Abbreviated terms (e.g. ``p.falciparum``) are automatically expanded
        before the embedding lookup. Both the original and expanded form are
        searched; the result with the higher similarity score is returned.
        
        Args:
            term: Term to normalize
            entity_type: Type of entity (determines ontology)
            ontology_id: Specific ontology to use
            top_k: Number of candidates to return
            
        Returns:
            NormalizationResult with match details
        """
        top_k = top_k or self.config.top_k
        
        # Determine ontology
        if ontology_id is None and entity_type:
            ontology_id = self.get_ontology_for_entity(entity_type)
        
        if ontology_id is None:
            return NormalizationResult(
                original_term=term,
                entity_type=entity_type or '',
                is_normalized=False
            )
        
        if ontology_id not in self.indices:
            logger.warning(f"Ontology not loaded: {ontology_id}")
            return NormalizationResult(
                original_term=term,
                entity_type=entity_type or '',
                is_normalized=False
            )
        
        index = self.indices[ontology_id]
        graph = self.graphs.get(ontology_id)

        unimod_result = self._ptm_unimod_result(term, entity_type or '')
        if unimod_result is not None:
            return unimod_result

        # Deterministic lexical precedence avoids arbitrary FAISS tie ordering
        # and bypasses embeddings for resolved exact matches.
        if graph:
            lexical_result = self._exact_lexical_result(
                graph, term, ontology_id, entity_type or ''
            )
            if lexical_result is not None:
                return lexical_result

        def _search_and_build(query: str) -> Optional[NormalizationResult]:
            """Run index search for a single query string."""
            candidates = index.search(query, top_k=top_k)
            if not candidates:
                return None
            best_id, best_text, best_sim = candidates[0]
            node = graph.get_node(best_id) if graph else None
            ontology_name = node.name if node else best_text
            return NormalizationResult(
                original_term=term,
                ontology_id=best_id,
                ontology_name=ontology_name,
                similarity=best_sim,
                matched_text=best_text,
                candidates=candidates,
                entity_type=entity_type or '',
                is_normalized=best_sim >= self.config.similarity_threshold,
                normalization_method='semantic',
            )

        # --- Search original term ---
        result_original = _search_and_build(term)

        # --- Try abbreviation expansion ---
        expanded = self._expand_term(term)
        result_expanded = None
        if expanded:
            result_expanded = _search_and_build(expanded)

        # --- Pick the better result ---
        if result_expanded and (
            result_original is None
            or result_expanded.similarity > result_original.similarity
        ):
            result_expanded.expanded_term = expanded
            logger.info(
                f"Abbreviation expansion improved match: '{term}' → '{expanded}' "
                f"(sim {result_original.similarity:.3f} → {result_expanded.similarity:.3f})"
                if result_original else
                f"Abbreviation expansion found match: '{term}' → '{expanded}'"
            )
            return result_expanded

        if result_original is None:
            return NormalizationResult(
                original_term=term,
                entity_type=entity_type or '',
                is_normalized=False
            )

        return result_original
    
    def normalize_batch(self,
                        terms: List[str],
                        entity_type: Optional[str] = None,
                        ontology_id: Optional[str] = None) -> List[NormalizationResult]:
        """
        Normalize multiple terms.
        
        Args:
            terms: List of terms
            entity_type: Entity type for all terms
            ontology_id: Ontology for all terms
            
        Returns:
            List of NormalizationResult objects
        """
        if not terms:
            return []

        resolved_ontology = ontology_id
        if resolved_ontology is None and entity_type:
            resolved_ontology = self.get_ontology_for_entity(entity_type)

        if resolved_ontology is None or resolved_ontology not in self.indices:
            return [
                NormalizationResult(
                    original_term=term,
                    entity_type=entity_type or '',
                    is_normalized=False,
                )
                for term in terms
            ]

        index = self.indices[resolved_ontology]
        graph = self.graphs.get(resolved_ontology)
        top_k = self.config.top_k

        # Resolve unique exact primary labels without embeddings. Embed only
        # the unresolved original and expanded queries in one model call.
        exact_results: List[Optional[NormalizationResult]] = []
        queries: List[str] = []
        query_positions: List[Optional[Tuple[int, Optional[int], Optional[str]]]] = []
        for term in terms:
            lexical_result = self._ptm_unimod_result(
                term, entity_type or ''
            )
            if lexical_result is None and graph:
                lexical_result = self._exact_lexical_result(
                    graph, term, resolved_ontology, entity_type or ''
                )
            if lexical_result is not None:
                exact_results.append(lexical_result)
                query_positions.append(None)
                continue
            exact_results.append(None)
            original_pos = len(queries)
            queries.append(term)
            expanded = self._expand_term(term)
            expanded_pos: Optional[int] = None
            if expanded:
                expanded_pos = len(queries)
                queries.append(expanded)
            query_positions.append((original_pos, expanded_pos, expanded))

        all_candidates = (
            index.search_many(queries, top_k=top_k) if queries else []
        )

        def build_result(
            term: str,
            candidates: List[Tuple[str, str, float]],
        ) -> Optional[NormalizationResult]:
            if not candidates:
                return None
            best_id, best_text, best_sim = candidates[0]
            node = graph.get_node(best_id) if graph else None
            return NormalizationResult(
                original_term=term,
                ontology_id=best_id,
                ontology_name=node.name if node else best_text,
                similarity=best_sim,
                matched_text=best_text,
                candidates=candidates,
                entity_type=entity_type or '',
                is_normalized=best_sim >= self.config.similarity_threshold,
                normalization_method='semantic',
            )

        results: List[NormalizationResult] = []
        for term, exact_result, positions in zip(
            terms, exact_results, query_positions
        ):
            if exact_result is not None:
                results.append(exact_result)
                continue
            assert positions is not None
            original_pos, expanded_pos, expanded = positions
            original = build_result(term, all_candidates[original_pos])
            expanded_result = (
                build_result(term, all_candidates[expanded_pos])
                if expanded_pos is not None
                else None
            )
            if expanded_result and (
                original is None
                or expanded_result.similarity > original.similarity
            ):
                expanded_result.expanded_term = expanded
                results.append(expanded_result)
            elif original is not None:
                results.append(original)
            else:
                results.append(
                    NormalizationResult(
                        original_term=term,
                        entity_type=entity_type or '',
                        is_normalized=False,
                    )
                )

        return results
    
    def get_parent_terms(self,
                        term_id: str,
                        ontology_id: str,
                        max_depth: int = 3) -> List[Tuple[str, str]]:
        """
        Get parent terms for enrichment.
        
        Args:
            term_id: Ontology term ID
            ontology_id: Ontology identifier
            max_depth: Maximum depth to traverse
            
        Returns:
            List of (parent_id, parent_name) tuples
        """
        if ontology_id not in self.graphs:
            return []
        
        graph = self.graphs[ontology_id]
        ancestors = graph.get_ancestors(term_id, max_depth=max_depth)
        
        results = []
        for ancestor_id in ancestors:
            node = graph.get_node(ancestor_id)
            if node:
                results.append((ancestor_id, node.name))
        
        return results
    
    def disambiguate(self,
                    candidates: List[Tuple[str, str, float]],
                    context: str,
                    ontology_id: str) -> Tuple[str, str, float]:
        """
        Disambiguate between candidates using context.
        
        Args:
            candidates: List of (id, text, similarity) candidates
            context: Additional context text
            ontology_id: Ontology identifier
            
        Returns:
            Best matching (id, text, similarity) tuple
        """
        if len(candidates) <= 1:
            return candidates[0] if candidates else ('', '', 0.0)
        
        # Simple heuristic: prefer exact matches
        for cand_id, cand_text, sim in candidates:
            if cand_text.lower() == context.lower():
                return (cand_id, cand_text, sim)
        
        # Otherwise return highest similarity
        return candidates[0]
    
    def find_common_ancestor(self,
                            term1_id: str,
                            term2_id: str,
                            ontology_id: str) -> Optional[Tuple[str, str]]:
        """
        Find common ancestor of two terms.
        
        Useful for parent enrichment when two methods disagree.
        
        Args:
            term1_id: First term ID
            term2_id: Second term ID
            ontology_id: Ontology identifier
            
        Returns:
            (ancestor_id, ancestor_name) or None
        """
        if ontology_id not in self.graphs:
            return None
        
        graph = self.graphs[ontology_id]
        common_id = graph.get_common_ancestor(term1_id, term2_id)
        
        if common_id:
            node = graph.get_node(common_id)
            if node:
                return (common_id, node.name)
        
        return None


class MultiOntologyNormalizer:
    """
    Normalize terms across multiple ontologies.
    
    Automatically selects ontology based on entity type.
    
    Example:
        >>> normalizer = MultiOntologyNormalizer()
        >>> normalizer.load_all()
        >>> result = normalizer.normalize('liver', 'tissue')
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize multi-ontology normalizer.
        
        Args:
            config: Configuration
        """
        self.config = config or NormalizationConfig()
        self.normalizer = TermNormalizer(config)
    
    def load_all(self) -> None:
        """Load all configured ontologies."""
        self.normalizer.load_all_ontologies()
    
    def normalize(self,
                  term: str,
                  entity_type: str) -> NormalizationResult:
        """
        Normalize term using appropriate ontology.
        
        Args:
            term: Term to normalize
            entity_type: Entity type
            
        Returns:
            NormalizationResult
        """
        return self.normalizer.normalize(term, entity_type=entity_type)
    
    def normalize_extraction_result(self,
                                   extraction: Dict[str, Any]) -> Dict[str, NormalizationResult]:
        """
        Normalize all entities in an extraction result.
        
        Args:
            extraction: Dictionary with entity fields
            
        Returns:
            Dictionary mapping field to NormalizationResult
        """
        results = {}
        
        entity_types = ['species', 'cell_type', 'tissue', 'disease', 'organism']
        
        for entity_type in entity_types:
            if entity_type in extraction:
                term = extraction[entity_type]
                if term and isinstance(term, str):
                    results[entity_type] = self.normalize(term, entity_type)
        
        return results
