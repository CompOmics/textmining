"""
Embedding-based index for ontology term search.
"""

import os
import pickle
import logging
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .ontology import OntologyGraph, OntologyNode
from .ontology import OntologyGraph, OntologyNode
from .config import NormalizationConfig

logger = logging.getLogger(__name__)


class OntologyIndex:
    """
    Searchable index for ontology terms using SapBERT embeddings.
    
    Uses semantic embeddings for similarity-based term matching.
    Supports sklearn, faiss, and annoy backends.
    
    Example:
        >>> config = NormalizationConfig()
        >>> index = OntologyIndex(config)
        >>> index.build(ontology_graph)
        >>> matches = index.search("epithelial cell", top_k=5)
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize index.
        
        Args:
            config: Normalization configuration
        """
        self.config = config or NormalizationConfig()
        self.model = None
        self.tokenizer = None
        self.embeddings: Optional[np.ndarray] = None
        self.term_ids: List[str] = []  # Maps index position to term ID
        self.term_texts: List[str] = []  # Maps index position to text
        self.id_to_indices: Dict[str, List[int]] = {}  # Maps term ID to embedding indices
        self.index = None  # Backend-specific index
        self._device = None
    
    def _load_model(self) -> None:
        """Load SapBERT model for embeddings."""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required for embeddings")
        
        # Resolve the device before cache lookup so CPU and GPU models are not
        # accidentally shared with one another.
        if self.config.use_gpu and torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        cache_key = (self.config.embedding_model, str(self._device))
        with self._model_cache_lock:
            cached = self._model_cache.get(cache_key)
            if cached is None:
                logger.info(
                    "Loading shared embedding model: %s on %s",
                    self.config.embedding_model,
                    self._device,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.embedding_model
                )
                model = AutoModel.from_pretrained(self.config.embedding_model)
                model = model.to(self._device)
                model.eval()
                self._model_cache[cache_key] = (tokenizer, model)
                cached = (tokenizer, model)
            else:
                logger.debug(
                    "Reusing shared embedding model: %s on %s",
                    self.config.embedding_model,
                    self._device,
                )

        self.tokenizer, self.model = cached
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings (num_texts x embedding_dim)
        """
        import torch
        
        self._load_model()
        
        all_embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def build(self, 
              graph: OntologyGraph,
              include_synonyms: bool = True,
              save_path: Optional[str] = None) -> None:
        """
        Build index from ontology graph.
        
        Args:
            graph: Ontology graph to index
            include_synonyms: Whether to include synonyms
            save_path: Path to save index
        """
        logger.info(f"Building index for {len(graph)} terms...")
        
        # Collect all terms and texts
        self.term_ids = []
        self.term_texts = []
        self.id_to_indices = {}
        
        for node in graph:
            if include_synonyms:
                texts = node.all_names
            else:
                texts = [node.name]
            
            start_idx = len(self.term_ids)
            indices = []
            
            for text in texts:
                self.term_ids.append(node.id)
                self.term_texts.append(text)
                indices.append(len(self.term_ids) - 1)
            
            self.id_to_indices[node.id] = indices
        
        logger.info(f"Computing embeddings for {len(self.term_texts)} terms...")
        
        # Compute embeddings
        self.embeddings = self._embed_texts(self.term_texts)
        
        # Build search index
        self._build_search_index()
        
        # Save if path provided
        if save_path:
            self.save(save_path)
        
        logger.info("Index built successfully")
    
    def _build_search_index(self) -> None:
        """Build backend-specific search index."""
        backend = self.config.index_backend
        
        if backend == 'faiss':
            self._build_faiss_index()
        elif backend == 'annoy':
            self._build_annoy_index()
        else:  # sklearn
            self._build_sklearn_index()
    
    def _build_sklearn_index(self) -> None:
        """Build sklearn NearestNeighbors index."""
        from sklearn.neighbors import NearestNeighbors
        
        self.index = NearestNeighbors(
            n_neighbors=self.config.top_k,
            metric='cosine',
            algorithm='brute'
        )
        self.index.fit(self.embeddings)
        logger.info("Built sklearn index")
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not available, falling back to sklearn")
            self.config.index_backend = "sklearn"
            return self._build_sklearn_index()
        
        dim = self.embeddings.shape[1]
        n_samples = self.embeddings.shape[0]
        
        # Normalize for cosine similarity (Inner Product)
        norms = getattr(self, '_cached_norms', None)
        if norms is None:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / np.maximum(norms, 1e-10)
        
        # Use Quantization for large indices check
        if self.config.use_quantization and n_samples >= 10000:
            logger.info("Using Quantized FAISS Index (IVF + ScalarQuantizer)")
            # IVF search with Scalar Quantizer (8-bit)
            # nlist: number of centroids (clusters)
            nlist = min(4096, int(n_samples / 30))
            
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFScalarQuantizer(
                quantizer, dim, nlist, 
                faiss.ScalarQuantizer.QT_8bit, 
                faiss.METRIC_INNER_PRODUCT
            )
            
            # Train index
            logger.info(f"Training FAISS index with {n_samples} vectors...")
            self.index.train(self.embeddings.astype('float32'))
        else:
            logger.info("Using Flat FAISS Index (Exact Search)")
            self.index = faiss.IndexFlatIP(dim)
            
        self.index.add(self.embeddings.astype('float32'))
        logger.info("Built FAISS index")
    
    def _build_annoy_index(self) -> None:
        """Build Annoy index."""
        try:
            from annoy import AnnoyIndex
        except ImportError:
            logger.warning("Annoy not available, falling back to sklearn")
            return self._build_sklearn_index()
        
        dim = self.embeddings.shape[1]
        self.index = AnnoyIndex(dim, 'angular')
        
        for i, emb in enumerate(self.embeddings):
            self.index.add_item(i, emb)
        
        self.index.build(10)  # 10 trees
        logger.info("Built Annoy index")
    
    def search(self, 
               query: str,
               top_k: Optional[int] = None) -> List[Tuple[str, str, float]]:
        """
        Search for similar terms.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (term_id, term_text, similarity) tuples
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build() first.")
        
        top_k = top_k or self.config.top_k
        
        # Embed query
        query_emb = self._embed_texts([query])[0]
        
        # Search
        results = self._search_index(query_emb, top_k)
        
        return results

    def search_many(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Tuple[str, str, float]]]:
        """Search several query strings using one batched embedding pass."""
        if self.embeddings is None:
            raise ValueError("Index not built. Call build() first.")
        if not queries:
            return []

        resolved_top_k = top_k or self.config.top_k
        query_embeddings = self._embed_texts(queries)
        return [
            self._search_index(query_embedding, resolved_top_k)
            for query_embedding in query_embeddings
        ]
    
    def _search_index(self, 
                      query_emb: np.ndarray,
                      top_k: int) -> List[Tuple[str, str, float]]:
        """Search using backend-specific method."""
        backend = self.config.index_backend
        
        if backend == 'faiss' and hasattr(self.index, 'search'):
            return self._search_faiss(query_emb, top_k)
        elif backend == 'annoy' and hasattr(self.index, 'get_nns_by_vector'):
            return self._search_annoy(query_emb, top_k)
        else:
            return self._search_sklearn(query_emb, top_k)
    
    def _search_sklearn(self,
                       query_emb: np.ndarray,
                       top_k: int) -> List[Tuple[str, str, float]]:
        """Search using sklearn."""
        # Temporarily adjust n_neighbors if needed
        self.index.n_neighbors = min(top_k, len(self.term_ids))
        
        distances, indices = self.index.kneighbors([query_emb])
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            similarity = 1 - dist  # Cosine distance to similarity
            results.append((
                self.term_ids[idx],
                self.term_texts[idx],
                float(similarity)
            ))
        
        return results
    
    def _search_faiss(self,
                     query_emb: np.ndarray,
                     top_k: int) -> List[Tuple[str, str, float]]:
        """Search using FAISS."""
        # Normalize query
        query_norm = query_emb / np.linalg.norm(query_emb)
        query_norm = query_norm.reshape(1, -1).astype('float32')
        
        similarities, indices = self.index.search(query_norm, top_k)
        
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx >= 0:  # FAISS returns -1 for not enough results
                results.append((
                    self.term_ids[idx],
                    self.term_texts[idx],
                    float(sim)
                ))
        
        return results
    
    def _search_annoy(self,
                     query_emb: np.ndarray,
                     top_k: int) -> List[Tuple[str, str, float]]:
        """Search using Annoy."""
        indices, distances = self.index.get_nns_by_vector(
            query_emb, top_k, include_distances=True
        )
        
        results = []
        for idx, dist in zip(indices, distances):
            # Angular distance to similarity
            similarity = 1 - (dist ** 2) / 2
            results.append((
                self.term_ids[idx],
                self.term_texts[idx],
                float(similarity)
            ))
        
        return results
    
    def save(self, path: str) -> None:
        """Save index to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save metadata (pickle)
        data = {
            'term_ids': self.term_ids,
            'term_texts': self.term_texts,
            'id_to_indices': self.id_to_indices,
            'config': self.config,
            # Always save embeddings for now to allow backend switching/rebuilding
            # Optimally we could drop this for FAISS if space is critical
            'embeddings': self.embeddings, 
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
        # Save FAISS index separately
        if self.config.index_backend == 'faiss' and self.index:
            try:
                import faiss
                faiss.write_index(self.index, path + ".faiss")
                logger.info(f"Saved FAISS index to {path}.faiss")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
        
        logger.info(f"Saved metadata to {path}")
    
    def load(self, path: str) -> None:
        """Load index from file."""
        logger.info(f"Loading index from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.term_ids = data['term_ids']
        self.term_texts = data['term_texts']
        self.id_to_indices = data['id_to_indices']
        
        loaded_config = data.get('config')
        
        # Automatic Upgrade Logic:
        # If we loaded an sklearn index but want FAISS, force a rebuild using embeddings
        if (loaded_config and loaded_config.index_backend != 'faiss' 
            and self.config.index_backend == 'faiss'):
            logger.warning("Detected legacy index backend. Upgrading to FAISS...")
            self._build_faiss_index()
            # Save immediately to complete upgrade
            self.save(path)
            return

        # Use loaded config if no override needed
        if 'config' in data:
            self.config = data['config']
            
        # Load FAISS index if applicable
        if self.config.index_backend == 'faiss':
            import faiss
            index_path = path + ".faiss"
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info("Loaded FAISS index from disk")
            else:
                logger.warning("FAISS index file missing, rebuilding from embeddings...")
                self._build_faiss_index()
                self.save(path)
        else:
            # Rebuild sklearn index
            self._build_search_index()
        
        logger.info(f"Index ready ({len(self.term_ids)} terms)")
    
    @classmethod
    def load_or_build(cls,
                      graph: OntologyGraph,
                      cache_path: str,
                      config: Optional[NormalizationConfig] = None) -> 'OntologyIndex':
        """
        Load from cache or build new index.
        
        Args:
            graph: Ontology graph
            cache_path: Path to cache file
            config: Configuration
            
        Returns:
            OntologyIndex instance
        """
        index = cls(config)
        
        if os.path.exists(cache_path):
            try:
                index.load(cache_path)
                return index
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        index.build(graph, save_path=cache_path)
        return index
    # All ontology indexes in a normalization process use the same embedding
    # model.  Keeping a cache here avoids loading and copying SapBERT to the GPU
    # once per ontology (up to 19 times in the full pipeline).
    _model_cache: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
    _model_cache_lock = threading.Lock()
