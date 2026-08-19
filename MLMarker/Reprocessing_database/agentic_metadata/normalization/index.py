"""
Embedding-based index for ontology term search.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .ontology import OntologyGraph, OntologyNode
from .config import NormalizationConfig

logger = logging.getLogger(__name__)

# Suppress FAISS loader messages (AVX512/AVX2 fallback noise)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

# Module-level shared model — loaded once, reused by all OntologyIndex instances
_shared_model = None
_shared_tokenizer = None
_shared_device = None


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
        """Load SapBERT model for embeddings (shared across all instances)."""
        global _shared_model, _shared_tokenizer, _shared_device

        if self.model is not None:
            return

        # Reuse already-loaded model
        if _shared_model is not None:
            self.model = _shared_model
            self.tokenizer = _shared_tokenizer
            self._device = _shared_device
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required for embeddings")

        # Suppress all transformers/HF noise during model load
        import os as _os
        import transformers
        import warnings
        transformers.logging.set_verbosity_error()
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        _prev_tqdm = _os.environ.get("TQDM_DISABLE")
        _os.environ["TQDM_DISABLE"] = "1"

        device_name = "GPU" if self.config.use_gpu and torch.cuda.is_available() else "CPU"

        _shared_tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model)
        _shared_model = AutoModel.from_pretrained(self.config.embedding_model)

        # Restore tqdm
        if _prev_tqdm is None:
            _os.environ.pop("TQDM_DISABLE", None)
        else:
            _os.environ["TQDM_DISABLE"] = _prev_tqdm

        logger.info("SapBERT model loaded (%s)", device_name)

        if self.config.use_gpu and torch.cuda.is_available():
            _shared_device = torch.device('cuda')
            _shared_model = _shared_model.to(_shared_device)
        else:
            _shared_device = torch.device('cpu')

        _shared_model.eval()

        self.model = _shared_model
        self.tokenizer = _shared_tokenizer
        self._device = _shared_device
    
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
        # Collect all terms and texts
        self.term_ids = []
        self.term_texts = []
        self.id_to_indices = {}

        for node in graph:
            texts = node.all_names if include_synonyms else [node.name]
            indices = []
            for text in texts:
                self.term_ids.append(node.id)
                self.term_texts.append(text)
                indices.append(len(self.term_ids) - 1)
            self.id_to_indices[node.id] = indices

        # Compute embeddings
        self.embeddings = self._embed_texts(self.term_texts)

        # Build search index
        self._build_search_index()

        # Save if path provided
        if save_path:
            self.save(save_path)
    
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
        
        # Use Quantization for large indices
        if self.config.use_quantization and n_samples >= 10000:
            nlist = min(4096, int(n_samples / 30))
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFScalarQuantizer(
                quantizer, dim, nlist,
                faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT
            )
            # Suppress FAISS C++ clustering warnings on stderr
            import os as _os, sys as _sys
            _stderr_fd = _sys.stderr.fileno()
            _old_stderr = _os.dup(_stderr_fd)
            _devnull = _os.open(_os.devnull, _os.O_WRONLY)
            _os.dup2(_devnull, _stderr_fd)
            try:
                self.index.train(self.embeddings.astype('float32'))
            finally:
                _os.dup2(_old_stderr, _stderr_fd)
                _os.close(_old_stderr)
                _os.close(_devnull)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(self.embeddings.astype('float32'))
    
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
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")

    def load(self, path: str) -> None:
        """Load index from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embeddings']
        self.term_ids = data['term_ids']
        self.term_texts = data['term_texts']
        self.id_to_indices = data['id_to_indices']

        loaded_config = data.get('config')

        if (loaded_config and loaded_config.index_backend != 'faiss'
            and self.config.index_backend == 'faiss'):
            self._build_faiss_index()
            self.save(path)
            return

        if 'config' in data:
            self.config = data['config']

        if self.config.index_backend == 'faiss':
            import faiss
            index_path = path + ".faiss"
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            else:
                self._build_faiss_index()
                self.save(path)
        else:
            self._build_search_index()
    
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

        display_name = config.get_display_name(graph.ontology_id) if config else graph.ontology_id

        if os.path.exists(cache_path):
            try:
                index.load(cache_path)
                logger.info(
                    "  %-28s %6d terms (cached)",
                    display_name, len(index.term_ids),
                )
                return index
            except Exception as e:
                logger.warning(f"  Cache load failed for {display_name}: {e}")

        index.build(graph, save_path=cache_path)
        logger.info(
            "  %-28s %6d terms (built)",
            display_name, len(index.term_ids),
        )
        return index
