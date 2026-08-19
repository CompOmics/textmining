"""SapBERT embedder for computing biomedical term similarity."""

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class SapBERTEmbedder:
    """Lazy-loaded SapBERT model for computing biomedical term similarity.

    Used for comparing predicted values against ground truth when
    case-insensitive string matching fails. Cosine similarity >= threshold
    (default 0.75) counts as a match.
    """

    def __init__(self, model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, D) L2-normalized numpy array."""
        import torch

        self._load()
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            output = self._model(**encoded) # forward pass
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy() # extract [CLS] token
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) 
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms # L2 normalized CLS embeddings

    def cosine_similarity_matrix(
        self, texts_a: list[str], texts_b: list[str]
    ) -> np.ndarray:
        """Pairwise cosine similarities. Returns (len(a), len(b)) matrix."""
        if not texts_a or not texts_b:
            return np.zeros((len(texts_a), len(texts_b)))
        vecs_a = self.embed(texts_a)
        vecs_b = self.embed(texts_b)
        return vecs_a @ vecs_b.T
