"""
retrieval/embeddings.py — Local embeddings using sentence-transformers.

Provides a unified interface for generating vector embeddings from text.
Using 'all-MiniLM-L6-v2' as it is fast, lightweight, and runs well on CPU.
"""

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Wrapper for local sentence-transformer embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialise the embedding model. This downloads the model on first run
        if it is not already cached locally.
        """
        # Load the model (runs on CPU automatically if no GPU is available)
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a vector."""
        # encode() returns a numpy array, we convert to a standard Python list of floats
        # which is what ChromaDB expects
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings into vectors."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
