"""
retrieval/vectorstore.py — ChromaDB integration for vector storage.

Manages connection to the local persistent ChromaDB instance, collection
management, and basic CRUD operations for vectors.
"""

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app_logging.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Initialise and return a persistent ChromaDB client.
    The database files are stored locally at CHROMA_DB_PATH.
    """
    db_path = str(settings.chroma_db_resolved_path)
    
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=db_path,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    return client


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Get or create the main KB collection.
    """
    # We use cosine distance as our similarity metric. Since we normalize
    # embeddings in embeddings.py, inner product (ip) or cosine are equivalent.
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"}
    )


class VectorStore:
    """High-level wrapper around ChromaDB collection operations."""

    def __init__(self):
        try:
            self.client = get_chroma_client()
            self.collection = get_collection(self.client)
            self._available = True
        except Exception as e:
            logger.error(f"Failed to initialise ChromaDB: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if the vector store is healthy and ready."""
        return self._available

    def count(self) -> int:
        """Return the number of documents in the collection."""
        if not self._available:
            return 0
        return self.collection.count()

    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add a batch of chunks to the vector store."""
        if not self._available:
            raise RuntimeError("VectorStore is not available")
            
        # ChromaDB requires batches. We do it in one go assuming the batch
        # isn't massive (20 articles ~ 100 chunks is perfectly fine for one batch).
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Perform a vector similarity search.
        
        Returns a dict containing 'ids', 'distances', 'metadatas', 'documents'.
        Note: ChromaDB with cosine distance returns distances where 0 is identical,
        and higher is further apart. We will convert this to similarity score
        (1 - distance) in the Retriever.
        """
        if not self._available:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
            
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )


def reset_collection() -> None:
    """
    Utility to completely wipe the ChromaDB collection.
    Used by 'make reset-db' and test teardown.
    """
    try:
        client = get_chroma_client()
        client.delete_collection(name=settings.chroma_collection_name)
        logger.info(f"Deleted collection: {settings.chroma_collection_name}")
    except ValueError:
        # Collection didn't exist, which is fine
        pass
    except Exception as e:
        logger.error(f"Error resetting collection: {e}")
