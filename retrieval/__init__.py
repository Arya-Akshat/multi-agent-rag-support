"""
retrieval/__init__.py — Public exports for the retrieval package.
"""

from retrieval.chunking import chunk_document
from retrieval.embeddings import EmbeddingService
from retrieval.ingest import ingest_articles, load_articles_from_disk
from retrieval.query_rewriter import QueryRewriter
from retrieval.retriever import Retriever
from retrieval.vectorstore import VectorStore, reset_collection

__all__ = [
    "chunk_document",
    "EmbeddingService",
    "ingest_articles",
    "load_articles_from_disk",
    "QueryRewriter",
    "Retriever",
    "VectorStore",
    "reset_collection"
]
