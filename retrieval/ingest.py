"""
retrieval/ingest.py — Coordination of article loading, chunking, and storage.

This module reads the raw JSON articles from disk, chunks them,
embeds them, and inserts them into ChromaDB.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from app_logging.logger import get_logger
from retrieval.chunking import chunk_document
from retrieval.embeddings import EmbeddingService
from retrieval.vectorstore import VectorStore

logger = get_logger(__name__)


def load_articles_from_disk(data_dir: str = "knowledge_base/data/articles") -> List[Dict[str, Any]]:
    """Read all JSON articles from the specified directory."""
    articles = []
    path = Path(data_dir)
    
    if not path.exists():
        logger.warning(f"KB data directory {data_dir} does not exist.")
        return articles

    for file_path in path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                article = json.load(f)
                # Ensure minimal required fields
                if "id" in article and "content" in article:
                    articles.append(article)
                else:
                    logger.warning(f"Skipping {file_path.name}: missing 'id' or 'content'")
        except Exception as e:
            logger.error(f"Failed to read {file_path.name}: {e}")

    logger.info(f"Loaded {len(articles)} articles from disk.")
    return articles


def ingest_articles(articles: List[Dict[str, Any]]) -> int:
    """
    Process a list of articles: chunk, embed, and store in vector DB.
    Returns the number of chunks successfully ingested.
    """
    if not articles:
        return 0

    vector_store = VectorStore()
    if not vector_store.is_available():
        logger.error("Cannot ingest: VectorStore is unavailable.")
        return 0

    embedding_service = EmbeddingService()
    
    all_chunks = []
    
    # Step 1: Chunk all articles
    for article in articles:
        content = article.pop("content")
        # The remaining dictionary becomes the metadata
        chunks = chunk_document(content=content, metadata=article)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} chunks from {len(articles)} articles.")

    if not all_chunks:
        return 0

    # Step 2: Prepare batches for ChromaDB
    # (Chroma requires lists of IDs, embeddings, documents, and metadatas)
    ids = []
    documents = []
    metadatas = []
    
    for chunk in all_chunks:
        # ID format: article_id#chunk_index
        chunk_id = f"{chunk['metadata']['id']}#{chunk['metadata']['chunk_index']}"
        ids.append(chunk_id)
        documents.append(chunk["content"])
        metadatas.append(chunk["metadata"])

    # Step 3: Embed documents
    logger.info("Generating embeddings for chunks...")
    embeddings = embedding_service.embed_batch(documents)

    # Step 4: Insert into Vector Store
    logger.info("Inserting into ChromaDB...")
    vector_store.add_chunks(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    logger.info(f"Successfully ingested {len(all_chunks)} chunks.")
    return len(all_chunks)
