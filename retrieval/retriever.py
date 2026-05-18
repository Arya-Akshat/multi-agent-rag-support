"""
retrieval/retriever.py — Core RAG retrieval logic.

Implements Hybrid Search (Vector + BM25 keyword) with Reciprocal Rank Fusion,
converting raw chunk results into Citation objects.
"""

from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from app_logging.logger import get_logger
from config.settings import settings
from models.conversation import Citation
from retrieval.embeddings import EmbeddingService
from retrieval.reranker import reciprocal_rank_fusion
from retrieval.vectorstore import VectorStore

logger = get_logger(__name__)


class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self._available = self.vector_store.is_available()

    def is_available(self) -> bool:
        """Check if the retrieval subsystem is functional."""
        return self._available and self.vector_store.count() > 0

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
        if not self._available:
            return []

        query_emb = self.embedding_service.embed_text(query)
        # We query for 2x top_k to give RRF more candidates to fuse
        raw_results = self.vector_store.search(query_emb, top_k=top_k * 2)
        
        results = []
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return results

        # Chroma returns lists of lists since it supports batched queries.
        # We only sent one query, so we take index 0.
        ids = raw_results["ids"][0]
        distances = raw_results["distances"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]

        for i in range(len(ids)):
            # Convert cosine distance to similarity score
            # Cosine distance: 0 is identical, 2 is opposite.
            # Similarity: 1 - (distance / 2) -> 1 is identical, 0 is opposite.
            similarity = 1.0 - (distances[i] / 2.0)
            
            results.append({
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "score": similarity
            })

        # Sort by score descending
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform sparse keyword search using BM25.
        
        Note: For a true production system at scale, you would use ElasticSearch or 
        OpenSearch. For this implementation, we pull all docs and run BM25 in memory.
        Since we have ~100 chunks total, this is extremely fast and perfectly fine.
        """
        if not self._available:
            return []
            
        try:
            # Pull all docs from ChromaDB
            # This is a hack for the prototype. In prod, use a real search engine.
            all_data = self.vector_store.client.get_collection(
                name=settings.chroma_collection_name
            ).get()
            
            if not all_data["ids"]:
                return []
                
            corpus = all_data["documents"]
            tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
            
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split(" ")
            
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Combine scores with docs
            scored_docs = []
            for i in range(len(all_data["ids"])):
                if doc_scores[i] > 0:  # Only include if there's a keyword match
                    scored_docs.append({
                        "id": all_data["ids"][i],
                        "content": all_data["documents"][i],
                        "metadata": all_data["metadatas"][i],
                        "score": doc_scores[i]
                    })
                    
            # Sort descending and take top_k * 2
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:top_k * 2]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def get_relevant_documents(self, query: str, top_k: int = None) -> List[Citation]:
        """Alias for retrieve() to support audit test scripts."""
        return self.retrieve(query, top_k)

    def retrieve(self, query: str, top_k: int = None) -> List[Citation]:
        """
        Execute a hybrid search (Vector + Keyword) and fuse results with RRF.
        Maps the fused chunk results into Citation objects.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        if not self.is_available():
            logger.warning("Retrieval requested but vector store is empty/unavailable.")
            return []

        logger.info(f"Retrieving for query: '{query}'")

        # 1. Execute parallel searches (sequential here for simplicity)
        vector_results = self._vector_search(query, top_k)
        keyword_results = self._keyword_search(query, top_k)
        
        # 2. Fuse with RRF
        fused_chunks = reciprocal_rank_fusion(
            ranked_lists=[vector_results, keyword_results],
            k=60
        )
        
        # 3. Take final top_k
        final_chunks = fused_chunks[:top_k]
        
        # 4. Map to Citation objects and deduplicate by article_id (merging snippets)
        citations_by_id = {}
        for chunk in final_chunks:
            meta = chunk["metadata"]
            art_id = meta.get("id", "unknown")
            title = meta.get("title", "Untitled Article")
            snippet = chunk["content"]
            relevance = chunk.get("relevance_score", 0.0)
            
            if art_id in citations_by_id:
                existing = citations_by_id[art_id]
                if snippet not in existing.snippet:
                    existing.snippet += "\n\n" + snippet
                if relevance > existing.relevance_score:
                    existing.relevance_score = relevance
            else:
                citations_by_id[art_id] = Citation(
                    article_id=art_id,
                    title=title,
                    snippet=snippet,
                    relevance_score=relevance,
                    url=f"https://kb.clouddash.io/articles/{art_id}"
                )
        citations = list(citations_by_id.values())
            
        logger.info(f"Retrieved {len(citations)} citations.")
        return citations
