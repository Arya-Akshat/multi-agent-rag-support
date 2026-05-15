"""
retrieval/reranker.py — Reciprocal Rank Fusion (RRF) for hybrid search.

Combines rankings from multiple retrieval methods (e.g. vector + keyword)
into a single ranked list without requiring a separate ML model.
"""

from typing import Any, Dict, List

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion (RRF) on multiple ranked lists.
    
    RRF score = sum(1 / (k + rank_in_list))
    where k is a constant (standard default is 60).
    
    Args:
        ranked_lists: A list of lists, where each inner list contains chunk dicts.
                      Each chunk dict MUST contain an 'id' key.
        k: The smoothing constant.
        
    Returns:
        A single list of chunk dicts, sorted by descending RRF score.
        The returned dicts will have a new 'rrf_score' key.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}
    
    # Calculate RRF scores
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            chunk_id = chunk["id"]
            
            # Store the chunk data if we haven't seen it yet
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
                rrf_scores[chunk_id] = 0.0
                
            # Add to the RRF score
            # rank is 0-indexed, so we add 1 for the formula
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
            
    # Sort chunks by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # Build the final list
    fused_results = []
    for chunk_id in sorted_ids:
        chunk_data = dict(chunk_map[chunk_id])  # Copy to avoid mutating original
        chunk_data["rrf_score"] = rrf_scores[chunk_id]
        # Overwrite relevance_score with the normalized RRF score
        # RRF max possible score for 2 lists is (1/61) + (1/61) = 0.0327
        # We just pass it through directly, or we can normalize.
        # For simplicity, we just assign it to relevance_score.
        chunk_data["relevance_score"] = rrf_scores[chunk_id]
        fused_results.append(chunk_data)
        
    return fused_results
