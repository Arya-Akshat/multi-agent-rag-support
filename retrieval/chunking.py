"""
retrieval/chunking.py — Text chunking for the RAG pipeline.

Uses LangChain's RecursiveCharacterTextSplitter as requested in PROMPT.md.
Chunk size: 512, overlap: 64.
"""

from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(
    content: str,
    metadata: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Split a document's content into smaller chunks.

    Args:
        content: The raw markdown/text content of the article.
        metadata: Article metadata (id, title, category, tags).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Number of overlapping characters between adjacent chunks.

    Returns:
        A list of dicts, each containing the 'content' of the chunk
        and a 'metadata' dict that includes the chunk_index.
    """
    # RecursiveCharacterTextSplitter tries to split on paragraphs (\n\n),
    # then sentences (\n), then words, then characters, respecting the
    # chunk_size limit. This prevents splitting mid-sentence where avoidable.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    text_chunks = splitter.split_text(content)

    results = []
    for i, text in enumerate(text_chunks):
        # Create a fresh copy of the metadata for each chunk and inject the index
        chunk_meta = dict(metadata)
        chunk_meta["chunk_index"] = i
        
        # Tags are usually lists, but ChromaDB metadata requires str, int, or float.
        # We must serialize the tags list to a comma-separated string.
        if "tags" in chunk_meta and isinstance(chunk_meta["tags"], list):
            chunk_meta["tags"] = ",".join(chunk_meta["tags"])
            
        if "applies_to" in chunk_meta and isinstance(chunk_meta["applies_to"], list):
            chunk_meta["applies_to"] = ",".join(chunk_meta["applies_to"])

        results.append({
            "content": text,
            "metadata": chunk_meta
        })

    return results
