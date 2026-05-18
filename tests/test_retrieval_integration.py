from retrieval import Retriever, QueryRewriter
from models.conversation import Message

def test_query_rewriter_and_retriever():
    print("--- Testing QueryRewriter ---")
    rewriter = QueryRewriter()
    # Using an empty history, should gracefully handle it.
    query = rewriter.rewrite([Message(role="user", content="How do I reset my password?")])
    print("Rewritten query:", query)
    assert "reset" in query.lower() or "password" in query.lower()

    print("\n--- Testing Retriever ---")
    retriever = Retriever()
    assert retriever.is_available(), "Retriever should be available"
    
    results = retriever.retrieve(query, top_k=3)
    assert len(results) > 0, "Should retrieve at least one result"
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"ID: {res.article_id}")
        print(f"Title: {res.title}")
        print(f"Score: {res.relevance_score}")
        print(f"Snippet preview: {res.snippet[:100]}...")
        
        assert res.article_id is not None
        assert res.title is not None
        assert res.relevance_score >= 0.0

if __name__ == "__main__":
    test_query_rewriter_and_retriever()
