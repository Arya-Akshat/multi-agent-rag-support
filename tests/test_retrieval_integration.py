from retrieval import Retriever, QueryRewriter
from models.conversation import Message

print("--- Testing QueryRewriter ---")
rewriter = QueryRewriter()
# Using an empty history, should gracefully handle it.
query = rewriter.rewrite([Message(role="user", content="How do I reset my password?")])
print("Rewritten query:", query)

print("\n--- Testing Retriever ---")
retriever = Retriever()
if not retriever.is_available():
    print("Retriever unavailable")
else:
    results = retriever.retrieve(query, top_k=3)
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"ID: {res.article_id}")
        print(f"Title: {res.title}")
        print(f"Score: {res.relevance_score}")
        print(f"Snippet preview: {res.snippet[:100]}...")
