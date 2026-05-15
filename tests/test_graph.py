import uuid
from agents.orchestrator import graph
from models.conversation import Message

print("--- Testing LangGraph Orchestrator ---")

# Initial state pointing to triage
initial_state = {
    "conversation_id": str(uuid.uuid4()),
    "trace_id": str(uuid.uuid4()),
    "messages": [Message(role="user", content="How do I reset my AWS alerts? I'm frustrated!")],
    "current_agent": "triage",
    "previous_agent": "",
    "extracted_entities": {},
    "handover_history": [],
    "escalated": False
}

try:
    print("\nInvoking graph turn 1 (Triage -> ?)...")
    state1 = graph.invoke(initial_state)
    print("New Current Agent:", state1["current_agent"])
    print("Extracted Entities:", state1["extracted_entities"])
    if state1["handover_history"]:
        print("Handover Triggered:", state1["handover_history"][-1].reason)
    
    print("\nInvoking graph turn 2 (Target Agent -> ?)...")
    # In a real app, we pause for user input. For testing, we just let the next agent run
    # immediately because state1 now points to the new agent (e.g., technical).
    state2 = graph.invoke(state1)
    print("New Current Agent:", state2["current_agent"])
    print("Latest Message:", state2["messages"][-1].content)
    
    print("\nSuccess! The graph is routing correctly.")
except Exception as e:
    print(f"\nGraph invocation failed: {e}")
    import traceback
    traceback.print_exc()
