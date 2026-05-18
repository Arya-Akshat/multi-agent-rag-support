import uuid
from agents.orchestrator import graph
from models.conversation import Message

def test_graph_orchestrator():
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
        
        assert state1["current_agent"] == "technical"
        assert state1["extracted_entities"].get("cloud_provider") == "AWS"
        assert len(state1["handover_history"]) > 0
        assert state1["handover_history"][-1].target_agent == "technical"
        
        if state1["handover_history"]:
            print("Handover Triggered:", state1["handover_history"][-1].reason)
        
        print("\nInvoking graph turn 2 (Target Agent -> ?)...")
        state2 = graph.invoke(state1)
        print("New Current Agent:", state2["current_agent"])
        print("Latest Message:", state2["messages"][-1].content)
        
        assert len(state2["messages"]) > 1
        assert state2["current_agent"] in ["technical", "escalation"]
        
        print("\nSuccess! The graph is routing correctly.")
    except Exception as e:
        print(f"\nGraph invocation failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_graph_orchestrator()
