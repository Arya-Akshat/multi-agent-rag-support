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

def test_multi_intent_orchestration():
    print("--- Testing Multi-Intent Sequential Orchestration ---")
    initial_state = {
        "conversation_id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "messages": [Message(role="user", content="I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue I reported last week has been resolved?")],
        "current_agent": "triage",
        "previous_agent": "",
        "extracted_entities": {},
        "handover_history": [],
        "escalated": False,
        "intents": []
    }

    state = graph.invoke(initial_state)
    
    # 1. Assert triage initialized intents properly
    assert "intents" in state
    intents = state["intents"]
    assert len(intents) >= 2
    
    # Check that both technical and billing intents are represented
    types = [i.type if hasattr(i, "type") else i.get("type") for i in intents]
    assert any("technical" in t for t in types)
    assert any("billing" in t for t in types)
    
    # 2. Check statuses - both should be completed because sequential execution chained them in one turn!
    statuses = [i.status if hasattr(i, "status") else i.get("status") for i in intents]
    assert all(s == "completed" for s in statuses)
    
    # 3. Assert escalation was suppressed
    assert not state.get("escalated", False)
    
    # 4. Assert handover history tracks the chain
    assert len(state["handover_history"]) >= 2
    
    # 5. Check both responses are in message history
    assistant_msgs = [m for m in state["messages"] if m.role == "assistant"]
    assert len(assistant_msgs) >= 2
    assert any(m.agent_name == "technical" for m in assistant_msgs)
    assert any(m.agent_name == "billing" for m in assistant_msgs)
    print("Multi-Intent Sequential Orchestration verified successfully!")

if __name__ == "__main__":
    test_graph_orchestrator()
    test_multi_intent_orchestration()
