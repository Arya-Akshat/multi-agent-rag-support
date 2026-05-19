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

def test_scenario_2_complete():
    print("--- Test A: Scenario 2 Multi-Intent Verification ---")
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
    
    # Apply centralized validation
    from models.state import ConversationState
    from guardrails.validators import validate_workflow_execution
    pyd_state = ConversationState(**state)
    validate_workflow_execution(pyd_state)
    
    # Assert nodes execution
    assistant_msgs = [m for m in pyd_state.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 2, f"Expected exactly 2 assistant messages, got {len(assistant_msgs)}"
    assert assistant_msgs[0].agent_name == "technical"
    assert assistant_msgs[1].agent_name == "billing"
    assert not pyd_state.escalated
    print("Scenario 2 Complete Integration Test Passed!")

def test_datadog_grounding():
    print("--- Test B: Datadog Grounding Verification ---")
    initial_state = {
        "conversation_id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "messages": [Message(role="user", content="Does CloudDash support integration with Datadog for cross-platform alerting?")],
        "current_agent": "triage",
        "previous_agent": "",
        "extracted_entities": {},
        "handover_history": [],
        "escalated": False,
        "intents": []
    }
    
    state = graph.invoke(initial_state)
    
    # Apply centralized validation
    from models.state import ConversationState
    from guardrails.validators import validate_workflow_execution
    pyd_state = ConversationState(**state)
    validate_workflow_execution(pyd_state)
    
    # Assert agent response
    assistant_msgs = [m for m in pyd_state.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1
    datadog_res = assistant_msgs[-1].content
    
    # Assert no unsupported claim hallucinated
    assert "CloudDash does not support" not in datadog_res
    assert "I could not find information about this feature in the CloudDash knowledge base." in datadog_res
    print("Datadog Grounding Integration Test Passed!")

def test_enterprise_okta_split():
    print("--- Test C: Enterprise + Okta Split Verification ---")
    initial_state = {
        "conversation_id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "messages": [Message(role="user", content="I want to upgrade from Pro to Enterprise to get audit log exports, but first does CloudDash support SSO integration with Okta?")],
        "current_agent": "triage",
        "previous_agent": "",
        "extracted_entities": {},
        "handover_history": [],
        "escalated": False,
        "intents": []
    }
    
    state = graph.invoke(initial_state)
    
    # Apply centralized validation
    from models.state import ConversationState
    from guardrails.validators import validate_workflow_execution
    pyd_state = ConversationState(**state)
    validate_workflow_execution(pyd_state)
    
    assistant_msgs = [m for m in pyd_state.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 2, f"Expected at least 2 assistant messages, got {len(assistant_msgs)}"
    
    tech_msg = next(m for m in assistant_msgs if m.agent_name == "technical")
    billing_msg = next(m for m in assistant_msgs if m.agent_name == "billing")
    
    # Verify technical agent strictly discussed SSO/Okta and omitted billing concepts
    assert "sso" in tech_msg.content.lower() or "okta" in tech_msg.content.lower()
    assert "audit log export" not in tech_msg.content.lower()
    assert "pricing" not in tech_msg.content.lower()
    
    # Verify billing agent strictly discussed enterprise/upgrade and omitted SAML troubleshooting/debugging
    assert "enterprise" in billing_msg.content.lower() or "upgrade" in billing_msg.content.lower()
    assert "verify idp" not in billing_msg.content.lower()
    assert "signing certificate" not in billing_msg.content.lower()
    
    print("Enterprise + Okta Domain Isolation Split Test Passed!")

if __name__ == "__main__":
    test_graph_orchestrator()
    test_multi_intent_orchestration()
    test_scenario_2_complete()
    test_datadog_grounding()
    test_enterprise_okta_split()
