import uuid
from agents.orchestrator import graph
from models.conversation import Message

def run_scenario(name: str, query: str):
    print(f"\n=============================================")
    print(f"RUNNING SCENARIO: {name}")
    print(f"Query: \"{query}\"")
    print(f"=============================================")
    
    # Start in triage
    state = {
        "conversation_id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "messages": [Message(role="user", content=query)],
        "current_agent": "triage",
        "previous_agent": "",
        "extracted_entities": {},
        "handover_history": [],
        "escalated": False
    }
    
    # First turn (Triage -> Agent)
    print("\n[Turn 1] Invoking graph...")
    state = graph.invoke(state)
    print(f"-> Agent: {state['current_agent']}")
    print(f"-> Extracted Entities: {state['extracted_entities']}")
    if state['handover_history']:
        h = state['handover_history'][-1]
        print(f"-> Handover: {h.source_agent} -> {h.target_agent} (Reason: {h.reason})")
    
    # Check if there is an assistant message. If yes, the turn ended.
    last_msg = state['messages'][-1] if state['messages'] else None
    if last_msg and last_msg.role == 'assistant':
        print(f"-> Response: {last_msg.content}")
        if last_msg.citations:
            print(f"-> Citations used: {[c.title for c in last_msg.citations]}")
        return
        
    # Second turn (Agent -> Response or Escalation)
    print("\n[Turn 2] Invoking graph...")
    state = graph.invoke(state)
    print(f"-> Agent: {state['current_agent']}")
    print(f"-> Escalated: {state['escalated']}")
    if state['handover_history']:
        h = state['handover_history'][-1]
        print(f"-> Handover: {h.source_agent} -> {h.target_agent} (Reason: {h.reason})")
    
    last_msg = state['messages'][-1] if state['messages'] else None
    if last_msg:
        print(f"-> Response: {last_msg.content}")
        if last_msg.citations:
            print(f"-> Citations used: {[c.title for c in last_msg.citations]}")

if __name__ == "__main__":
    # Scenario 1: Single-Agent Resolution
    run_scenario(
        "Scenario 1 — Single-Agent Resolution", 
        "My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday. I'm on the Pro plan."
    )
    
    # Scenario 2: Cross-Agent Handover
    run_scenario(
        "Scenario 2 — Cross-Agent Handover",
        "I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue I reported last week has been resolved?"
    )
    
    # Scenario 3: Escalation to Human
    run_scenario(
        "Scenario 3 — Escalation to Human",
        "I've been charged twice for April. I need an immediate refund and I want to speak to a manager."
    )
    
    # Scenario 4: KB Retrieval Failure
    run_scenario(
        "Scenario 4 — KB Retrieval Failure",
        "Does CloudDash support integration with Datadog for cross-platform alerting?"
    )
