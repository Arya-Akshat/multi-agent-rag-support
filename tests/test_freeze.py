import uuid
import time
from agents.orchestrator import graph
from models.conversation import Message

def test_pipeline_turn_execution_time():
    print("\n--- Running Scenario 1 (AWS Alerts) Turn Freeze Regression Test ---")
    
    # Scenario 1 Query representing a single pipeline turn
    initial_state = {
        "conversation_id": str(uuid.uuid4()),
        "trace_id": str(uuid.uuid4()),
        "messages": [Message(role="user", content="How do I reset my AWS alerts? I'm frustrated!")],
        "current_agent": "triage",
        "previous_agent": "",
        "extracted_entities": {},
        "handover_history": [],
        "escalated": False,
        "intents": []
    }
    
    start_time = time.time()
    state = graph.invoke(initial_state)
    elapsed_time = time.time() - start_time
    
    print(f"Graph execution completed in {elapsed_time:.3f} seconds.")
    
    # Assert execution completes under the hard 55s threshold
    assert elapsed_time < 55.0, f"Pipeline turn execution was too slow! Took {elapsed_time:.3f} seconds."
    print("Scenario 1 Freeze Regression Test Passed! (Execution is safely under 55s limit)")

if __name__ == "__main__":
    test_pipeline_turn_execution_time()
