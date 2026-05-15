"""
agents/triage_agent.py — The entry point agent.

Responsible for greeting, intent extraction, entity extraction, and routing.
Never attempts to answer technical or billing questions directly.
"""

from agents.base_agent import BaseAgent
from handover.router import router
from models.state import ConversationState, Message, HandoverEvent
from models.responses import TriageResponse
from app_logging.logger import get_logger

logger = get_logger(__name__)


class TriageAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="triage")

    def __call__(self, state: ConversationState) -> dict:
        logger.info(f"Triage agent processing conversation {state.conversation_id}")
        
        last_msg = state.last_user_message
        if not last_msg:
            return {"current_agent": "triage"}

        history = self.format_history(state)
        user_input = f"Conversation History:\n{history}\n\nLast Message: {last_msg}"
        
        response: TriageResponse = self.invoke_structured(
            prompt_variables={"user_input": user_input},
            response_model=TriageResponse
        )

        # 1. Update extracted entities
        new_entities = state.extracted_entities.copy()
        if response.entities:
            new_entities.update(response.entities.model_dump(exclude_none=True))

        # 2. Determine Routing
        primary_intent = response.intents[0] if response.intents else "unknown"
        # We can use the router to validate the intent, or just trust the LLM's primary_agent
        target_agent = router.get_target_agent(primary_intent, self.name)
        
        # Fallback to LLM's choice if router doesn't know the intent
        if not target_agent:
             target_agent = response.routing_decision.primary_agent

        updates = {"extracted_entities": new_entities}

        if target_agent and target_agent != self.name:
            logger.info(f"Triage initiating handover to {target_agent}. Reason: {response.routing_decision.reason}")
            
            handover_event = HandoverEvent(
                source_agent=self.name,
                target_agent=target_agent,
                reason=response.routing_decision.reason,
                success=True,
                trace_id=state.trace_id
            )
            updates["current_agent"] = target_agent
            updates["handover_history"] = [handover_event]
        else:
            # If we are staying in triage, we MUST add a message to stop the loop
            # Triage can handle general greetings or unknown queries.
            updates["messages"] = [Message(role="assistant", content=response.routing_decision.reason, agent_name=self.name)]
            
        return updates
