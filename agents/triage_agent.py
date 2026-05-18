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

        # 2. Save the intents to state
        intents_list = response.intents or []
        updates = {
            "extracted_entities": new_entities,
            "intents": intents_list
        }

        # 3. Determine Routing based on first pending intent
        pending = [i for i in intents_list if i.status == "pending"]
        target_agent = None
        
        if pending:
            # Sort by priority
            pending.sort(key=lambda x: x.priority)
            primary_intent_obj = pending[0]
            primary_intent = primary_intent_obj.type
            
            # Map primary_intent to agent name
            if "technical" in primary_intent:
                target_agent = "technical"
            elif "billing" in primary_intent:
                target_agent = "billing"
            elif "escalation" in primary_intent:
                target_agent = "escalation"
        
        # Fallback to LLM's explicit choice if target_agent couldn't be determined from intents
        if not target_agent:
            target_agent = response.routing_decision.primary_agent

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
