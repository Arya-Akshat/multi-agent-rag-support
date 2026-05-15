"""
agents/escalation_agent.py — The Escalation Specialist.

Activated when automated resolution fails or the user demands a human.
Summarises the conversation context to assist the human support rep.
"""

from agents.base_agent import BaseAgent
from models.state import ConversationState, Message
from models.responses import EscalationResponse
from app_logging.logger import get_logger

logger = get_logger(__name__)


class EscalationAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="escalation")

    def __call__(self, state: ConversationState) -> dict:
        logger.info(f"Escalation agent processing conversation {state.conversation_id}")
        
        last_msg = state.last_user_message
        if not last_msg:
            return {"current_agent": "escalation"}

        # Format input for the prompt
        history = self.format_history(state)
        user_input = (
            f"Conversation History:\n{history}\n\n"
            f"Please generate a summary for the human agent, and a soothing response for the user."
        )
        
        # Invoke LLM
        response: EscalationResponse = self.invoke_structured(
            prompt_variables={"user_input": user_input},
            response_model=EscalationResponse
        )

        updates = {}

        # The message provided to the user (hardcoded as model doesn't have a field for it)
        msg = Message(
            role="assistant",
            content="I am escalating your issue to a human support representative who will be with you shortly. Thank you for your patience.",
            agent_name=self.name,
            metadata={
                "escalation_summary": response.model_dump()
            }
        )
        
        updates["messages"] = [msg]
        updates["escalated"] = True

        return updates
