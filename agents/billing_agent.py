"""
agents/billing_agent.py — The Billing Specialist.

Handles invoices, upgrades, downgrades, and refunds.
Loads policy definitions directly from config/billing_plans.yaml to ensure
strict adherence to source-of-truth pricing, preventing LLM hallucinations.
"""

import pathlib
import yaml
from typing import Dict, Any

from agents.base_agent import BaseAgent
from handover.router import router
from models.state import ConversationState, Message, HandoverEvent
from models.responses import BillingResponse
from app_logging.logger import get_logger

logger = get_logger(__name__)


class BillingAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="billing")
        self.billing_policy = self._load_billing_policy()

    def _load_billing_policy(self) -> str:
        """Load and format the billing policy from YAML."""
        try:
            path = pathlib.Path("config/billing_plans.yaml")
            if not path.exists():
                return "Billing policy unavailable."
            
            # Read raw text to pass to LLM directly, or format it
            raw_yaml = path.read_text(encoding="utf-8")
            return f"Official CloudDash Billing Policy:\n\n{raw_yaml}"
        except Exception as e:
            logger.error(f"Failed to load billing policy: {e}")
            return "Billing policy unavailable."

    def __call__(self, state: ConversationState) -> dict:
        logger.info(f"Billing agent processing conversation {state.conversation_id}")
        
        last_msg = state.last_user_message
        if not last_msg:
            return {"current_agent": "billing"}

        # Format input for the prompt
        history = self.format_history(state)
        user_input = (
            f"Conversation History:\n{history}\n\n"
            f"{self.billing_policy}\n\n"
            f"Last User Message: {last_msg}"
        )
        
        # Invoke LLM
        response: BillingResponse = self.invoke_structured(
            prompt_variables={"user_input": user_input},
            response_model=BillingResponse
        )

        updates = {}

        # 1. Handle Escalation
        if response.escalate:
            logger.info("Billing agent escalating to human.")
            target_agent = "escalation"
            handover_event = HandoverEvent(
                source_agent=self.name,
                target_agent=target_agent,
                reason=response.escalation_reason or "Escalated by Billing Agent",
                success=True,
                trace_id=state.trace_id
            )
            updates["current_agent"] = target_agent
            updates["handover_history"] = [handover_event]
            return updates

        # 2. Handle normal response
        logger.info("Billing responding to user.")
        
        msg = Message(
            role="assistant",
            content=response.response,
            agent_name=self.name,
            metadata={
                "action_taken": response.action_taken,
                "plan_details": response.plan_details,
                "invoice_summary": response.invoice_summary,
                "policy_citations": response.policy_citations
            }
        )
        updates["messages"] = [msg]

        return updates
