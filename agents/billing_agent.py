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
        logger.info("[BILLING] starting execution")
        logger.info(f"Billing agent processing conversation {state.conversation_id}")
        
        last_msg = state.last_user_message
        if not last_msg:
            return {"current_agent": "billing"}

        # Extract active billing intent
        from models.state import TriageIntent
        active_intent = None
        for i in state.intents:
            status = getattr(i, "status", None) or (i.get("status") if isinstance(i, dict) else "")
            itype = getattr(i, "type", "") or (i.get("type", "") if isinstance(i, dict) else "")
            if "billing" in itype and status == "pending":
                active_intent = i
                break
        
        if not active_intent:
            active_intent = TriageIntent(
                type="billing",
                priority=1,
                status="pending",
                active_intent_query=last_msg
            )
            
        active_type = getattr(active_intent, "type", "") or (active_intent.get("type") if isinstance(active_intent, dict) else "")
        logger.info(f"[BILLING] active_intent={active_type}")

        # Format input for the prompt using build_agent_task_context helper
        from agents.context_helper import build_agent_task_context
        history = self.format_history(state)
        task_context = build_agent_task_context(state, active_intent)

        user_input = (
            f"Conversation History:\n{history}\n\n"
            f"{self.billing_policy}\n\n"
            f"{task_context}"
        )
        
        # Invoke LLM
        logger.info("[BILLING] llm_invoked=True")
        response: BillingResponse = self.invoke_structured(
            prompt_variables={"user_input": user_input},
            response_model=BillingResponse
        )
        logger.info("[BILLING] response_generated=True")
        preview = response.response[:60].replace("\n", " ") + "..." if response.response else ""
        logger.info(f'[BILLING] response_preview="{preview}"')

        updates = {}

        # Track incoming background handover
        if state.current_agent != self.name:
            handover_event = HandoverEvent(
                source_agent=state.current_agent,
                target_agent=self.name,
                reason="Automatic background handover for pending billing intent",
                success=True,
                trace_id=state.trace_id
            )
            updates["current_agent"] = self.name
            updates["previous_agent"] = state.current_agent
            updates["handover_history"] = state.handover_history + [handover_event]

        # Mark active billing intent as completed
        new_intents = [i.model_copy() if hasattr(i, "model_copy") else i for i in state.intents]
        for idx, i in enumerate(new_intents):
            status = getattr(i, "status", None) or (i.get("status") if isinstance(i, dict) else "")
            itype = getattr(i, "type", "") or (i.get("type", "") if isinstance(i, dict) else "")
            if "billing" in itype and status == "pending":
                if hasattr(i, "status"):
                    i.status = "completed"
                elif isinstance(i, dict):
                    new_intents[idx]["status"] = "completed"
        updates["intents"] = new_intents

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
            updates["handover_history"] = updates.get("handover_history", state.handover_history) + [handover_event]
            return updates

        # 2. Handle normal response
        logger.info("Billing responding to user.")
        
        # Check if we should handover to Technical
        has_pending_tech = any("technical" in (getattr(x, "type", "") or (x.get("type", "") if isinstance(x, dict) else "")) and (getattr(x, "status", "") or (x.get("status", "") if isinstance(x, dict) else "")) == "pending" for x in new_intents)
        handovers = []
        if has_pending_tech:
            handover_event = HandoverEvent(
                source_agent=self.name,
                target_agent="technical",
                reason="Automatic handover to Technical Agent for SSO troubleshooting.",
                success=True,
                trace_id=state.trace_id
            )
            handovers.append(handover_event)

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
        if handovers:
            updates["handover_history"] = updates.get("handover_history", state.handover_history) + handovers

        return updates
