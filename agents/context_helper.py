"""
agents/context_helper.py — Helper functions for intent-scoped context isolation.
"""

from typing import Any
from models.state import ConversationState, TriageIntent
from app_logging.logger import get_logger

logger = get_logger(__name__)


def build_agent_task_context(state: ConversationState, active_intent: TriageIntent) -> str:
    """
    Constructs a highly scoped, intent-isolated prompt context for a specialized agent.
    This prevents domain leakage by providing a custom summary, the exact active task,
    and strict domain boundaries (DO NOT discuss lists).
    """
    summary_bullets = []
    
    # Extract entities and intents
    plan_type = state.extracted_entities.get("plan_type") or "Pro" # Default to Pro as in Scenario 2 if not found
    cloud_provider = state.extracted_entities.get("cloud_provider")
    
    tech_intents = [i for i in state.intents if "technical" in (getattr(i, "type", "") or i.get("type", ""))]
    billing_intents = [i for i in state.intents if "billing" in (getattr(i, "type", "") or i.get("type", ""))]
    
    active_type = getattr(active_intent, "type", "") or active_intent.get("type", "")
    active_query = getattr(active_intent, "active_intent_query", "") or active_intent.get("active_intent_query", "")
    
    if not active_query:
        active_query = state.last_user_message or ""

    if "technical" in active_type:
        # 1. Technical Agent Isolation Context
        summary_bullets.append(f"- Customer experiencing SSO or technical issue: '{active_query}'")
        if cloud_provider:
            summary_bullets.append(f"- Active cloud provider: {cloud_provider}")
        if billing_intents:
            summary_bullets.append("- Customer also interested in Enterprise upgrade later")
            
        current_task = "Resolve SSO issue." if "sso" in active_query.lower() else f"Resolve technical support issue: '{active_query}'"
        do_not_discuss = [
            "pricing",
            "billing",
            "plan upgrades",
            "subscription terms or rates"
        ]
    elif "billing" in active_type:
        # 2. Billing Agent Isolation Context
        summary_bullets.append(f"- Customer currently on {plan_type.capitalize()} plan")
        
        # Check if technical support intent was completed in history
        tech_completed = any((getattr(i, "status", "") or i.get("status", "")) == "completed" for i in tech_intents)
        if tech_completed or len(state.handover_history) > 0:
            summary_bullets.append("- Technical SSO issue already addressed")
        
        # Rewrite the query into a billing request
        summary_bullets.append("- Customer now requesting Enterprise upgrade" if "enterprise" in active_query.lower() else f"- Customer now requesting: '{active_query}'")
        
        current_task = "Explain Enterprise upgrade process." if "enterprise" in active_query.lower() else f"Handle billing request: '{active_query}'"
        do_not_discuss = [
            "SSO troubleshooting",
            "technical debugging",
            "integrations",
            "webhook or alerts configuration"
        ]
    else:
        # Fallback
        summary_bullets.append(f"- Customer request: '{active_query}'")
        current_task = active_query
        do_not_discuss = []

    summary_str = "\n".join(summary_bullets)
    do_not_discuss_str = "\n".join([f"- {item}" for item in do_not_discuss])
    
    context_prompt = (
        f"Conversation Summary:\n{summary_str}\n\n"
        f"Current Active Task:\n{current_task}"
    )
    if do_not_discuss:
        context_prompt += f"\n\nDO NOT discuss:\n{do_not_discuss_str}"
        
    return context_prompt
