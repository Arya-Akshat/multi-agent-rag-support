"""
models/responses.py — Structured output models for every agent type.

These Pydantic models enforce the exact JSON contract that each agent MUST
return. They are used both for LLM structured output parsing (via
`with_structured_output`) and for API response serialisation.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------
class RoutingDecision(BaseModel):
    """The routing decision produced by the Triage Agent."""

    primary_agent: str = Field(description="Agent to handle this query")
    secondary_agents: list[str] = Field(
        default_factory=list,
        description="Additional agents to involve (e.g., for multi-intent)",
    )
    reason: str = Field(description="One-sentence explanation of routing choice")


class TriageEntities(BaseModel):
    """Entities extracted by the Triage Agent from the user's message."""

    customer_id: Optional[str] = Field(default=None)
    cloud_provider: Optional[str] = Field(default=None)
    plan_type: Optional[str] = Field(default=None)
    issue_type: Optional[str] = Field(default=None)
    urgency: Optional[str] = Field(
        default="low",
        description="Urgency level",
    )
    sentiment: Optional[str] = Field(
        default="neutral",
        description="Customer sentiment",
    )


# ---------------------------------------------------------------------------
# Triage Agent structured output
# ---------------------------------------------------------------------------
from models.state import TriageIntent

class TriageResponse(BaseModel):
    """The complete structured output from the Triage Agent."""

    intents: list[TriageIntent] = Field(
        description="List of detected intents with priority and status"
    )
    entities: TriageEntities
    routing_decision: RoutingDecision
    requires_multi_step: bool = Field(
        default=False,
        description="True when multiple agents must process this query sequentially",
    )


# ---------------------------------------------------------------------------
# Technical Agent structured output
# ---------------------------------------------------------------------------
class TechnicalCitation(BaseModel):
    """A KB article citation produced by the Technical Agent."""

    article_id: str
    title: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class TechnicalResponse(BaseModel):
    """The complete structured output from the Technical Support Agent."""

    response: str = Field(description="Human-readable answer to the customer's query")
    citations: list[TechnicalCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Agent's confidence (0–1)")
    escalate: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(default=None)
    suggested_next_steps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Billing Agent structured output
# ---------------------------------------------------------------------------
class BillingResponse(BaseModel):
    """The complete structured output from the Billing Agent."""

    response: str = Field(description="Human-readable billing answer")
    action_taken: Optional[str] = Field(
        default=None,
        description="Action simulated, e.g. 'plan_upgrade_simulated'",
    )
    plan_details: Optional[dict[str, Any]] = Field(default=None)
    invoice_summary: Optional[dict[str, Any]] = Field(default=None)
    policy_citations: list[str] = Field(
        default_factory=list,
        description="KB article IDs cited (billing policy references)",
    )
    escalate: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Escalation Agent structured output
# ---------------------------------------------------------------------------
class HumanHandoffPayload(BaseModel):
    """The handoff context package for the human support team."""

    customer_id: Optional[str] = Field(default=None)
    full_history: list[dict[str, Any]] = Field(default_factory=list)
    notes_for_agent: str = Field(description="Contextual notes for the human agent")


class EscalationResponse(BaseModel):
    """The complete structured output from the Escalation Agent."""

    escalation_id: str = Field(description="UUID for this escalation event")
    priority: str = Field(
        description="Priority classification",
        pattern="^(P1|P2|P3|P4)$",
    )
    urgency: str = Field(
        description="Urgency level",
        pattern="^(critical|high|medium|low)$",
    )
    sentiment: str = Field(description="Detected customer sentiment")
    issue_category: str = Field(description="High-level issue category")
    conversation_summary: str = Field(description="2–3 sentence conversation summary")
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = Field(description="Recommended human action")
    human_handoff_payload: HumanHandoffPayload
    trace_id: str = Field(description="Conversation trace ID")
    timestamp: str = Field(description="ISO8601 timestamp of escalation creation")


# ---------------------------------------------------------------------------
# Generic AgentResponse — the envelope returned by every agent to the orchestrator
# ---------------------------------------------------------------------------
class AgentResponse(BaseModel):
    """Unified response envelope wrapping any agent's structured output.

    The orchestrator works with this common wrapper; the actual payload
    is stored in `data` and the agent-specific Pydantic type is preserved
    in `response_type`.
    """

    agent_name: str
    response_type: str = Field(
        description="The response model class name: 'TriageResponse', 'TechnicalResponse', etc."
    )
    data: dict[str, Any] = Field(
        description="The agent's structured output, serialised as a dict"
    )
    trace_id: str
    latency_ms: float = Field(description="Agent processing time in milliseconds")
    error: Optional[str] = Field(
        default=None,
        description="Error message if the agent failed (after retries exhausted)",
    )
