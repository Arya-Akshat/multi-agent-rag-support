"""
models/events.py — System event models for logging, audit, and observability.

These models define the structured payload for every loggable event in the
system. Using typed models ensures logs are consistent and parseable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Enumeration of all trackable event types in the system."""

    # Agent lifecycle
    AGENT_INVOCATION = "agent_invocation"
    AGENT_RESPONSE = "agent_response"
    AGENT_FAILURE = "agent_failure"
    AGENT_RETRY = "agent_retry"

    # Routing
    ROUTING_DECISION = "routing_decision"

    # Retrieval
    RETRIEVAL_QUERY = "retrieval_query"
    RETRIEVAL_RESULT = "retrieval_result"
    RETRIEVAL_EMPTY = "retrieval_empty"

    # Handover
    HANDOVER_INITIATED = "handover_initiated"
    HANDOVER_SUCCESS = "handover_success"
    HANDOVER_FAILURE = "handover_failure"
    HANDOVER_FALLBACK = "handover_fallback"

    # Escalation
    ESCALATION_TRIGGERED = "escalation_triggered"

    # Guardrails
    GUARDRAIL_INPUT_BLOCKED = "guardrail_input_blocked"
    GUARDRAIL_OUTPUT_MODIFIED = "guardrail_output_modified"

    # API
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    API_ERROR = "api_error"

    # Rate limiting
    RATE_LIMIT_HIT = "rate_limit_hit"


class BaseEvent(BaseModel):
    """Common fields present in every event log entry."""

    event_type: EventType
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    trace_id: str = Field(description="Trace ID of the conversation turn")
    conversation_id: str = Field(description="Parent conversation identifier")


class AgentInvocationEvent(BaseEvent):
    """Logged when an agent begins processing a message."""

    event_type: EventType = EventType.AGENT_INVOCATION
    agent_name: str
    input_summary: str = Field(description="Short summary of the input (not full content)")
    retries_so_far: int = Field(default=0)


class AgentResponseEvent(BaseEvent):
    """Logged when an agent produces a successful response."""

    event_type: EventType = EventType.AGENT_RESPONSE
    agent_name: str
    latency_ms: float
    response_type: str = Field(description="Pydantic model class name of the response")
    escalated: bool = Field(default=False)


class AgentFailureEvent(BaseEvent):
    """Logged when an agent exhausts retries and returns an error."""

    event_type: EventType = EventType.AGENT_FAILURE
    agent_name: str
    error_message: str
    retries_attempted: int


class RoutingDecisionEvent(BaseEvent):
    """Logged when the triage agent produces a routing decision."""

    event_type: EventType = EventType.ROUTING_DECISION
    intents: list[str]
    primary_agent: str
    secondary_agents: list[str] = Field(default_factory=list)
    requires_multi_step: bool = Field(default=False)
    urgency: str
    sentiment: str


class RetrievalQueryEvent(BaseEvent):
    """Logged when a retrieval query is issued to the vector store."""

    event_type: EventType = EventType.RETRIEVAL_QUERY
    original_query: str
    rewritten_query: str
    top_k: int


class RetrievalResultEvent(BaseEvent):
    """Logged when retrieval returns results."""

    event_type: EventType = EventType.RETRIEVAL_RESULT
    chunks_returned: int
    top_score: Optional[float] = None
    article_ids: list[str] = Field(default_factory=list)


class HandoverEvent(BaseEvent):
    """Logged for every handover attempt (success or failure)."""

    event_type: EventType = EventType.HANDOVER_INITIATED
    source_agent: str
    target_agent: str
    reason: str
    success: bool
    fallback_triggered: bool = Field(default=False)
    context_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Lightweight snapshot: message_count, entities, last_user_message",
    )


class GuardrailInputEvent(BaseEvent):
    """Logged when an input guardrail blocks or flags a message."""

    event_type: EventType = EventType.GUARDRAIL_INPUT_BLOCKED
    risk_category: str
    risk_score: float
    reason: str


class GuardrailOutputEvent(BaseEvent):
    """Logged when an output guardrail modifies an agent response."""

    event_type: EventType = EventType.GUARDRAIL_OUTPUT_MODIFIED
    agent_name: str
    modification_reason: str


class APIRequestEvent(BaseEvent):
    """Logged for every inbound API request."""

    event_type: EventType = EventType.API_REQUEST
    method: str
    path: str
    client_ip: str


class APIErrorEvent(BaseEvent):
    """Logged for API errors returned to the client."""

    event_type: EventType = EventType.API_ERROR
    status_code: int
    error_code: str
    error_message: str
