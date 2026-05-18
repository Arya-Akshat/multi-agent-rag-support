"""
models/state.py — Core conversation data models and state management.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict
from pydantic import BaseModel, Field
from models.conversation import Message, HandoverEvent

class TriageIntent(BaseModel):
    type: str = Field(description="Intent type, e.g. 'technical_support', 'billing_upgrade'")
    priority: int = Field(default=1)
    status: str = Field(default="pending")

class ConversationState(BaseModel):
    """The complete, mutable state object for a single conversation session."""
    conversation_id: str = Field(description="Unique conversation identifier (UUID)")
    trace_id: str = Field(description="Current turn trace ID (UUID, rotates per turn)")
    messages: list[Message] = Field(default_factory=list)
    current_agent: str = Field(default="triage")
    previous_agent: Optional[str] = Field(default=None)
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    handover_history: list[HandoverEvent] = Field(default_factory=list)
    escalated: bool = Field(default=False)
    intents: list[TriageIntent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def last_user_message(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    @property
    def recent_messages(self) -> list[Message]:
        return self.messages[-10:]

class GraphState(TypedDict, total=False):
    conversation_id: str
    trace_id: str
    messages: list[Message]
    current_agent: str
    previous_agent: Optional[str]
    extracted_entities: dict[str, Any]
    handover_history: list[HandoverEvent]
    escalated: bool
    intents: list[TriageIntent]
