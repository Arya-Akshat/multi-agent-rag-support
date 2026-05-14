"""
models/conversation.py — Core conversation data models.

These are the fundamental building blocks used across every layer:
agents, API, memory, handover. Keep them dependency-free (no local imports).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Citation model — referenced by Message and agent response models
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    """A reference to a knowledge-base article used to ground a response."""

    article_id: str = Field(description="Unique article identifier")
    title: str = Field(description="Human-readable article title")
    snippet: str = Field(default="", description="The text chunk from the article")
    url: str = Field(default="", description="URL to the full article")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Retrieval similarity score (0–1)"
    )


# ---------------------------------------------------------------------------
# Message model — a single turn in the conversation history
# ---------------------------------------------------------------------------
class Message(BaseModel):
    """One message in a conversation thread."""

    role: str = Field(
        description="Speaker role: 'user', 'assistant', or 'system'",
        pattern="^(user|assistant|system)$",
    )
    content: str = Field(description="Message text content")
    agent_name: Optional[str] = Field(
        default=None,
        description="Name of the agent that produced this message (assistant only)",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of message creation",
    )
    citations: Optional[list[Citation]] = Field(
        default=None,
        description="KB citations attached to this message (assistant messages only)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extensible metadata",
    )


# ---------------------------------------------------------------------------
# HandoverEvent model — records a single agent-to-agent transfer
# ---------------------------------------------------------------------------
class HandoverEvent(BaseModel):
    """An immutable record of one agent handover within a conversation."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_agent: str = Field(description="Name of the agent initiating handover")
    target_agent: str = Field(description="Name of the agent receiving control")
    reason: str = Field(description="Why this handover was triggered")
    success: bool = Field(description="Whether the handover completed successfully")
    fallback_triggered: bool = Field(
        default=False,
        description="True if target was unavailable and fallback was used",
    )
    trace_id: str = Field(description="Trace ID of the parent conversation turn")


    @property
    def recent_messages(self) -> list[Message]:
        """Return the last 10 messages for context window purposes."""
        return self.messages[-10:]
