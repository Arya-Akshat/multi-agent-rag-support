"""
handover/models.py — Data models for agent handovers.
"""

from datetime import datetime, timezone
from pydantic import BaseModel, Field

class HandoverEvent(BaseModel):
    """An immutable record of one agent handover within a conversation."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_agent: str = Field(description="Name of the agent initiating handover")
    target_agent: str = Field(description="Name of the agent receiving control")
    reason: str = Field(description="Why this handover was triggered")
    success: bool = Field(default=True)
    conversation_summary: str = Field(default="", description="Summary of conversation up to this point")
    extracted_entities: dict = Field(default_factory=dict)
    trace_id: str = Field(description="Trace ID of the parent conversation turn")
