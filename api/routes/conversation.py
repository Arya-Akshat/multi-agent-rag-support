"""
api/routes/conversation.py — Conversation endpoints.
"""
from fastapi import APIRouter, HTTPException, Request
import uuid
from typing import Dict, Any, List
from pydantic import BaseModel

from agents.orchestrator import graph
from models.state import ConversationState, Message
from memory.session_store import session_store
from guardrails.pii_scrubber import scrub_pii
from app_logging.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/conversation")

class StartResponse(BaseModel):
    conversation_id: str
    message: str

class MessageRequest(BaseModel):
    conversation_id: str
    message: str

class MessageResponse(BaseModel):
    response: str
    agent: str
    escalated: bool
    handovers: List[dict] = []
    citations: List[dict] = []

@router.post("/start", response_model=StartResponse)
def start_conversation():
    conversation_id = str(uuid.uuid4())
    trace_id = str(uuid.uuid4())
    initial_state = ConversationState(
        conversation_id=conversation_id,
        trace_id=trace_id,
        messages=[],
        current_agent="triage"
    )
    session_store.save_session(initial_state)
    return StartResponse(
        conversation_id=conversation_id,
        message="Conversation started. How can CloudDash help you today?"
    )

@router.post("/message", response_model=MessageResponse)
def send_message(request: MessageRequest):
    session = session_store.get(request.conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
        
    scrubbed_message = scrub_pii(request.message)
    user_msg = Message(role="user", content=scrubbed_message)
    session.messages.append(user_msg)
    
    state_dict = session.model_dump()
    new_state_dict = graph.invoke(state_dict)
    
    updated_session = ConversationState(**new_state_dict)
    session_store.save_session(updated_session)
    
    last_message = updated_session.messages[-1] if updated_session.messages else None
    response_content = last_message.content if last_message else "No response"
    
    return MessageResponse(
        response=response_content,
        agent=updated_session.current_agent,
        escalated=updated_session.escalated
    )
