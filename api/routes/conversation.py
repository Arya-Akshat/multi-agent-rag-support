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
        
    # 1. Run Input Guardrails (check for prompt injection)
    from guardrails.input_guard import InputGuard
    input_guard = InputGuard()
    input_check = input_guard.check(request.message)
    if not input_check["allowed"]:
        logger.warning(f"Input blocked for session {request.conversation_id}: {input_check['risk_category']}")
        raise HTTPException(
            status_code=400,
            detail=f"Security Alert: Message blocked. Reason: {input_check['risk_category']}"
        )
        
    # 2. Run PII Scrubbing
    scrubbed_message = scrub_pii(request.message)
    user_msg = Message(role="user", content=scrubbed_message)
    session.messages.append(user_msg)
    
    # Track existing handovers to return only new ones
    existing_handovers_count = len(session.handover_history)
    
    # 3. Invoke LangGraph Orchestrator
    state_dict = session.model_dump()
    new_state_dict = graph.invoke(state_dict)
    
    updated_session = ConversationState(**new_state_dict)
    session_store.save_session(updated_session)
    
    last_message = updated_session.messages[-1] if updated_session.messages else None
    response_content = last_message.content if last_message else "No response"
    
    # 4. Run Output Guardrails (verify pricing or hallucinations against context)
    from guardrails.output_guard import OutputGuard
    output_guard = OutputGuard()
    chunks = []
    if last_message and last_message.citations:
        for c in last_message.citations:
            chunks.append({"title": c.title, "snippet": c.snippet})
            
    guard_result = output_guard.check(response_content, retrieved_chunks=chunks)
    if not guard_result["passed"]:
        logger.warning(f"Output guardrail failed for session {request.conversation_id}: {guard_result.get('reason')}")
        response_content = "I apologize, but I am unable to verify the pricing or policy details for that request. I can escalate this issue to a billing representative for accurate details."
        if last_message:
            last_message.content = response_content
            session_store.save_session(updated_session)
            
    # 5. Extract Citations
    citations_data = []
    if last_message and last_message.citations:
        for c in last_message.citations:
            citations_data.append({
                "article_id": c.article_id,
                "title": c.title,
                "snippet": c.snippet,
                "url": c.url,
                "relevance_score": c.relevance_score
            })
            
    # 6. Extract New Handovers
    handovers_data = []
    new_handovers = updated_session.handover_history[existing_handovers_count:]
    for h in new_handovers:
        handovers_data.append({
            "source": h.source_agent,
            "target": h.target_agent,
            "reason": h.reason
        })
        
    return MessageResponse(
        response=response_content,
        agent=updated_session.current_agent,
        escalated=updated_session.escalated,
        handovers=handovers_data,
        citations=citations_data
    )
