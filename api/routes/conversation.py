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
    
    # Find all new assistant messages added during this turn
    user_msg_idx = -1
    for idx, msg in enumerate(updated_session.messages):
        if msg.role == "user":
            user_msg_idx = idx
            
    new_assistant_msgs = updated_session.messages[user_msg_idx + 1:]
    
    # 4. Run Output Guardrails (verify pricing or hallucinations against context)
    from guardrails.output_guard import OutputGuard
    output_guard = OutputGuard()
    
    response_parts = []
    all_citations = []
    for msg in new_assistant_msgs:
        # Run output guardrail per message content based on the issuing agent
        chunks = []
        if msg.citations:
            for c in msg.citations:
                chunks.append({"title": c.title, "snippet": c.snippet})
                
        guard_result = output_guard.check(msg.content, retrieved_chunks=chunks, agent_name=msg.agent_name)
        msg_content = msg.content
        if not guard_result["passed"]:
            logger.warning(f"Output guardrail failed for agent {msg.agent_name} in session {request.conversation_id}: {guard_result.get('reason')}")
            msg_content = "I apologize, but I am unable to verify the pricing or policy details for that request. I can escalate this issue to a billing representative for accurate details."
            msg.content = msg_content
            session_store.save_session(updated_session)
            
        response_parts.append(msg_content)
        if msg.citations:
            all_citations.extend(msg.citations)
            
    response_content = "\n\n[Automatic Handover]\n\n".join(response_parts) if response_parts else "No response"
            
    # 5. Extract Citations (deduplicated by article_id)
    citations_data = []
    seen_articles = set()
    for c in all_citations:
        art_id = c.article_id or c.title
        if art_id not in seen_articles:
            seen_articles.add(art_id)
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

@router.get("/{conversation_id}", response_model=Dict[str, Any])
def get_conversation(conversation_id: str):
    session = session_store.get(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return session.model_dump()
