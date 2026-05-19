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

import concurrent.futures
workflow_executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

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
    messages: List[dict] = []

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
    
    # Every new turn must start at triage to classify the new user message
    session.current_agent = "triage"
    session.previous_agent = ""
    
    # Track existing handovers to return only new ones
    existing_handovers_count = len(session.handover_history)
    
    # 3. Invoke LangGraph Orchestrator
    state_dict = session.model_dump()
    
    import traceback
    import time
    from datetime import datetime
    
    MAX_WORKFLOW_SECONDS = 55
    
    logger.info(f"[TRACE] workflow_start={datetime.now().isoformat()}")
    logger.info(f"[TRACE] {session.current_agent}_start={datetime.now().isoformat()}")
    
    print("[TRACE] graph.invoke START")
    logger.info("[TRACE] graph.invoke START")
    
    try:
        # Submit task to the global workflow_executor instead of using a 'with' context manager
        # that implicitly waits/joins on timed-out worker threads.
        future = workflow_executor.submit(graph.invoke, state_dict)
        try:
            new_state_dict = future.result(timeout=MAX_WORKFLOW_SECONDS)
            print("[TRACE] graph.invoke END")
            logger.info("[TRACE] graph.invoke END")
        except concurrent.futures.TimeoutError:
            print("[TIMEOUT] workflow exceeded 30s")
            logger.error(f"[TIMEOUT] workflow exceeded {MAX_WORKFLOW_SECONDS} seconds!")
            fallback_msg = Message(
                role="assistant",
                content="CloudDash support is temporarily experiencing high latency. Please retry your request.",
                agent_name=session.current_agent
            )
            session.messages.append(fallback_msg)
            session_store.save_session(session)
            new_state_dict = session.model_dump()
    except Exception as e:
        logger.error(f"[CRITICAL] Runtime error in graph.invoke() on agent '{session.current_agent}': {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        fallback_msg = Message(
            role="assistant",
            content="CloudDash support is temporarily experiencing high latency. Please retry your request.",
            agent_name=session.current_agent
        )
        session.messages.append(fallback_msg)
        session_store.save_session(session)
        new_state_dict = session.model_dump()
        
    logger.info(f"[TRACE] workflow_end={datetime.now().isoformat()}")
    
    updated_session = ConversationState(**new_state_dict)
    
    # Run Centralized Execution Validator before final API response
    from guardrails.validators import validate_workflow_execution, validate_agent_domain_response, validate_grounding
    
    logger.info("[VALIDATOR] starting workflow validation")
    try:
        validate_workflow_execution(updated_session)
        logger.info("[VALIDATOR] workflow validation complete")
    except Exception as val_err:
        logger.error(f"[VALIDATOR ERROR] validate_workflow_execution failed: {val_err}")
    
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
    messages_data = []
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
            if "rewrite" in guard_result:
                msg_content = guard_result["rewrite"]
            else:
                msg_content = "I apologize, but I am unable to verify the pricing or policy details for that request. I can escalate this issue to a billing representative for accurate details."
        
        # Double-assurance centralized validation pass
        try:
            msg_content = validate_agent_domain_response(msg.agent_name, msg_content)
        except Exception as domain_err:
            logger.error(f"[VALIDATOR ERROR] validate_agent_domain_response failed: {domain_err}")
            
        try:
            msg_content = validate_grounding(msg_content, chunks, updated_session.messages[user_msg_idx].content)
        except Exception as ground_err:
            logger.error(f"[VALIDATOR ERROR] validate_grounding failed: {ground_err}")
            
        msg.content = msg_content
        session_store.save_session(updated_session)
            
        response_parts.append(msg_content)
        
        # Add citations for this specific message block
        msg_citations = []
        if msg.citations:
            all_citations.extend(msg.citations)
            for c in msg.citations:
                msg_citations.append({
                    "article_id": c.article_id,
                    "title": c.title,
                    "snippet": c.snippet,
                    "url": c.url,
                    "relevance_score": c.relevance_score
                })
                
        messages_data.append({
            "role": "assistant",
            "content": msg_content,
            "agent": msg.agent_name or "unknown",
            "citations": msg_citations
        })
            
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
        citations=citations_data,
        messages=messages_data
    )

@router.get("/{conversation_id}", response_model=Dict[str, Any])
def get_conversation(conversation_id: str):
    session = session_store.get(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return session.model_dump()
