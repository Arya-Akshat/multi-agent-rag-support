"""
agents/technical_agent.py — The Technical Specialist.

Handles AWS, Azure, GCP, and metrics/alerts troubleshooting.
Relies heavily on RAG retrieval. Can escalate if a solution cannot be found.
"""

from agents.base_agent import BaseAgent
from handover.router import router
from models.state import ConversationState, Message, HandoverEvent
from models.responses import TechnicalResponse
from retrieval import Retriever, QueryRewriter
from app_logging.logger import get_logger

logger = get_logger(__name__)


class TechnicalAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_name="technical")
        self.retriever = Retriever()
        self.query_rewriter = QueryRewriter()

    def __call__(self, state: ConversationState) -> dict:
        logger.info(f"Technical agent processing conversation {state.conversation_id}")
        
        last_msg = state.last_user_message
        if not last_msg:
            return {"current_agent": "technical"}

        # Extract active technical support intent
        from models.state import TriageIntent
        active_intent = None
        for i in state.intents:
            status = getattr(i, "status", None) or (i.get("status") if isinstance(i, dict) else "")
            itype = getattr(i, "type", "") or (i.get("type", "") if isinstance(i, dict) else "")
            if "technical" in itype and status == "pending":
                active_intent = i
                break
        
        if not active_intent:
            active_intent = TriageIntent(
                type="technical",
                priority=1,
                status="pending",
                active_intent_query=last_msg
            )

        # Scoped active intent query
        active_query = getattr(active_intent, "active_intent_query", None) or last_msg

        # 1. RAG Pipeline: Rewrite ONLY the active intent query
        from models.conversation import Message
        scoped_recent_messages = []
        user_replaced = False
        for msg in reversed(state.recent_messages):
            if msg.role == "user" and not user_replaced:
                scoped_recent_messages.append(
                    Message(
                        role="user",
                        content=active_query,
                        agent_name=msg.agent_name,
                        timestamp=msg.timestamp
                    )
                )
                user_replaced = True
            else:
                scoped_recent_messages.append(msg)
        scoped_recent_messages.reverse()
        
        if not user_replaced:
            scoped_recent_messages.append(Message(role="user", content=active_query))

        rewritten_query = self.query_rewriter.rewrite(scoped_recent_messages)
        citations = []
        kb_context = "No relevant knowledge base articles found."
        
        if rewritten_query:
            citations = self.retriever.retrieve(rewritten_query)
            if citations:
                kb_context = "\n\n".join([f"Source: {c.title}\n{c.snippet}" for c in citations])

        # 2. Format input for the prompt using build_agent_task_context helper
        from agents.context_helper import build_agent_task_context
        history = self.format_history(state)
        task_context = build_agent_task_context(state, active_intent)
        
        user_input = (
            f"Conversation History:\n{history}\n\n"
            f"Knowledge Base Context:\n{kb_context}\n\n"
            f"{task_context}"
        )
        
        # 3. Invoke LLM
        response: TechnicalResponse = self.invoke_structured(
            prompt_variables={"user_input": user_input},
            response_model=TechnicalResponse
        )

        updates = {}

        # Track incoming background handover
        if state.current_agent != self.name:
            handover_event = HandoverEvent(
                source_agent=state.current_agent,
                target_agent=self.name,
                reason="Automatic background handover for pending technical support intent",
                success=True,
                trace_id=state.trace_id
            )
            updates["current_agent"] = self.name
            updates["previous_agent"] = state.current_agent
            updates["handover_history"] = state.handover_history + [handover_event]

        # Mark active technical support intent as completed
        new_intents = [i.model_copy() if hasattr(i, "model_copy") else i for i in state.intents]
        for idx, i in enumerate(new_intents):
            status = getattr(i, "status", None) or (i.get("status") if isinstance(i, dict) else "")
            itype = getattr(i, "type", "") or (i.get("type", "") if isinstance(i, dict) else "")
            if "technical" in itype and status == "pending":
                if hasattr(i, "status"):
                    i.status = "completed"
                elif isinstance(i, dict):
                    new_intents[idx]["status"] = "completed"
        updates["intents"] = new_intents

        # 4. Handle Escalation
        if response.escalate:
            # Escalation suppression: if there is a pending billing upgrade, route to billing first
            # unless there is an explicit request for human or manager.
            has_pending_billing = any("billing" in (getattr(x, "type", "") or x.get("type", "")) and (getattr(x, "status", "") or x.get("status", "")) == "pending" for x in new_intents)
            explicit_escalation = "manager" in last_msg.lower() or "human" in last_msg.lower() or "escalate" in last_msg.lower()
            
            if not has_pending_billing or explicit_escalation:
                logger.info("Technical agent escalating to human.")
                target_agent = "escalation"
                handover_event = HandoverEvent(
                    source_agent=self.name,
                    target_agent=target_agent,
                    reason=response.escalation_reason or "Escalated by Technical Agent",
                    success=True,
                    trace_id=state.trace_id
                )
                updates["current_agent"] = target_agent
                updates["handover_history"] = updates.get("handover_history", state.handover_history) + [handover_event]
                return updates

        # 5. Handle normal response
        logger.info("Technical responding to user.")
        
        # Check if we should handover to Billing
        has_pending_billing = any("billing" in (getattr(x, "type", "") or (x.get("type", "") if isinstance(x, dict) else "")) and (getattr(x, "status", "") or (x.get("status", "") if isinstance(x, dict) else "")) == "pending" for x in new_intents)
        handovers = []
        if has_pending_billing:
            handover_event = HandoverEvent(
                source_agent=self.name,
                target_agent="billing",
                reason="Automatic handover to Billing Agent for plan upgrade.",
                success=True,
                trace_id=state.trace_id
            )
            handovers.append(handover_event)
        
        used_citations = []
        if response.citations:
            for tech_cit in response.citations:
                for c in citations:
                    if tech_cit.article_id == c.article_id:
                        if c not in used_citations:
                            used_citations.append(c)
        
        if not used_citations and citations and response.confidence > 0.5:
            used_citations = citations

        msg = Message(
            role="assistant",
            content=response.response,
            agent_name=self.name,
            citations=used_citations if used_citations else None,
            metadata={"suggested_next_steps": response.suggested_next_steps}
        )
        
        updates["messages"] = [msg]
        if handovers:
            updates["handover_history"] = handovers

        return updates
