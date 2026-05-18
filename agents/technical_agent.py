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

        # 1. RAG Pipeline: Rewrite query and retrieve context
        rewritten_query = self.query_rewriter.rewrite(state.recent_messages)
        citations = []
        kb_context = "No relevant knowledge base articles found."
        
        if rewritten_query:
            citations = self.retriever.retrieve(rewritten_query)
            if citations:
                kb_context = "\n\n".join([f"Source: {c.title}\n{c.snippet}" for c in citations])

        # 2. Format input for the prompt
        history = self.format_history(state)
        # Pass extracted entities so the agent knows what we already know
        entities_str = ", ".join([f"{k}: {v}" for k, v in state.extracted_entities.items()]) or "None"
        
        user_input = (
            f"Conversation History:\n{history}\n\n"
            f"Extracted Entities context: {entities_str}\n\n"
            f"Knowledge Base Context:\n{kb_context}\n\n"
            f"Last User Message: {last_msg}"
        )
        
        # 3. Invoke LLM
        response: TechnicalResponse = self.invoke_structured(
            prompt_variables={"user_input": user_input},
            response_model=TechnicalResponse
        )

        updates = {}

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
                updates["handover_history"] = [handover_event]
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
