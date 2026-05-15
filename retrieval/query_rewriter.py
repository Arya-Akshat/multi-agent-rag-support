"""
retrieval/query_rewriter.py — Conversation-aware query rewriting.

Uses an LLM to rewrite a multi-turn conversation into a single standalone
query optimized for retrieval. Falls back to returning the last message
if the LLM call fails or no API key is provided.
"""

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app_logging.logger import get_logger
from config.settings import settings
from models.conversation import Message

logger = get_logger(__name__)


# System prompt for the rewriter
REWRITER_PROMPT = """You are a search query formulation expert.
Given a conversation history, rewrite the final user message into a single, standalone search query.
The rewritten query must contain all relevant context from previous turns to be useful for a vector search.
Strip conversational filler words (e.g., "hi", "can you help me with", "thanks").
Preserve domain-specific terms, acronyms, and error messages exactly.

If the last user message is already a standalone query and needs no context, just return it mostly unchanged (but strip filler).

Return ONLY the rewritten query string. Do not include quotes, explanations, or Markdown formatting.
"""

class QueryRewriter:
    def __init__(self):
        # We initialise the LLM lazily or handle missing keys gracefully.
        self.llm = None
        if settings.groq_api_key:
            try:
                self.llm = ChatGroq(
                    model_name=settings.groq_model,
                    groq_api_key=settings.groq_api_key,
                    temperature=0.0, # Zero temp for deterministic, focused rewrites
                    max_tokens=100
                )
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", REWRITER_PROMPT),
                    ("user", "Conversation History:\n{history}\n\nRewrite the last user message:")
                ])
            except Exception as e:
                logger.warning(f"Failed to initialise QueryRewriter LLM: {e}")

    def rewrite(self, recent_messages: List[Message]) -> str:
        """
        Rewrite the recent conversation history into a standalone query.
        
        Args:
            recent_messages: The last N messages from the conversation.
            
        Returns:
            The rewritten standalone query string.
        """
        if not recent_messages:
            return ""

        # Extract the very last message. If it's not a user message, we can't rewrite it well.
        last_msg = recent_messages[-1]
        if last_msg.role != "user":
            logger.debug("Last message is not from user. Returning empty string for retrieval.")
            return ""

        # If LLM is not configured, gracefully degrade to just returning the last message
        if not self.llm:
            logger.debug("QueryRewriter LLM unavailable. Falling back to raw last message.")
            return last_msg.content.strip()

        # Format history into a readable string
        history_str = ""
        for msg in recent_messages:
            role = msg.role.capitalize()
            history_str += f"{role}: {msg.content}\n"

        try:
            # Execute the LangChain runnable
            chain = self.prompt | self.llm
            response = chain.invoke({"history": history_str})
            rewritten = response.content.strip()
            
            logger.debug(f"Query rewritten: '{last_msg.content}' -> '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}. Falling back to raw message.")
            return last_msg.content.strip()
