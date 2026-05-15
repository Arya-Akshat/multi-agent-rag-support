"""
memory/conversation_store.py — Persistence layer for conversation sessions.
"""

from typing import Optional
from models.state import ConversationState

class ConversationStore:
    def __init__(self):
        self._store = {}

    def create(self, conversation_id: str, trace_id: str):
        state = ConversationState(conversation_id=conversation_id, trace_id=trace_id)
        self._store[conversation_id] = state
        return state

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        return self._store.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str):
        state = self.get(conversation_id)
        if state:
            from models.conversation import Message
            state.messages.append(Message(role=role, content=content))
    
    def delete(self, conversation_id: str):
        if conversation_id in self._store:
            del self._store[conversation_id]

    def save_session(self, state: ConversationState):
        self._store[state.conversation_id] = state
