"""
memory/session_store.py — Shim for ConversationStore.
"""
from memory.conversation_store import ConversationStore

class SessionStore(ConversationStore):
    pass

session_store = SessionStore()
