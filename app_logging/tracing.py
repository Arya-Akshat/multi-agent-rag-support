"""
logging/tracing.py — Lightweight per-request trace context management.

Every conversation turn gets a unique trace_id. This module provides
utilities to generate and manage trace IDs so every log line and API
response carries the same ID for end-to-end request tracing.

We intentionally keep this lightweight (no external dependency like
Langfuse/Phoenix) so it works out of the box. The ARCHITECTURE.md
documents how to integrate Langfuse as a future enhancement.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Optional

# ---------------------------------------------------------------------------
# Context variable — stores the active trace_id for the current async context.
# Using ContextVar means it's safe in concurrent asyncio tasks (each task has
# its own copy of the variable).
# ---------------------------------------------------------------------------
_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


def new_trace_id() -> str:
    """Generate a fresh UUID4 trace ID string."""
    return str(uuid.uuid4())


def new_conversation_id() -> str:
    """Generate a fresh UUID4 conversation ID string."""
    return str(uuid.uuid4())


def set_trace_id(trace_id: str) -> None:
    """Store a trace_id in the current async context."""
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """Retrieve the current trace_id, generating one if none is set.

    This is a safe fallback — callers should ideally set a trace_id
    explicitly at request boundaries (API middleware).
    """
    tid = _trace_id_var.get()
    if tid is None:
        tid = new_trace_id()
        _trace_id_var.set(tid)
    return tid


def clear_trace_id() -> None:
    """Reset the trace_id in the current context (used in test teardown)."""
    _trace_id_var.set(None)


class TraceContext:
    """Context manager that sets a trace_id for a block of code.

    Usage:
        with TraceContext(trace_id="abc-123"):
            # all code here runs with trace_id="abc-123"
            do_something()

    If no trace_id is provided, a new UUID is generated automatically.
    """

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id or new_trace_id()
        self._token = None

    def __enter__(self) -> "TraceContext":
        self._token = _trace_id_var.set(self.trace_id)
        return self

    def __exit__(self, *_: object) -> None:
        if self._token is not None:
            _trace_id_var.reset(self._token)
