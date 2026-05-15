"""
logging/logger.py — Centralised structured logging setup using loguru.

Design decisions:
- loguru is used instead of stdlib logging for its simpler API and
  first-class structured JSON serialisation.
- A single `get_logger` function returns a bound logger pre-tagged with
  the caller's module name — callers should NOT create loguru loggers
  directly.
- All logs go to: (1) stdout as JSON for container log aggregation,
  (2) a rotating file at logs/app.log.
- Log level is read from settings at import time; no reconfiguration needed.
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger as _base_logger

from config.settings import settings

# Track whether we've already configured the loguru sinks.
# This prevents double-registration if this module is imported multiple times
# (e.g. in tests).
_configured = False


def _configure_logger() -> None:
    """Configure loguru sinks: stdout (JSON) + rotating file (JSON).

    Called once at module import time. Subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    # Remove the default loguru sink (plain text to stderr)
    _base_logger.remove()

    log_level = settings.log_level.upper()

    # ---- Sink 1: stdout, structured JSON for container log aggregation ----
    _base_logger.add(
        sys.stdout,
        level=log_level,
        format="{message}",       # loguru serialise=True handles the JSON format
        serialize=True,            # output as JSON
        colorize=False,
        backtrace=False,           # keep security-safe: no tracebacks in stdout
        diagnose=False,
    )

    # ---- Sink 2: rotating file at logs/app.log ----
    log_path = settings.log_file_path  # property creates parent dirs
    _base_logger.add(
        str(log_path),
        level=log_level,
        format="{message}",
        serialize=True,
        rotation="10 MB",
        retention=3,               # keep 3 backup files
        compression="gz",
        backtrace=True,            # full tracebacks in file logs for debugging
        diagnose=True,
        enqueue=True,              # non-blocking writes (thread-safe)
    )

    _configured = True


# Run once on import
_configure_logger()


def get_logger(name: str, **context: object):
    """Return a loguru logger bound with a module name and optional context.

    Usage:
        logger = get_logger(__name__, agent="triage")
        logger.info("Processing message", trace_id=..., conversation_id=...)

    Args:
        name:    Typically `__name__` of the calling module.
        **context: Additional key-value pairs bound to every log record from
                   this logger instance (e.g. agent="triage").
    """
    return _base_logger.bind(module=name, **context)


def get_agent_logger(agent_name: str, trace_id: str, conversation_id: str):
    """Convenience factory for agent-specific loggers with required trace fields.

    Every agent log record will automatically carry:
    - module: agent class name
    - agent_name: e.g. "triage"
    - trace_id: current turn trace ID
    - conversation_id: parent conversation ID
    """
    return _base_logger.bind(
        module=f"agents.{agent_name}",
        agent_name=agent_name,
        trace_id=trace_id,
        conversation_id=conversation_id,
    )


def log_agent_invocation(
    agent_name: str,
    trace_id: str,
    conversation_id: str,
    input_summary: str,
    retries_so_far: int = 0,
) -> None:
    """Emit a structured INFO log for an agent invocation."""
    _base_logger.bind(
        event="agent_invocation",
        agent_name=agent_name,
        trace_id=trace_id,
        conversation_id=conversation_id,
        input_summary=input_summary,
        retries_so_far=retries_so_far,
    ).info(f"[{agent_name}] Invocation started")


def log_agent_response(
    agent_name: str,
    trace_id: str,
    conversation_id: str,
    latency_ms: float,
    escalated: bool = False,
) -> None:
    """Emit a structured INFO log for a successful agent response."""
    _base_logger.bind(
        event="agent_response",
        agent_name=agent_name,
        trace_id=trace_id,
        conversation_id=conversation_id,
        latency_ms=round(latency_ms, 2),
        escalated=escalated,
    ).info(f"[{agent_name}] Response produced in {latency_ms:.0f}ms")


def log_handover(
    source_agent: str,
    target_agent: str,
    trace_id: str,
    conversation_id: str,
    reason: str,
    success: bool,
    fallback_triggered: bool = False,
) -> None:
    """Emit a structured INFO/WARNING log for an agent handover event."""
    level = "info" if success else "warning"
    getattr(
        _base_logger.bind(
            event="handover",
            source_agent=source_agent,
            target_agent=target_agent,
            trace_id=trace_id,
            conversation_id=conversation_id,
            reason=reason,
            success=success,
            fallback_triggered=fallback_triggered,
        ),
        level,
    )(f"Handover {source_agent} → {target_agent} ({'ok' if success else 'failed'})")


def log_retrieval(
    trace_id: str,
    conversation_id: str,
    query: str,
    chunks_found: int,
    top_score: Optional[float] = None,
) -> None:
    """Emit a structured INFO/WARNING log for a retrieval query result."""
    level = "warning" if chunks_found == 0 else "info"
    getattr(
        _base_logger.bind(
            event="retrieval",
            trace_id=trace_id,
            conversation_id=conversation_id,
            query=query[:100],      # truncate for log safety
            chunks_found=chunks_found,
            top_score=top_score,
        ),
        level,
    )(f"Retrieval: {chunks_found} chunks found (top_score={top_score})")
