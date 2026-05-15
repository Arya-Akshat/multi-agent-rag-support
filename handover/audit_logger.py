"""
handover/audit_logger.py — Logs handover events for auditing.
"""

from app_logging.logger import get_logger

logger = get_logger(__name__)

class AuditLogger:
    def __init__(self):
        pass

    def log_handover(self, event_data: dict):
        logger.info(f"AUDIT: Handover event: {event_data}")
