"""
guardrails/input_guard.py — Protects against prompt injection and PII leaks.
"""

import re
from typing import Dict, Any

class InputGuard:
    def __init__(self):
        self.pii_patterns = {
            "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "phone": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
            "credit_card": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"
        }
        self.injection_keywords = ["ignore previous", "revert all instructions", "system prompt", "DAN"]

    def scrub(self, text: str) -> str:
        for pattern in self.pii_patterns.values():
            text = re.sub(pattern, "[REDACTED]", text)
        return text

    def check(self, text: str) -> Dict[str, Any]:
        risk_score = 0.0
        allowed = True
        risk_category = None

        text_lower = text.lower()
        for kw in self.injection_keywords:
            if kw.lower() in text_lower:
                risk_score = 0.9
                allowed = False
                risk_category = "prompt_injection"
                break
        
        return {
            "allowed": allowed,
            "risk_score": risk_score,
            "risk_category": risk_category
        }
