"""
guardrails/pii_scrubber.py — Shim for InputGuard.
"""
from guardrails.input_guard import InputGuard

def scrub_pii(text: str) -> str:
    guard = InputGuard()
    return guard.scrub(text)
