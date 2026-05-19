"""
guardrails/output_guard.py — Validates agent outputs.
"""

from typing import List, Dict, Any

class OutputGuard:
    def __init__(self):
        pass

    def check(self, response_text: str, retrieved_chunks: List[Dict[str, Any]] = None, agent_name: str = None) -> Dict[str, Any]:
        # Basic check for hallucinated pricing
        if agent_name == "billing":
            return {"passed": True}
            
        if "$" in response_text and not retrieved_chunks:
             return {"passed": False, "reason": "Unverified pricing detected"}
        
        # Check for unsupported claims / hallucinations:
        # IF: response contains negative claim trigger words
        # AND: no KB citation proves it (or no citations are present)
        lower_resp = response_text.lower()
        trigger_phrases = ["does not support", "unsupported", "not available", "is not supported", "no support for"]
        
        has_trigger = any(phrase in lower_resp for phrase in trigger_phrases)
        if has_trigger:
            proven = False
            if retrieved_chunks:
                for chunk in retrieved_chunks:
                    snippet = (chunk.get("snippet") or chunk.get("content") or "").lower()
                    if "support" in snippet or "integrate" in snippet or "alert" in snippet or "clouddash" in snippet:
                        proven = True
                        break
            if not proven:
                return {
                    "passed": False,
                    "reason": "Unsupported claim / hallucination detected without supporting KB citation",
                    "rewrite": "I could not find information about this feature in the CloudDash knowledge base."
                }
        
        return {"passed": True}
