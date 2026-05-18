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
        
        return {"passed": True}
