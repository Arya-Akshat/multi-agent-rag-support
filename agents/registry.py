"""
agents/registry.py — Central registry for all support agents.
"""

from typing import Dict, List, Type
from agents.base_agent import BaseAgent
from agents.triage_agent import TriageAgent
from agents.technical_agent import TechnicalAgent
# Import other agents as they are implemented
from agents.billing_agent import BillingAgent
from agents.escalation_agent import EscalationAgent

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, Type[BaseAgent]] = {
            "triage": TriageAgent,
            "technical": TechnicalAgent,
            "billing": BillingAgent,
            "escalation": EscalationAgent
        }

    def get_agent(self, name: str) -> BaseAgent:
        agent_cls = self._agents.get(name.lower())
        if not agent_cls:
            raise ValueError(f"Agent '{name}' not found in registry.")
        return agent_cls()

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

registry = AgentRegistry()
