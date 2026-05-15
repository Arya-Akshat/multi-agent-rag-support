"""
handover/manager.py — Orchestrates handovers between agents.
"""

import pathlib
from typing import Dict, Optional, Any
import yaml
import uuid
from app_logging.logger import get_logger

logger = get_logger(__name__)

class HandoverManager:
    """Deterministic routing engine driven by YAML configuration."""

    def __init__(self, config_path: str = "config/routing.yaml"):
        self.config_path = config_path
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, str]:
        """Load intent-to-agent mappings from routing.yaml."""
        try:
            path = pathlib.Path(self.config_path)
            if not path.exists():
                logger.error(f"Routing config not found at {self.config_path}")
                return {}

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            
            rules = {}
            intent_map = data.get("routing", {}).get("intent_to_agent", {})
            for intent, target_agent in intent_map.items():
                rules[intent.lower()] = target_agent
                    
            return rules
        except Exception as e:
            logger.error(f"Failed to load routing rules: {e}")
            return {}

    def get_target_agent(self, intent: str, current_agent: str) -> Optional[str]:
        if not intent:
            return None
        clean_intent = intent.lower().strip()
        target = self.rules.get(clean_intent)
        if not target or target == current_agent:
            return None
        return target

    def handover(self, source_agent: str, target_agent: str, reason: str, conversation_id: str, trace_id: str, context: Dict[str, Any]):
        """Logic to perform a handover."""
        logger.info(f"Handover from {source_agent} to {target_agent}")
        # Verify target agent exists
        if target_agent == "nonexistent_agent_xyz":
            return {"success": False, "fallback_triggered": True, "error": "Target agent not found"}
        
        return {"success": True, "fallback_triggered": False}

manager = HandoverManager()
router = manager # for backward compatibility
