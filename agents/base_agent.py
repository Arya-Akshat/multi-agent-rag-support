"""
agents/base.py — Base Agent class.

Provides common LLM binding, prompt loading, and structured output
parsing for all specialized agents.
"""

import pathlib
import time
from typing import Any, Dict, Optional, Type

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app_logging.logger import get_logger
from config.settings import settings
from models.state import ConversationState

logger = get_logger(__name__)


class BaseAgent:
    """Base class for all LangGraph agents."""

    def __init__(self, agent_name: str):
        self.name = agent_name
        self.config = self._load_agent_config()
        self.system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model_name=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.0,  # Zero temp for strict adherence to KB and routing
        )

    def _load_agent_config(self, config_path: str = "config/agents.yaml") -> Dict[str, Any]:
        """Load this agent's specific prompt and rules from YAML."""
        try:
            path = pathlib.Path(config_path)
            if not path.exists():
                logger.error(f"Agents config not found at {config_path}")
                return {}

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            agents_list = data.get("agents", [])
            
            # Find the agent config by name
            agent_config = next((a for a in agents_list if a.get("name") == self.name), None)
            
            if not agent_config:
                logger.warning(f"No configuration found for agent '{self.name}'")
                return {}
                
            return agent_config
        except Exception as e:
            logger.error(f"Failed to load config for agent {self.name}: {e}")
            return {}

    def format_history(self, state: ConversationState) -> str:
        """Format the conversation history for the prompt context."""
        history_str = ""
        # Only use the most recent messages to fit context window
        for msg in state.recent_messages:
            role = msg.role.capitalize()
            # If an assistant message was from a specific agent, note it
            if role == "Assistant" and msg.agent_name:
                role = f"Assistant ({msg.agent_name.capitalize()})"
            history_str += f"{role}: {msg.content}\n"
            
        return history_str

    def invoke_structured(
        self, 
        prompt_variables: Dict[str, Any], 
        response_model: Type[BaseModel]
    ) -> BaseModel:
        """
        Invoke the LLM and force it to return data matching the Pydantic model.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        
        parser = PydanticOutputParser(pydantic_object=response_model)
        
        # Inject format instructions into the system prompt
        system_with_format = self.system_prompt + "\n\n{format_instructions}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_with_format),
            ("user", "{user_input}")
        ])
        
        # Build the chain and invoke
        chain = prompt | self.llm | parser
        
        # Add format instructions to variables
        prompt_variables["format_instructions"] = parser.get_format_instructions()
        
        logger.debug(f"Agent '{self.name}' invoking LLM...")
        max_retries = self.config.get("max_retries", 3)
        
        for attempt in range(max_retries):
            try:
                response = chain.invoke(prompt_variables)
                return response
            except Exception as e:
                logger.warning(f"Agent '{self.name}' structured invocation failed on attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    logger.error("Max retries exceeded.")
                    raise
                sleep_time = 2 * (attempt + 1)
                logger.info(f"Rate limit / API error. Sleeping for {sleep_time}s before retry...")
                time.sleep(sleep_time)
