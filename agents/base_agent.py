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
        
        # Initialize Groq LLM with a hard timeout of 15 seconds to prevent network hangs
        self.llm = ChatGroq(
            model_name=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=0.0,  # Zero temp for strict adherence to KB and routing
            timeout=15.0,     # Enforce strict 15s timeout
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
        if self.name in ["technical", "billing"]:
            from agents.context_helper import get_isolated_history
            return get_isolated_history(state, self.name)

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
        import json
        import traceback
        from datetime import datetime
        from langchain_core.output_parsers import PydanticOutputParser
        
        parser = PydanticOutputParser(pydantic_object=response_model)
        
        # Inject format instructions into the system prompt
        system_with_format = self.system_prompt + "\n\n{format_instructions}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_with_format),
            ("user", "{user_input}")
        ])
        
        # Build the chain
        chain = prompt | self.llm
        
        # Add format instructions to variables
        prompt_variables["format_instructions"] = parser.get_format_instructions()
        
        # STEP 1: Instrument execution start timestamp
        llm_start_time = datetime.now().isoformat()
        print(f"[TRACE] {self.name}_llm_invoke_start={llm_start_time}")
        logger.info(f"[TRACE] {self.name}_llm_invoke_start={llm_start_time}")
        
        max_retries = 2  # Hard limit max retries to 2 as requested in STEP 4
        
        for attempt in range(max_retries):
            # Print retry count to stdout and log
            print(f"[LLM] agent='{self.name}' attempt={attempt+1}/{max_retries} model='{settings.groq_model}'")
            logger.info(f"[LLM] agent='{self.name}' attempt={attempt+1}/{max_retries} model='{settings.groq_model}'")
            
            try:
                # Track raw API response time (STEP 3)
                api_call_start = time.time()
                api_call_start_str = datetime.now().isoformat()
                print(f"[TRACE] {self.name}_api_call_start={api_call_start_str}")
                logger.info(f"API call attempt {attempt+1}/{max_retries} for agent '{self.name}' using model '{settings.groq_model}'")
                
                msg = chain.invoke(prompt_variables)
                
                api_call_end = time.time()
                api_call_end_str = datetime.now().isoformat()
                print(f"[TRACE] {self.name}_api_call_end={api_call_end_str}")
                
                elapsed = api_call_end - api_call_start
                logger.info(f"Raw API response received in {elapsed:.3f}s. Response length: {len(msg.content)} characters. (attempt {attempt+1}/{max_retries})")
                
                content = msg.content
                
                # Robust extraction of JSON from conversational text
                print(f"[PARSING] {self.name} parsing START")
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = content[first_brace:last_brace+1]
                else:
                    json_str = content
                
                try:
                    parsed_dict = json.loads(json_str)
                    response = response_model.model_validate(parsed_dict)
                    
                    llm_end_time = datetime.now().isoformat()
                    print(f"[TRACE] {self.name}_llm_invoke_end={llm_end_time}")
                    logger.info(f"[TRACE] {self.name}_llm_invoke_end={llm_end_time}")
                    print(f"[PARSING] {self.name} parsing END (success)")
                    return response
                except Exception as parse_err:
                    logger.warning(f"Custom JSON parsing failed: {parse_err}. Falling back to standard parser. (attempt {attempt+1}/{max_retries})")
                    response = parser.parse(content)
                    
                    llm_end_time = datetime.now().isoformat()
                    print(f"[TRACE] {self.name}_llm_invoke_end={llm_end_time}")
                    logger.info(f"[TRACE] {self.name}_llm_invoke_end={llm_end_time}")
                    print(f"[PARSING] {self.name} parsing END (fallback parser success)")
                    return response
                    
            except Exception as e:
                logger.warning(f"Agent '{self.name}' structured invocation failed on attempt {attempt+1}/{max_retries}: {e}")
                logger.warning(f"Raw content was:\n{getattr(e, 'llm_output', 'No llm_output')} | Stacktrace:\n{traceback.format_exc()}")
                if attempt < max_retries - 1:
                    sleep_time = 2 * (attempt + 1)
                    logger.info(f"Rate limit / API error. Sleeping for {sleep_time}s before retry...")
                    time.sleep(sleep_time)
        
        # Max retries exceeded: log raw response, fail gracefully, return fallback structured response
        logger.error(f"[CRITICAL] Max retries ({max_retries}) exceeded for agent '{self.name}' on structured output.")
        
        # Build fallback structured response
        logger.info(f"Generating fallback structured response for {response_model.__name__}")
        if response_model.__name__ == "TriageResponse":
            fallback = response_model(
                intents=[],
                entities={
                    "customer_id": None,
                    "cloud_provider": None,
                    "plan_type": None,
                    "issue_type": None,
                    "urgency": "low",
                    "sentiment": "neutral"
                },
                routing_decision={
                    "primary_agent": "triage",
                    "secondary_agents": [],
                    "reason": "Fallback due to LLM parsing error."
                },
                requires_multi_step=False
            )
        elif response_model.__name__ == "TechnicalResponse":
            fallback = response_model(
                response="I apologize, but I encountered an error processing your technical request. Please try again or rephrase your query.",
                citations=[],
                confidence=0.5,
                escalate=False,
                suggested_next_steps=[]
            )
        elif response_model.__name__ == "BillingResponse":
            fallback = response_model(
                response="I apologize, but I encountered an error processing your billing request. Please try again or rephrase your query.",
                action_taken=None,
                plan_details=None,
                invoice_summary=None,
                policy_citations=[],
                escalate=False
            )
        elif response_model.__name__ == "EscalationResponse":
            import uuid
            fallback = response_model(
                escalation_id=str(uuid.uuid4()),
                priority="P3",
                urgency="medium",
                sentiment="neutral",
                issue_category="general",
                conversation_summary="Fallback escalation due to system exception.",
                extracted_entities={},
                recommended_action="Review conversation logs.",
                human_handoff_payload={
                    "customer_id": None,
                    "full_history": [],
                    "notes_for_agent": "System fallback escalation."
                },
                trace_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            try:
                fallback = response_model()
            except Exception:
                fallback = response_model.construct()
                
        llm_end_time = datetime.now().isoformat()
        logger.info(f"[TRACE] {self.name}_llm_invoke_end={llm_end_time}")
        return fallback
