"""
guardrails/validators.py — Centralized architectural safeguards and validation engine.
"""
import re
from typing import List, Dict, Any
from models.state import ConversationState
from app_logging.logger import get_logger

logger = get_logger(__name__)

def validate_agent_domain_response(agent_name: str, content: str) -> str:
    """
    Enforces strict domain-isolation boundaries by sentence filtering,
    while preserving newlines, list formatting, and numbering.
    """
    if not content:
        return content
        
    lines = content.split('\n')
    filtered_lines = []
    
    for line in lines:
        if not line.strip():
            filtered_lines.append(line)
            continue
            
        # Split this line into sentences
        # Use negative lookbehind to ensure we don't split on a list digit like "1." or "2."
        sentences = re.split(r'(?<!\b\d)(?<=[.!?])\s+', line)
        filtered_sentences = []
        
        if agent_name == "technical":
            # Technical response must NEVER discuss plans, Enterprise features, pricing, upgrades, billing policies.
            billing_terms = [
                "pricing", "price", "billing", "payment", "cost", "dollar",
                "upgrade from", "upgrade to", "upgrade policy", "starter plan", "pro plan",
                "enterprise feature", "audit log export", "dedicated support", "custom contract",
                "refund", "subscription", "plan tier", "sales@clouddash.io", "customer success manager",
                "upgrade to enterprise", "upgrade your plan", "billing plan"
            ]
            for s in sentences:
                s_lower = s.lower()
                if any(term in s_lower for term in billing_terms):
                    continue
                filtered_sentences.append(s)
                
        elif agent_name == "billing":
            # Billing response must NEVER discuss technical troubleshooting steps, SAML/SSO debugging, APIs, integrations.
            technical_terms = [
                "troubleshooting", "saml debugging", "metadata", "acs url", "signing certificate",
                "verify idp", "check your idp", "identity provider", "sso issue", "sso integration",
                "aws console", "aws alerts", "update credentials", "save & test", "connection successful"
            ]
            for s in sentences:
                s_lower = s.lower()
                if any(term in s_lower for term in technical_terms):
                    continue
                filtered_sentences.append(s)
        else:
            filtered_sentences = sentences
            
        if filtered_sentences:
            filtered_lines.append(" ".join(filtered_sentences))
            
    return "\n".join(filtered_lines).strip()

def validate_grounding(response_text: str, retrieved_docs: list, user_query: str = "") -> str:
    """
    Universally ensures no brand/product names queried by the user are mentioned in the response
    unless they are supported and grounded by the retrieved KB documents.
    """
    logger.info("[VALIDATOR] starting grounding validation")
    if not response_text:
        logger.info("[VALIDATOR] grounding validation complete (empty content)")
        return response_text
        
    import re
    if not user_query:
        return response_text
        
    # Extract unique capital brand names queried by the user
    query_words = re.findall(r'\b[A-Za-z]{3,}\b', user_query)
    brand_names = []
    exclude_set = {
        "clouddash", "cloud", "dash", "the", "this", "that", "sso", "saml", "aws", 
        "pro", "enterprise", "starter", "plan", "how", "why", "what", "where", "who", "when", 
        "yes", "not", "sso", "saml", "okta", "our", "get", "we", "give", "can", "please", "you", 
        "your", "are", "but", "first", "now", "out", "new"
    }
    for w in query_words:
        if w[0].isupper() and w.lower() not in exclude_set:
            brand_names.append(w.lower())
            
    # For each brand name mentioned in the response, it MUST be grounded in at least one KB snippet
    response_lower = response_text.lower()
    if brand_names:
        for brand in brand_names:
            # Enforce whole-word boundaries to avoid false substring matches (e.g. 'our' in 'your')
            if re.search(r'\b' + re.escape(brand) + r'\b', response_lower):
                brand_grounded = False
                if retrieved_docs:
                    for doc in retrieved_docs:
                        snippet = ""
                        if hasattr(doc, "snippet"):
                            snippet = doc.snippet
                        elif isinstance(doc, dict):
                            snippet = doc.get("snippet", "") or doc.get("content", "") or doc.get("text", "")
                        elif isinstance(doc, str):
                            snippet = doc
                        if brand in snippet.lower():
                            brand_grounded = True
                            break
                if not brand_grounded:
                    # Hallucination detected
                    logger.info(f"[VALIDATOR] grounding validation complete (hallucination detected: {brand})")
                    return "I could not find information about this feature in the CloudDash knowledge base."
                
    logger.info("[VALIDATOR] grounding validation complete")
    return response_text

def validate_workflow_execution(state: ConversationState) -> None:
    """
    Verifies full intent processing, domain compliance, and grounding before API response.
    """
    if not state or not state.messages:
        return
        
    # Find index of the most recent user query in the turn
    last_user_idx = -1
    for idx, msg in enumerate(state.messages):
        if msg.role == "user":
            last_user_idx = idx
            
    user_query = ""
    if last_user_idx != -1:
        user_query = state.messages[last_user_idx].content
        current_turn_msgs = state.messages[last_user_idx + 1:]
        for msg in current_turn_msgs:
            if msg.role == "assistant":
                # 1. Strip out-of-domain responses
                msg.content = validate_agent_domain_response(msg.agent_name, msg.content)
                # 2. Prevent ungrounded hallucinations
                retrieved_chunks = []
                if msg.citations:
                    for c in msg.citations:
                        retrieved_chunks.append({"title": getattr(c, "title", ""), "snippet": getattr(c, "snippet", "")})
                msg.content = validate_grounding(msg.content, retrieved_chunks, user_query)
