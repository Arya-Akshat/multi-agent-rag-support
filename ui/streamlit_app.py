"""
ui/app.py — Streamlit Frontend.

Provides a chat interface for the CloudDash AI Support System.
Connects to the FastAPI backend.
"""

import streamlit as st
import httpx

# Configure Streamlit page
st.set_page_config(
    page_title="CloudDash Support",
    page_icon="☁️",
    layout="centered"
)

# Constants
API_BASE_URL = "http://localhost:8000/api/v1"

# Initialize Session State
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.escalated = False


def start_conversation():
    """Call API to start a new conversation."""
    try:
        response = httpx.post(f"{API_BASE_URL}/conversation/start")
        response.raise_for_status()
        data = response.json()
        st.session_state.conversation_id = data["conversation_id"]
        st.session_state.messages = [{"role": "assistant", "content": data["message"], "agent": "triage"}]
        st.session_state.escalated = False
    except Exception as e:
        st.error(f"Failed to connect to support server: {e}")


def send_message(user_text: str):
    """Send user message to API and get response."""
    # Add user message to UI immediately
    st.session_state.messages.append({"role": "user", "content": user_text, "agent": "user"})
    
    try:
        payload = {
            "conversation_id": st.session_state.conversation_id,
            "message": user_text
        }
        import time
        start_api = time.time()
        
        print("[UI] request sent")
        with st.spinner("CloudDash AI is thinking..."):
            response = httpx.post(
                f"{API_BASE_URL}/conversation/message", 
                json=payload,
                timeout=60.0
            )
            
        elapsed_api = time.time() - start_api
        print("[UI] response received")
        print(f"[STREAMLIT DEBUG] API Response time: {elapsed_api:.3f} seconds")
            
        if response.status_code == 400 and "escalated" in response.text:
            st.session_state.escalated = True
            st.error("This conversation has been escalated to a human. Please wait.")
            return
            
        response.raise_for_status()
        data = response.json()
        print("[UI] response parsed")
        
        print(f"[STREAMLIT DEBUG] New messages received: {len(data.get('messages', []))} | Total handovers: {len(data.get('handovers', []))}")
        
        # Display handover toasts if any
        for h in data.get("handovers", []):
            st.toast(f"🔄 Routed to **{h['target'].capitalize()} Support**\n\n_{h['reason']}_", icon="ℹ️")
            
        # Update state
        st.session_state.escalated = data.get("escalated", False)
        
        # Format response with citations
        new_msgs = data.get("messages", [])
        if new_msgs:
            for m in new_msgs:
                content = m["content"]
                citations = m.get("citations", [])
                if citations:
                    content += "\n\nSources:\n"
                    seen_sources = set()
                    for c in citations:
                        source_key = (c['title'], c.get('url', '#'))
                        if source_key not in seen_sources:
                            seen_sources.add(source_key)
                            content += f"- [{c['title']}]({c.get('url', '#')})\n"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "agent": m["agent"]
                })
        else:
            content = data["response"]
            citations = data.get("citations", [])
            if citations:
                content += "\n\nSources:\n"
                seen_sources = set()
                for c in citations:
                    source_key = (c['title'], c.get('url', '#'))
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        content += f"- [{c['title']}]({c.get('url', '#')})\n"
                    
            st.session_state.messages.append({
                "role": "assistant", 
                "content": content, 
                "agent": data["agent"]
            })
            
    except Exception as e:
        st.error(f"Error communicating with support server: {e}")


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.title("☁️ CloudDash Multi-Agent Support")
st.markdown("Powered by the **Multi-Agent RAG Support Engine**")

# Start conversation if none exists
if st.session_state.conversation_id is None:
    start_conversation()

# Display Chat History
last_agent = None
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        last_agent = None
    else:
        agent_name = msg.get("agent", "assistant")
        if last_agent is not None and last_agent != agent_name:
            st.markdown(
                """
                <div style="margin: 15px 0; border-top: 1px dashed #ccc; border-bottom: 1px dashed #ccc; padding: 6px 0; text-align: center; color: #666; font-size: 0.9em; font-weight: bold;">
                    [Automatic Handover]
                </div>
                """,
                unsafe_allow_html=True
            )
        with st.chat_message("assistant"):
            st.markdown(f"**Agent: {agent_name.capitalize()}**")
            st.markdown("---")
            st.markdown(msg["content"])
        last_agent = agent_name

# Chat Input
if st.session_state.escalated:
    st.warning("A human support representative will be with you shortly. The AI is now paused.")
else:
    user_input = st.chat_input("Type your message here...")
    if user_input:
        send_message(user_input)
        st.rerun()

# Sidebar
with st.sidebar:
    st.header("🏢 Agent Roster")
    st.markdown("- **Triage**: Intent Router")
    st.markdown("- **Technical**: RAG Expert")
    st.markdown("- **Billing**: Policy & Pricing")
    st.markdown("- **Escalation**: Human Handoff")
    
    st.divider()
    
    st.header("Debug Info")
    if st.button("Restart Conversation", use_container_width=True):
        st.session_state.conversation_id = None
        st.rerun()
        
    if st.session_state.conversation_id:
        st.caption(f"Session ID: {st.session_state.conversation_id[:8]}...")
        st.caption(f"Escalated: {st.session_state.escalated}")
