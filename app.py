import streamlit as st
import os
import json
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent_graph import agent_graph

# Load environment variables
load_dotenv()

# Streamlit App Config
st.set_page_config(page_title="Riverline Debt Resolution", page_icon="💸")
st.title("Riverline AI: Debt Resolution Assistant")

# Verify API Keys
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# --- Load User Profiles ---
PROFILE_FILE = "user_profile.json"
try:
    with open(PROFILE_FILE, "r") as f:
        users_db = json.load(f)
except FileNotFoundError:
    st.error("User database not found!")
    st.stop()

# --- Sidebar: User Switcher ---
with st.sidebar:
    st.header("👤 Select Persona")
    # Dropdown to select user
    user_ids = list(users_db.keys())
    # Format options as "Name (ID)"
    user_options = {uid: f"{u['name']} ({uid})" for uid, u in users_db.items()}
    
    selected_uid = st.selectbox(
        "Simulate User:", 
        options=user_ids, 
        format_func=lambda x: user_options[x]
    )
    
    current_user = users_db[selected_uid]
    
    st.divider()
    st.header("Account Details")
    st.write(f"**Name:** {current_user['name']}")
    st.write(f"**Outstanding Debt:** {current_user['currency']} {current_user['outstanding_debt']}")
    st.write(f"**Risk Score:** {current_user['risk_score']}")
    
    st.divider()
    st.markdown("### 🧠 Try Model Context Protocol")
    st.caption("Ask me these to test real-time data access:")
    st.code("What is my risk score?")
    st.code(f"Update my preference to Email (User ID: {selected_uid})")

# --- Session Management ---

# Ensure thread_id is unique per user (Persistent Memory key)
if "current_uid" not in st.session_state:
    st.session_state.current_uid = selected_uid

# If user changed, clear chat history from UI (Agent memory persists in SQLite though!)
if st.session_state.current_uid != selected_uid:
    st.session_state.messages = []
    st.session_state.current_uid = selected_uid
    st.rerun()

# Thread ID based on User ID -> "riverline_user_123"
# This ensures that when you switch back to John Doe, his chat history remains!
thread_id = f"riverline_{selected_uid}"

# SYSTEM PROMPT (Dynamic based on selected User)
# Define this globally so it's available for the chat input handler
sys_msg_content = f"""
You are 'River', an empathetic Debt Resolution Agent for Riverline AI. 
The user is **{current_user['name']}** (ID: {selected_uid}).
Outstanding Debt: {current_user['currency']} {current_user['outstanding_debt']}.
Risk Score: {current_user['risk_score']}.

**Rules:**
1. Be polite but firm.
2. CURRENT DEBT IS {current_user['currency']} {current_user['outstanding_debt']}. Do NOT hallucinate.
3. Settlement Policy: 
   - If User offers > 70% of debt, ACCEPT.
   - If User offers < 70%, REJECT and explain max discount is 30%.

You have access to tools for EMI calculation and Profile lookup. Use them when needed.
"""

# Initialize Chat History if empty
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    st.session_state.messages.append(SystemMessage(content=sys_msg_content))

    # Initial Greeting
    greeting = f"Hello {current_user['name']}. I'm River from Riverline AI. I see you have an outstanding balance of {current_user['currency']} {current_user['outstanding_debt']}. I'm here to help. How are you?"
    st.session_state.messages.append(AIMessage(content=greeting))

# Display Chat History
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat Input
if prompt := st.chat_input("Type your response..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Pass the Thread ID for SQLite Persistence
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use stream() to get events
        try:
            # We send the full history ensuring SystemPrompt is always 0
            # For ReAct, we just send the new message, BUT since we want to enforce System Prompt context
            # we rely on the fact that ReAct usually takes 'messages' as state. 
            # Ideally, we should just send [HumanMessage] and let graph handle state, 
            # BUT our graph is generic. Let's send the prompt.
            # The SystemMessage is already in history? No, LangGraph manages state. 
            # We need to send the SystemMessage only ONCE per thread or via `state_modifier` (which failed earlier).
            # Workaround: We prepend SystemMessage to the input if it's the FIRST run, but here it's cleaner to just
            # prepend it to the input list every time if the model context allows, OR trust the Checkpointer.
            # TRUSTING CHECKPOINTER: If we already sent SysMsg, it's in DB.
            # However, for simplicity and robustness against context window limits, let's just send the User message.
            # The FIRST time we spoke (above), we setup the history. `agent_graph` uses `messages` state.
            
            # Correction: `create_react_agent` manages state automatically.
            # If we want to inject System Prompt, and we don't have `messages_modifier`, 
            # we should update the graph state manually.
            # For this Demo: sending [SysMsg, UserMsg] every turn is stateless but forceful. 
            # With Checkpointer, we should inspect if history exists.
            
            # Better Strategy: Just prepend SystemPrompt to the current input list.
            messages_to_send = [SystemMessage(content=sys_msg_content), HumanMessage(content=prompt)]
            
            events = agent_graph.stream(
                {"messages": messages_to_send}, 
                config=config, 
                stream_mode="values"
            )
            
            for event in events:
                if "messages" in event:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        full_response = last_msg.content
                        message_placeholder.markdown(full_response)
            
            if not full_response:
                full_response = "I'm thinking..."
                message_placeholder.markdown(full_response)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "I apologize, but I encountered an error connecting to my brain. Please try again."
    
    # Add to history
    st.session_state.messages.append(AIMessage(content=full_response))
