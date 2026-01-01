from typing import Annotated, Literal, TypedDict, Union
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import sqlite3
import os
from dotenv import load_dotenv

# ... (tools remain same) ...

def build_graph():
    """Constructs the LangGraph ReAct Agent."""
    tools = get_tools()
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Persistence Strategy (Hybrid: Local vs Cloud)
    db_url = os.getenv("DATABASE_URL")
    
    if db_url and "postgres" in db_url:
        # Production: Use PostgreSQL (requires psycopg_pool)
        # Note: We create a connection pool. 
        # In a real app, manage this pool's lifecycle (open/close) carefully.
        # For simplicity here, we open it globally or per request. 
        # Ideally, pass the pool instance.
        print("🌍 Using Cloud Database (PostgreSQL)...")
        pool = ConnectionPool(conninfo=db_url, max_size=20)
        memory = PostgresSaver(pool)
        
        # NOTE: PostgresSaver needs setup() to create tables if they don't exist
        memory.setup()
    else:
        # Development: Use Local SQLite
        print("💻 Using Local Database (SQLite)...")
        conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
        memory = SqliteSaver(conn)
    
    graph = create_react_agent(llm, tools, checkpointer=memory)    
    return graph

load_dotenv()

# --- 1. Custom Tools for Debt Collection ---

@tool
def calculate_emi(principal: float, rate_of_interest: float, tenure_months: int) -> str:
    """
    Calculates the Equated Monthly Installment (EMI) for a loan settlement.
    
    Args:
        principal: The amount of debt to be settled (e.g., 5000).
        rate_of_interest: The annual interest rate (percentage, e.g., 10 for 10%).
        tenure_months: The number of months to pay back the debt.
        
    Returns:
        A formatted string with the monthly EMI amount.
    """
    try:
        p = principal
        r = rate_of_interest / (12 * 100) # Monthly rate
        n = tenure_months
        
        if r == 0:
            emi = p / n
        else:
            emi = p * r * ((1 + r)**n) / (((1 + r)**n) - 1)
            
        return f"The EMI for a principal of ₹{p} at {rate_of_interest}% for {n} months is: ₹{emi:.2f}"
    except Exception as e:
        return f"Error calculating EMI: {e}"

@tool
def check_settlement_policy(debt_amount: float) -> str:
    """
    Checks the internal policy to see if a settlement discount is allowed.
    
    Args:
        debt_amount: The current total debt amount.
        
    Returns:
        The maximum discount percentage allowed.
    """
    if debt_amount > 10000:
        return "For debts over ₹10,000, we can offer a maximum discount of 20% on one-time settlement."
    else:
        return "For debts under ₹10,000, we can offer a maximum discount of 30% on one-time settlement."

# --- 2. Build the ReAct Agent ---

def get_tools():
    # Local Tools
    tools = [calculate_emi, check_settlement_policy]
    
    # MCP Tools (Filesystem Profile)
    try:
        from mcp_client import mcp_client
        mcp_tools = mcp_client.get_tools()
        tools.extend(mcp_tools)
        print(f"✅ Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")
    except Exception as e:
        print(f"⚠️ Failed to load MCP tools: {e}")
        
    return tools

def context_trimmer(state):
    """
    Trims chat history to the last 10 messages and sanitizes it
    to prevent 'INVALID_CHAT_HISTORY' errors (Dangling ToolCalls).
    """
    try:
        # 1. Standardize Input (List vs Dict)
        if isinstance(state, dict):
            messages = state.get("messages", [])
        elif isinstance(state, list):
            messages = state
        else:
            messages = getattr(state, "messages", state)
            
        if not isinstance(messages, list):
            messages = [messages]
            
        # 2. Slice (Keep last 10 for context)
        # We take a slightly larger window to increase chance of finding pairs
        trimmed = messages[-10:] if len(messages) > 10 else messages
        
        # 3. Sanitize: Remove dangling ToolCalls or Orphan ToolMessages
        sanitized = []
        skip_next = False
        
        # We iterate and rebuild the list to ensure coherence
        for i, msg in enumerate(trimmed):
            # If AI Message with Tool Calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Check if next message is a ToolMessage
                if i + 1 < len(trimmed):
                    next_msg = trimmed[i+1]
                    if hasattr(next_msg, 'tool_call_id') or next_msg.type == 'tool':
                        # Valid Pair: Keep AI
                        sanitized.append(msg)
                    else:
                        # Invalid: AI wanted tool, but next wasn't tool. 
                        # Convert to text-only to avoid crash (Strip tool_calls)
                        msg_copy = msg.model_copy()
                        msg_copy.tool_calls = []
                        msg_copy.content = str(msg.content) + " [Tool Call Aborted]"
                        sanitized.append(msg_copy)
                else:
                    # Last message is ToolCall - This is fine (Model will run tool)
                    sanitized.append(msg)
            
            # If ToolMessage (Orphan check)
            elif msg.type == 'tool':
                # Check if we have a preceding AI message in sanitized list
                # (Simple check: Just keep it if we added the parent)
                # For simplicity in this trimmer: We accept ToolMessages 
                # because omitting them confuses the model more.
                sanitized.append(msg)
            
            else:
                # User / System / AI Text
                sanitized.append(msg)
                
        return sanitized
        
    except Exception as e:
        print(f"⚠️ Context Trimmer Warning: {e}")
        return state # Fail open

def build_graph():
    """Constructs the LangGraph ReAct Agent."""
    tools = get_tools()
    
    # Initialize LLM with the correct model
    # Hardcoded to 'llama-3.1-8b-instant' to prevent config errors with decommissioned models
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7, 
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Persistence Strategy (Hybrid: Local vs Cloud)
    db_url = os.getenv("DATABASE_URL")
    
    if db_url and "postgres" in db_url:
        print("🌍 Using Cloud Database (PostgreSQL)...")
        # Initialize connection pool with Serverless-friendly settings
        # min_size=0: Don't hold connections (Neon scales to 0)
        # max_lifetime=120: Recycle connections frequently to avoid SSL timeouts
        pool = ConnectionPool(
            conninfo=db_url,
            min_size=0,
            max_size=20,
            max_lifetime=60, # Recycle very frequently
            check=ConnectionPool.check_connection, # Force PING before use
            kwargs={"autocommit": True}
        )
        memory = PostgresSaver(pool)
        memory.setup()
    else:
        print("💻 Using Local Database (SQLite)...")
        conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
        memory = SqliteSaver(conn)
    
    # Pass 'messages_modifier' to trim context before invoking LLM
    graph = create_react_agent(llm, tools, checkpointer=memory, messages_modifier=context_trimmer)    
    return graph

# Global instance
agent_graph = build_graph()
