# Riverline AI: Technical Codebase Deep Dive

This document explains **exactly** what was built, file-by-file, logic-by-logic. This is the "under the hood" explanation you need for a technical interview.

---

## 1. The Core Architecture: "ReAct Agent via LangGraph"
**File:** `agent_graph.py`

### What we built
We did not use a simple "chatbot" chain. We built a **Stateful Reasoning Graph**.
- **Library**: `LangGraph` (by LangChain).
- **Function**: `create_react_agent(llm, tools, checkpointer)`.

### A. The Graph Structure (Directed Cyclic Graph)
Most AI chains are DAGs (Directed Acyclic Graphs). This agent is **Cyclic**.
- **Nodes**:
    1.  `agent` node: Calls the LLM to decide the next step.
    2.  `tools` node: Executes the Python function (EMI Calculator, Policy Check).
- **Edges**:
    - `agent` -> `tools` (If LLM wants to call a function).
    - `tools` -> `agent` (After function returns, go back to LLM to generate response).
    - `agent` -> `END` (If LLM is ready to answer).

### B. The State Schema (`MessagesState`)
The "Memory" is not just a string of text. It is a strictly typed schema:
```python
class MessagesState(TypedDict):
    messages: list[AnyMessage]  # [SystemMessage, HumanMessage, AIMessage, ToolMessage]
```
- When the user speaks, a `HumanMessage` is appended.
- When the agent thinks, an `AIMessage` (with `tool_calls` parameter) is appended.
- When a tool runs, a `ToolMessage` (with `tool_call_id`) is appended.
**Why this matters**: This structured history allows the LLM to know exactly *which* result belongs to *which* request, essential for complex multi-turn reasoning.

---

## 2. Memory & Persistence (Hybrid Architecture)
**File:** `agent_graph.py` (Persistence Strategy Section)

### The Logic (Detailed)
We implemented a **Hybrid Factory Pattern** for the Checkpointer:
```python
if os.getenv("DATABASE_URL"):
    # PROD: Uses PostgreSQL
    pool = ConnectionPool(conninfo=db_url)
    memory = PostgresSaver(pool)
else:
    # DEV: Uses SQLite (Local File)
    conn = sqlite3.connect("memory.sqlite")
    memory = SqliteSaver(conn)
```

1.  **Serialization**: The `checkpointer` uses `serde` (usually msgpack/json types) to serialize the entire `MessagesState` object into binary.
2.  **Storage**: It writes to a `checkpoints` table with columns: `thread_id`, `checkpoint_id` (UUID), and `checkpoint` (BLOB).
3.  **Restoration**: When `agent_graph.stream(..., config={"thread_id": "riverline_user_123"})` is called:
    - The graph query the DB for the *latest* `checkpoint_id` for that thread.
    - Deserializes the BLOB back into Python objects.
    - Resumes execution *exactly* where it left off.

---

## 3. Model Context Protocol (MCP) - "The Future of Data"
**Files:** `mcp_server.py` (Server), `mcp_client.py` (Client)

### Authentication & Transport
1.  **Transport**: We used `stdio` (Standard Input/Output) transport. The Client spawns the Server as a sub-process and talks via pipes. This is secure because it's local process communication, not open HTTP.
2.  **Protocol**:
    - **Client**: "Initialize?" -> **Server**: "Ready. Capabilities: [tools]".
    - **Client**: "CallTool: get_user_profile" -> **Server**: "JSON Result".
3.  **Why use MCP?**
    - **Security**: The Agent **never** sees the full database. It can only see what the MCP tools explicitly expose.
    - **Modularity**: You can rewrite `mcp_server.py` in Rust or Go, and the Python Agent wouldn't care.

---

## 4. Anti-Hallucination Strategy (System Prompting)
**File:** `app.py`

### logic of "Dynamic Injection"
In `app.py`, line 96:
```python
messages_to_send = [SystemMessage(content=sys_msg_content), HumanMessage(content=prompt)]
```
We do **not** rely on the Chat History for the System Prompt. We re-inject it **fresh** every single turn.
- **Why?** As conversation history grows (context window limits), older system prompts might get truncated or "diluted". By injecting it partially every turn, we ensure the "Rules of Engagement" (e.g., *Debt is ₹5,000*) are always the most recent, dominant instruction.

### The "Tool Use Failed" Fix
We simplified the prompt from:
> *"Call function get_user_profile(id='123')."* (BAD - Triggers XML Hallucination)
To:
> *"You have access to tools. Use them."* (GOOD - Triggers Native API Call)

**Technical Answer**: Modern LLMs (Llama 3) have "Tool Tokens" in their vocabulary. Explicit text instructions confuse them into outputting text instead of the special token. Removing the text instructions allowed the semantic tokens to take over.

---

## 5. Deployment & Scalability (System Design)

### Current Architecture (MVP)
*   **Compute**: Streamlit (Python Process).
*   **State Store**: Neon Postgres (Serverless).
*   **Vector Store**: None (Context Window only).

### Scaling to 10k Users (The "Senior" Answer)
1.  **Async Workers**: The current `agent_graph.stream()` blocks the web thread. For 10k users, we would push the input to a **Redis Queue**. A fleet of Python Workers (Celery) would pop the message, run the Agent, and push the result to a WebSocket.
2.  **Semantic Caching**: Use **Redis** to cache Tool Outputs. If 100 users ask "What is the policy?", we call the tool ONCE and cache it for 1 hour.
3.  **Auth**: Replace sidebar ID selection with **OAuth2 (Auth0)**. We would decode the JWT Token to get the `user_id`, then pass that to `get_user_profile`.

---

## Summary for Interview
"I built a **Stateful, Cyclic Reasoning Engine** using LangGraph. It features a **Pluggable Persistence Layer** that swaps between SQLite (Dev) and Postgres (Prod) using Factory Logic. I decoupled the Data Layer using the **Model Context Protocol** over stdio transport. I ensured robustness using **Dynamic System Prompt Injection** and solved semantic tool-calling drift by optimizing the prompt for Llama 3's native tokens."
