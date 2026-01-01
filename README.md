# 🦅 Riverline Debt Resolution Agent

**A Stateful, Multi-Agent AI System for Empathetic Debt Recovery.**

This project demonstrates an advanced **"Agentic Workflow"** built with **LangGraph** and **Llama 3.1 (8B Instant)**. Unlike simple chatbots, this agent uses a **Cyclic ReAct Graph** to reason, call tools, and negotiate debt settlements while strictly adhering to internal policies.

![Tech Stack](https://img.shields.io/badge/Stack-LangGraph%20%7C%20Llama%203.1%20%7C%20Streamlit-blue)
![Database](https://img.shields.io/badge/Memory-PostgreSQL%20%2B%20SQLite-green)
![Protocol](https://img.shields.io/badge/Data-Model%20Context%20Protocol%20(MCP)-orange)
![Status](https://img.shields.io/badge/Status-Cloud%20Stable-green)

---

## 🧠 Key Innovations

### 1. **Stateful Reasoning Graph (ReAct)**
Instead of a linear chain, the "Brain" is a cyclic graph (`agent_graph.py`).
- **Loop**: Thought -> Action (Tool Call) -> Observation -> Response.
- **Tools**: It uses a custom **EMI Calculator** and **Policy Engine** to verify numbers before speaking.

### 2. **Hybrid Persistence Architecture**
The agent remembers conversations indefinitely using a "Factory Pattern" for memory:
- **💻 Development**: Automatically uses **SQLite** (local file) for zero-setup coding.
- **🌍 Production**: Automatically switches to **Neon PostgreSQL** when `DATABASE_URL` is detected.

### 3. **Model Context Protocol (MCP)**
The Agent connects to user data (`user_profile.json`) via the **MCP Standard**.
- **Security**: The Agent cannot read the raw database. It asks an MCP Server (`mcp_server.py`) for specific data.
- **Modularity**: The data layer is decoupled from the cognitive layer.

### 4. **Anti-Hallucination Guardrails**
- **Dynamic System Prompting**: We inject the *exact* debt amount into the context window at every turn (`app.py`), preventing the LLM from inventing numbers.
- **Tool Binding**: We use Llama 3's native tool tokens to ensure reliable API calls.

---

## 🛠️ Installation

### 1. Clone & Install
```bash
git clone https://github.com/singhan14/Riverline.git
cd Riverline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file:
```env
GROQ_API_KEY="gsk_..."
# Optional: Add for Cloud Persistence
# DATABASE_URL="postgresql://user:pass@host/db"
```

### 3. Run Locally
```bash
streamlit run app.py
```

---

## 🚀 Deployment

This app is designed for **Streamlit Cloud**.
1. Push to GitHub.
2. Deploy on Streamlit Cloud.
3. Add `GROQ_API_KEY` and `DATABASE_URL` (Neon.tech) to usage Secrets.
4. The app automatically detects the cloud environment and upgrades to **PostgreSQL Storage**.

---

## 📂 Project Structure
- `agent_graph.py`: **The Brain**. Defines the LangGraph nodes and persistence logic.
- `app.py`: **The Interface**. Streamlit UI with multi-user simulation.
- `mcp_server.py`: **The Data Layer**. FastMCP server for user profiles.
- `evaluate_agent.py`: **QA**. Automated evaluation pipeline using LangSmith.

---
*Built by Singhan Yadav.*
