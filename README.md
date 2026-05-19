# Multi-Agent RAG Support System (CloudDash)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Inference-Groq-green.svg)](https://groq.com/)

A production-grade, multi-agent AI customer support system built for **CloudDash**, a fictional cloud monitoring SaaS. This project demonstrates advanced **Agentic Orchestration** using LangGraph, **Hybrid RAG** (Vector + Keyword) retrieval, and enterprise-grade **Guardrails**.

## 🎥 System Demo
Watch the multi-agent system, transitions, sequential handovers, and guardrails in action:
- **[Download / View Demo Video](./demo.mp4)**

## 🔗 Live Deployments
- **🖥️ Live Frontend (Streamlit Cloud):** [arya-akshat-multi-agent-rag-support-uistreamlit-app-pdboh8.streamlit.app](https://arya-akshat-multi-agent-rag-support-uistreamlit-app-pdboh8.streamlit.app/)
- **⚡ Live Backend API (Render):** [multi-agent-rag-support.onrender.com](https://multi-agent-rag-support.onrender.com)

> [!NOTE]
> **Cold Start Warning:** The backend is hosted on a Render free instance. If the service is inactive, it will sleep. Live link testing may require **2-5 minutes** to spin up and become active on first load.


## ✨ Key Features

- **🧠 Agentic Orchestration**: Uses a central LangGraph state machine to route queries between specialized agents (Triage, Technical, Billing, Escalation).
- **🔎 Hybrid RAG Pipeline**: Combines dense vector search (ChromaDB) with sparse keyword search (BM25) and uses **Reciprocal Rank Fusion (RRF)** for optimal retrieval.
- **🛡️ Enterprise Guardrails**: Includes a PII Scrubber to redact sensitive data and a structured output validator to ensure 100% reliable JSON responses.
- **📈 Intent-Aware Routing**: The Triage Agent extracts intent, sentiment, and urgency to ensure high-priority issues are escalated instantly.
- **⚡ High Performance**: Powered by **Groq LPUs** for near-instant inference and low latency.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/multi-agent-rag-support.git
   cd multi-agent-rag-support
   ```

2. **Setup Environment:**
   ```bash
   make setup
   ```

3. **Configure API Keys:**
   Create a `.env` file and add your Groq key:
   ```env
   GROQ_API_KEY="gsk_your_key_here"
   GROQ_MODEL="llama-3.1-8b-instant"
   ```

## 🖥️ Usage

1. **Start the API Backend:**
   ```bash
   make run-api
   ```

2. **Start the Streamlit UI:**
   ```bash
   make run-ui
   ```

## 🧪 Testing

Run the full test suite to verify agent logic and retrieval accuracy:
```bash
make test
```

## 🏗️ Architecture

Refer to [ARCHITECTURE.md](./ARCHITECTURE.md) for a detailed breakdown of the system design, including Mermaid diagrams and component interaction flows.

## 📐 Design Decisions

1. **State Machine Orchestration via LangGraph**: Rather than relying on autonomous agent routing loops which are prone to infinite loops and high latency, we chose LangGraph to build a deterministic state machine. Routing decisions are structured and validated at each step.
2. **Hybrid RAG (Vector + Keyword Search)**: Combining ChromaDB dense embeddings with BM25 keyword matching ensures that queries containing exact technical keywords (e.g., API names or error codes) are retrieved with high precision while semantic queries are handled by vector search. Reciprocal Rank Fusion (RRF) merges the results.
3. **Decoupled Configuration**: All agent system prompts, capabilities, and routing schemas are declared externally in `config/*.yaml` files, allowing new agents to be integrated without modifying the orchestrator codebase.
4. **Input/Output Guardrails**: An input guardrail intercepts security attacks (prompt injection) and scrubs PII before the LLM execution. Output guardrails verify the validity of pricing and policy answers against the context, rewriting unsupported claims.

## ⚖️ Trade-offs

1. **In-Memory Session Store vs. Persistent DB**:
   * *Trade-off*: We used a thread-safe, TTL-based in-memory store for session states.
   * *Consequence*: Highly performant and simple for prototyping, but does not scale horizontally. In production, this would be replaced with Redis.
2. **Local Vector Store (ChromaDB) vs. Hosted Vector DB**:
   * *Trade-off*: ChromaDB is run locally in-process.
   * *Consequence*: Simplifies local setup and deployments without external dependencies, but increases container memory footprint and limits horizontal scaling.
3. **Single LLM Provider (Groq) vs. Multi-Model Failover**:
   * *Trade-off*: The system is designed primarily around Groq's high-speed inference.
   * *Consequence*: Extremely low latency (near real-time streaming), but lacks automatic model fallback if the primary API experiences rate limits or outages.
