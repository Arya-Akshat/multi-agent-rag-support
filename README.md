# Multi-Agent RAG Support System (CloudDash)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Inference-Groq-green.svg)](https://groq.com/)

A production-grade, multi-agent AI customer support system built for **CloudDash**, a fictional cloud monitoring SaaS. This project demonstrates advanced **Agentic Orchestration** using LangGraph, **Hybrid RAG** (Vector + Keyword) retrieval, and enterprise-grade **Guardrails**.

## 🔗 Live Deployments

- **🖥️ Live Frontend (Streamlit Cloud):** [arya-akshat-multi-agent-rag-support-uistreamlit-app-pdboh8.streamlit.app](https://arya-akshat-multi-agent-rag-support-uistreamlit-app-pdboh8.streamlit.app/)
- **⚡ Live Backend API (Render):** [multi-agent-rag-support.onrender.com](https://multi-agent-rag-support.onrender.com)


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
