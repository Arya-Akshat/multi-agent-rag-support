# CloudDash Multi-Agent Support Engine

A multi-agent customer support system built for CloudDash, a fictional cloud infrastructure monitoring platform. The system uses LangGraph to orchestrate state, a hybrid vector/keyword RAG pipeline for retrieval, and robust safety guardrails.

## Live Access

* **Web UI (Streamlit):** [arya-akshat-multi-agent-rag-support-uistreamlit-app-pdboh8.streamlit.app](https://arya-akshat-multi-agent-rag-support-uistreamlit-app-pdboh8.streamlit.app/)
* **Backend API (Render):** [multi-agent-rag-support.onrender.com](https://multi-agent-rag-support.onrender.com)

> **Render Cold Start:** The backend API runs on a Render free tier instance. If it has gone to sleep due to inactivity, it may take 2-5 minutes to boot on the first request.

## System Demo

A walkthrough showing agent transitions, sequential handovers, and guardrails in action is recorded here:
* **[Download / View Demo Video](./demo.mp4)**

## Key Features

* **State-Machine Routing (LangGraph)**: Uses a central graph orchestrator to manage state and transition between the specialized agents (Triage, Technical Support, Billing, and Escalation).
* **Hybrid RAG Pipeline**: Combines dense vector search (ChromaDB) with sparse keyword search (BM25) and merges rankings using Reciprocal Rank Fusion (RRF).
* **Guardrails**: Intercepts input prompt injections, redacts PII before processing, and runs output checks to verify pricing or feature claims against retrieved sources.
* **Low Latency**: Built on Groq's API for fast response generation.

## Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Arya-Akshat/multi-agent-rag-support.git
   cd multi-agent-rag-support
   ```

2. **Configure Environment:**
   Run the setup script or configure dependencies manually:
   ```bash
   make setup
   ```

3. **Set Up API Keys:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY="gsk_your_groq_key_here"
   GROQ_MODEL="llama-3.1-8b-instant"
   ```

## Running the Project

1. **Start the API server:**
   ```bash
   make run-api
   ```

2. **Start the frontend UI:**
   ```bash
   make run-ui
   ```

## Testing

Run unit and integration tests using pytest:
```bash
make test
```

## Design Decisions

* **Deterministic Graph vs. Autonomous Loops**: Autonomous agent loops are prone to hallucinations, infinite loops, and high token costs. We used a structured state machine (via LangGraph) to control routing transitions explicitly.
* **Hybrid Search with RRF**: Exact matches (like error codes, metric names, or CLI commands) are often missed by semantic vector searches. Combining BM25 keyword matching with dense ChromaDB vectors ensures both semantic and literal queries retrieve relevant guides.
* **Declarative Configs**: System prompts, specialist routing schemas, and agent profiles are separated into YAML configuration files in `config/` rather than being hardcoded. This allows quick adjustments to system instructions.
* **Output Verification**: LLMs frequently hallucinate negative claims (e.g. "We do not support feature X") when the retrieved context simply lacks the answer. The output guardrail validates assertions and rewrites unsupported claims to a standard fallback response.

## Trade-offs

* **In-Memory Session Store**: To simplify development and local runs, session state and conversation memory are kept in-memory with a TTL mechanism. For production scaling, this would be backed by Redis.
* **In-Process ChromaDB**: Using ChromaDB in-process removes the need for hosted database infrastructure but increases application memory requirements and limits horizontal scaling.
* **Single LLM Provider**: We rely on Groq for execution speed, but we lack automatic failover rules to fallback to OpenAI or Anthropic if Groq's API encounters rate limits or downtime.
