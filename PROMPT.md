You are an expert AI systems architect and senior AI engineer.

Your task is to build a COMPLETE production-style prototype for the following assessment:

====================================================
PROJECT
====================================================

Build a Multi-Agent Customer Support System for a fictional SaaS company called CloudDash.

The system must include:
- Multi-agent orchestration
- RAG pipeline
- Context-aware routing
- Agent handovers
- Escalation workflows
- Structured logging
- Guardrails
- Config-driven extensibility
- REST API
- Minimal UI
- Tests
- Documentation
- Deployment-ready structure

The implementation must prioritize:
1. Clean architecture
2. Modularity
3. Production engineering practices
4. Working functionality
5. Readability
6. Extensibility

This is NOT a toy chatbot.
This is a miniature AI support platform.

====================================================
AI ASSISTANT BEHAVIOR & WORKFLOW RULES (MANDATORY)
====================================================

Since you are executing this inside an IDE environment, you MUST follow these operational rules:

1. PAUSE AND WAIT: After completing a phase and its validation steps, you MUST STOP and wait for my explicit approval (e.g., "Proceed to Phase X") before writing any code for the next phase.
2. NO LAZY CODING: Do not write `pass`, `TODO`, or `// Implement logic here` for core requirements. Write the actual implementation. 
3. DIRECTORY CREATION: Always ensure directories exist before attempting to write files into them.
4. FULL FILE CONTEXT: When modifying a file, ensure you do not accidentally delete existing imports or logic. 
5. ERROR RECOVERY: If a validation step fails, you must debug and fix it within the current phase before asking to move on.
6. ONE FILE AT A TIME: Write and save each file completely before moving to the next. Never write partial files.
7. EXPLICIT IMPORTS: Every file must have all imports written at the top. Never assume an import is available from context.
8. NO CIRCULAR IMPORTS: Before creating any module that imports from another, verify the import chain is acyclic. If a circular import risk exists, use lazy imports or dependency injection to resolve it.
9. RELATIVE VS ABSOLUTE IMPORTS: Use absolute imports throughout (e.g., `from agents.base_agent import BaseAgent`, not `from .base_agent import BaseAgent`) to ensure compatibility across execution contexts (pytest, uvicorn, streamlit).
10. SHOW FULL FILE ON MODIFICATION: When modifying an existing file, always output the COMPLETE updated file — never just the diff or changed section.

====================================================
IMPORTANT EXECUTION RULES
====================================================

You MUST implement the project PHASE BY PHASE.

AFTER EVERY PHASE:
1. Run validation tests
2. Verify imports
3. Verify API/server starts correctly
4. Verify no broken dependencies
5. Verify previous phases still work
6. Explain what was completed
7. Explain what remains
8. Only then move to next phase

NEVER dump the whole codebase at once.

At the end of every phase:
- provide file tree updates
- explain architectural decisions
- explain tradeoffs
- explain any assumptions
- WAIT FOR USER APPROVAL TO PROCEED.

DO NOT skip testing.

====================================================
FILE CREATION ORDER WITHIN EACH PHASE (MANDATORY)
====================================================

Within each phase, always create files in this order:
1. __init__.py files for any new packages (always create these first)
2. Data models and Pydantic schemas (no dependencies on other local modules)
3. Utility/helper modules (depend only on models)
4. Core logic modules (depend on models + utils)
5. Integration/orchestration modules (depend on core logic)
6. Entry points (API routes, CLI, UI — depend on everything below)

This order prevents import errors during incremental development.

====================================================
TECH STACK (MANDATORY)
====================================================

Backend:
- Python 3.11+
- FastAPI

LLM Orchestration:
- LangGraph preferred (Use `StateGraph` for strict state management)
- LangChain allowed where useful
- Instructor OR LangChain Structured Output Parsers (to guarantee Pydantic validation of agent outputs)

Embeddings:
- sentence-transformers OR OpenAI embeddings

Vector Store:
- ChromaDB

Validation/Data Models:
- Pydantic

Logging:
- structlog OR loguru

Config:
- YAML
- python-dotenv (for environment variable management)

Frontend:
- Streamlit

Deployment:
- Render/Railway compatible

Testing:
- pytest
- pytest-asyncio (for async testing)

====================================================
DEPENDENCY & VERSION PINNING RULES
====================================================

- Pin all dependencies to exact versions in requirements.txt (e.g., `fastapi==0.111.0`, not `fastapi>=0.111.0`).
- Use a separate requirements-dev.txt for test and lint dependencies.
- After writing requirements.txt, verify that no two packages have conflicting version constraints before proceeding.
- LangGraph and LangChain versions must be explicitly compatible — check that `langgraph`, `langchain-core`, and `langchain-community` versions are aligned.
- ChromaDB version must be pinned and compatible with the chosen embedding library.
- If using sentence-transformers, pin torch to a CPU-only version to avoid bloated installs: `torch==2.2.0+cpu`.

====================================================
ARCHITECTURE REQUIREMENTS
====================================================

The architecture MUST follow strict separation of concerns.

Use this structure:

project_root/
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── triage_agent.py
│   ├── technical_agent.py
│   ├── billing_agent.py
│   ├── escalation_agent.py
│   ├── registry.py
│   └── orchestrator.py
│
├── api/
│   ├── __init__.py
│   ├── main.py (Must include CORS middleware)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── conversation.py
│   │   └── health.py
│   └── dependencies.py
│
├── config/
│   ├── agents.yaml
│   ├── routing.yaml
│   └── settings.py
│
├── retrieval/
│   ├── __init__.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── reranker.py
│   └── query_rewriter.py
│
├── handover/
│   ├── __init__.py
│   ├── models.py
│   ├── manager.py
│   ├── audit_logger.py
│   └── summarizer.py
│
├── knowledge_base/
│   ├── data/
│   │   └── articles/        ← raw JSON KB articles (one file per article)
│   └── generated_articles/  ← auto-generated or seeded articles
│
├── memory/
│   ├── __init__.py
│   ├── conversation_store.py
│   └── session_manager.py
│
├── models/
│   ├── __init__.py
│   ├── conversation.py
│   ├── responses.py
│   ├── events.py
│   └── state.py
│
├── guardrails/
│   ├── __init__.py
│   ├── input_guard.py
│   ├── output_guard.py
│   └── validators.py
│
├── logging/
│   ├── __init__.py
│   ├── logger.py
│   └── tracing.py
│
├── scripts/
│   ├── setup.sh
│   └── seed_kb.py
│
├── ui/
│   └── streamlit_app.py
│
├── tests/
│   ├── conftest.py           ← shared pytest fixtures (conversation state, mock KB, mock LLM)
│   ├── test_agents.py
│   ├── test_retrieval.py
│   ├── test_handover.py
│   ├── test_api.py
│   └── test_guardrails.py
│
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── README.md
├── ARCHITECTURE.md
├── PROMPT.md
└── docker-compose.yml

====================================================
MULTI-AGENT REQUIREMENTS
====================================================

The system MUST support:
- Triage Agent
- Technical Support Agent
- Billing Agent
- Escalation Agent

The system MUST be extensible.

Adding a new agent MUST NOT require:
- editing orchestrator logic
- adding giant if/else chains

Use:
- registry pattern
- config-driven loading
- dynamic routing
- Strict structured output parsing to ensure agents always return valid JSON/Pydantic objects.

====================================================
AGENT BASE CLASS REQUIREMENTS
====================================================

Every agent MUST extend a BaseAgent abstract class that enforces:
- A `process(state: ConversationState) -> AgentResponse` method signature
- A `can_handle(intent: str) -> bool` method
- A `name` property
- A `capabilities` list property
- Structured output validation via Pydantic before returning any response
- Built-in retry logic (max 2 retries) for LLM call failures
- Built-in fallback: if retries exhausted, return a structured error response — never raise unhandled exceptions to the orchestrator

====================================================
AGENT REQUIREMENTS
====================================================

----------------------------------------------------
1. TRIAGE AGENT
----------------------------------------------------

Responsibilities:
- classify intent
- extract entities
- detect urgency
- detect multi-intent queries
- determine routing

Must extract:
- customer_id
- cloud_provider
- plan_type
- issue_type
- urgency
- sentiment

Must support:
- single intent
- multi-intent decomposition
- fallback routing

Must produce structured outputs.

Structured output schema (enforce with Pydantic):
{
  "intents": ["billing", "technical"],   // list, supports multi-intent
  "entities": {
    "customer_id": str | None,
    "cloud_provider": str | None,
    "plan_type": str | None,
    "issue_type": str,
    "urgency": "low" | "medium" | "high" | "critical",
    "sentiment": "positive" | "neutral" | "frustrated" | "angry"
  },
  "routing_decision": {
    "primary_agent": str,
    "secondary_agents": list[str],
    "reason": str
  },
  "requires_multi_step": bool
}

----------------------------------------------------
2. TECHNICAL SUPPORT AGENT
----------------------------------------------------

Responsibilities:
- RAG retrieval
- troubleshooting
- API assistance
- onboarding help
- configuration support

Must:
- retrieve KB articles
- cite article IDs
- provide step-by-step instructions
- refuse hallucinations
- escalate when confidence low

Must support:
- AWS
- Azure
- GCP
- SSO
- API keys
- dashboards
- integrations
- webhooks
- alerting
- integrations

Structured output schema (enforce with Pydantic):
{
  "response": str,               // human-readable answer
  "citations": [
    {
      "article_id": str,
      "title": str,
      "relevance_score": float
    }
  ],
  "confidence": float,           // 0.0 - 1.0
  "escalate": bool,
  "escalation_reason": str | None,
  "suggested_next_steps": list[str]
}

----------------------------------------------------
3. BILLING AGENT
----------------------------------------------------

Responsibilities:
- plan upgrades
- plan downgrades
- invoice explanations
- refund policy references
- payment failures

Must use:
- mock customer data (handle edge cases gracefully, e.g., customer ID not found)
- mock invoices
- mock plans

Must NEVER fabricate:
- pricing
- refunds
- enterprise contracts

If info missing:
- escalate

Mock data MUST be defined as typed Python dataclasses or Pydantic models — not raw dicts.
Mock data must cover at least:
- 5 mock customer accounts (varying plans: Free, Starter, Pro, Enterprise)
- 3 mock invoices per customer
- All plan tier details (features, pricing, limits) stored in config/billing_plans.yaml

Structured output schema (enforce with Pydantic):
{
  "response": str,
  "action_taken": str | None,       // e.g., "plan_upgrade_simulated"
  "plan_details": dict | None,
  "invoice_summary": dict | None,
  "policy_citations": list[str],    // KB article IDs cited
  "escalate": bool,
  "escalation_reason": str | None
}

----------------------------------------------------
4. ESCALATION AGENT
----------------------------------------------------

Responsibilities:
- summarize conversation
- package escalation context
- classify urgency
- classify priority
- generate handoff payload

Must produce:
- structured escalation package
- human-readable summary
- sentiment analysis
- issue category
- recommended next action

Structured output schema (enforce with Pydantic):
{
  "escalation_id": str,             // UUID
  "priority": "P1" | "P2" | "P3" | "P4",
  "urgency": "critical" | "high" | "medium" | "low",
  "sentiment": str,
  "issue_category": str,
  "conversation_summary": str,
  "extracted_entities": dict,
  "recommended_action": str,
  "human_handoff_payload": {
    "customer_id": str | None,
    "full_history": list[dict],
    "notes_for_agent": str
  },
  "trace_id": str,
  "timestamp": str
}

====================================================
RAG REQUIREMENTS
====================================================

Implement a COMPLETE RAG pipeline.

Must include:
1. document ingestion
2. chunking
3. embedding
4. vector indexing
5. retrieval
6. query rewriting
7. citation generation
8. graceful degradation (what happens if vector DB is down or empty?)

Chunking strategy:
- Use recursive character text splitter with chunk_size=512 and chunk_overlap=64
- Each chunk must carry metadata: article_id, title, category, tags, chunk_index
- Do not split mid-sentence where avoidable

ChromaDB configuration:
- Use a persistent local ChromaDB instance stored at `./chroma_db/` in the project root
- Collection name: `clouddash_kb`
- On startup, check if collection exists and has documents; if empty, auto-trigger ingestion
- Expose a `reset_collection()` utility for test teardown

Retrieval configuration:
- Default top_k: 5
- Minimum similarity score threshold: 0.35 (chunks below this are discarded)
- If zero chunks pass the threshold, return an explicit empty result — do not hallucinate

====================================================
KNOWLEDGE BASE REQUIREMENTS
====================================================

Generate 20 HIGH QUALITY KB articles.

Categories:
- FAQs (4 articles)
- Troubleshooting (5 articles)
- Billing & Pricing (4 articles)
- API Documentation (3 articles)
- SSO & Account Management (2 articles)
- Onboarding (1 article)
- Integrations (1 article)

Each article MUST include:
- id
- title
- category
- tags
- content
- applies_to
- last_updated

Use realistic SaaS documentation style.

Articles should contain:
- troubleshooting steps
- API examples
- setup flows
- policy explanations

Article content length:
- Minimum 200 words per article
- Troubleshooting guides: minimum 300 words with numbered steps
- API docs: must include at least one code example block (curl or Python)
- Billing articles: must include concrete plan names, feature limits, and pricing tiers

====================================================
QUERY REWRITING
====================================================

Implement conversation-aware query rewriting.

Example:

Conversation:
User: "alerts broke yesterday"
User: "after rotating AWS credentials"

Rewrite:
"CloudDash alerts stopped firing after AWS credential rotation"

This rewritten query should be used for retrieval.

Query rewriter MUST:
- Take last N messages (configurable, default N=5) as context window
- Produce a single, standalone, semantically rich query string
- Strip filler words and pronouns
- Preserve domain-specific terms (AWS, CloudDash, SSO, webhook, etc.)
- Return the original query unchanged if no meaningful rewrite is possible (avoid over-rewriting)

====================================================
RETRIEVAL QUALITY
====================================================

Implement:
- vector retrieval
- keyword/BM25 retrieval
- hybrid retrieval
- optional reranking

If reranking implemented:
- explain methodology

Retrieved chunks MUST include:
- similarity score
- metadata
- citation info

Hybrid retrieval implementation:
- Vector score and BM25 score must be normalized to [0, 1] independently before fusion
- Use Reciprocal Rank Fusion (RRF) as the default score fusion strategy
- RRF parameter k=60 (standard default)
- Final ranked list must de-duplicate chunks from the same article (keep highest-scoring chunk per article)

====================================================
HANDOVER REQUIREMENTS
====================================================

The handover system is CRITICAL.

Implement:
- structured handover payloads
- context preservation
- entity transfer
- audit logging
- graceful fallback

A handover event MUST contain:
- timestamp
- source agent
- target agent
- reason
- conversation summary
- extracted entities
- trace ID

If handover fails:
- fallback to triage
OR
- escalate

Handover failure conditions that MUST be handled:
- Target agent not found in registry
- Target agent raises an unhandled exception during processing
- Target agent returns an invalid/unparseable response
- Handover loop detected (same agent handed over to more than once in a single conversation turn)

Audit log format (append-only, structured JSON, one event per line):
{
  "event": "handover",
  "timestamp": "ISO8601",
  "trace_id": str,
  "conversation_id": str,
  "source_agent": str,
  "target_agent": str,
  "reason": str,
  "success": bool,
  "fallback_triggered": bool,
  "context_snapshot": {
    "message_count": int,
    "entities": dict,
    "last_user_message": str
  }
}

====================================================
MEMORY + STATE MANAGEMENT
====================================================

Implement:
- conversation memory
- session state
- trace IDs
- message history
- structured state objects (Crucial for LangGraph implementations)

Every conversation MUST have:
- conversation_id
- trace_id
- timestamps

Use typed models.

ConversationState (LangGraph node-compatible TypedDict or Pydantic model) MUST include:
- conversation_id: str
- trace_id: str
- messages: list[Message]           // typed Message objects, not raw strings
- current_agent: str
- previous_agent: str | None
- extracted_entities: dict
- handover_history: list[HandoverEvent]
- escalated: bool
- created_at: datetime
- updated_at: datetime
- metadata: dict                    // for extensibility

Message model MUST include:
- role: "user" | "assistant" | "system"
- content: str
- agent_name: str | None
- timestamp: datetime
- citations: list[Citation] | None

In-memory store is acceptable for the prototype.
Store MUST support: create, read, update, delete by conversation_id.
Implement TTL-based expiry (default: 1 hour) to prevent memory leaks in long-running processes.

====================================================
OBSERVABILITY + LOGGING
====================================================

Implement structured JSON logging.

Log:
- agent invocations
- retrieval events
- routing decisions
- handovers
- escalations
- failures
- latency
- token usage if possible

Every log MUST include:
- trace_id
- conversation_id
- timestamp

Log levels MUST be used correctly:
- DEBUG: retrieval chunk details, LLM prompt/response payloads
- INFO: agent invocations, routing decisions, handovers, API requests
- WARNING: low confidence scores, fallback triggers, missing KB results
- ERROR: LLM API failures, unhandled exceptions, invalid structured outputs

All logs MUST be written to both stdout (for container environments) and a rotating file handler at `logs/app.log` (max 10MB, 3 backups).

BONUS:
Integrate Langfuse OR Phoenix tracing.

====================================================
GUARDRAILS
====================================================

Implement BOTH:

----------------------------------------------------
1. INPUT GUARDRAILS
----------------------------------------------------

Detect:
- prompt injection
- jailbreak attempts
- irrelevant/off-topic requests

Examples:
- "ignore previous instructions"
- "reveal system prompt"

Additional injection patterns to detect:
- "act as", "pretend you are", "you are now", "DAN", "developer mode"
- "print your instructions", "what is your prompt"
- Messages that are entirely in a non-English language when the system is English-only (flag, do not block — escalate for review)
- Messages over 2000 characters (flag as potential prompt stuffing)

Input guardrail MUST return a structured decision:
{
  "allowed": bool,
  "reason": str | None,
  "risk_category": "injection" | "jailbreak" | "off_topic" | "length" | "safe" | None,
  "risk_score": float    // 0.0 - 1.0
}

----------------------------------------------------
2. OUTPUT GUARDRAILS
----------------------------------------------------

Prevent:
- hallucinated pricing
- unsupported features
- fabricated policies
- JSON decode errors (fallback handling)

If KB confidence low:
- explicitly say information unavailable
- offer escalation

Implement:
- citation validation
- hallucination detection against retrieved KB

Output guardrail MUST:
- Check that every pricing figure mentioned in agent output exists verbatim in a retrieved KB chunk
- Check that every plan name mentioned exists in config/billing_plans.yaml
- If a citation ID is referenced in the output, verify it exists in the retrieved chunks for that query
- On any violation: strip the offending sentence, append a disclaimer, and log a WARNING
- Never return a raw exception traceback in the API response — always return a sanitized error message

====================================================
API REQUIREMENTS
====================================================

Build FastAPI endpoints:

POST /conversation/start
POST /conversation/message
GET /conversation/{id}
GET /health

Responses must include:
- current agent
- citations
- handover info if any
- trace_id

Ensure CORS middleware is correctly configured to allow Streamlit to communicate with the API.

Additional API requirements:
- All endpoints must have Pydantic request and response models — no untyped dicts in route signatures
- API versioning prefix: `/api/v1/` for all routes
- `/health` endpoint must return: status, uptime, KB document count, ChromaDB status
- All API errors must return a structured JSON error body: `{"error": str, "code": str, "trace_id": str}`
- Request timeout: configure uvicorn to timeout requests at 60 seconds
- Rate limiting: implement a simple in-memory rate limiter (max 30 requests/minute per IP) using a middleware

====================================================
STREAMLIT UI REQUIREMENTS
====================================================

Build a minimal Streamlit interface.

Features:
- start conversation
- send messages
- show citations
- show active agent
- show escalation events
- display logs/debug info optionally
- cleanly handle API timeouts or connection errors

Keep UI simple.
Focus on backend quality.

Additional UI requirements:
- Conversation ID and Trace ID must be visible in the sidebar
- Agent transitions must be visually indicated (e.g., colored badge showing current agent name)
- Citations must be expandable (st.expander) showing article ID, title, and relevance score
- If the API is unreachable, show a clear error banner — do not crash silently
- Include a "Reset Conversation" button that starts a new session
- API base URL must be configurable via an environment variable (STREAMLIT_API_URL) with a localhost default

====================================================
CONFIG REQUIREMENTS
====================================================

ALL prompts and routing rules MUST be configurable via YAML.

DO NOT hardcode:
- system prompts
- routing rules
- agent capabilities

Use:
config/agents.yaml
config/routing.yaml

config/agents.yaml MUST contain for each agent:
- name
- description
- system_prompt
- capabilities (list of intent strings this agent handles)
- max_retries
- confidence_threshold
- escalate_on_low_confidence (bool)

config/routing.yaml MUST contain:
- intent_to_agent mapping (dict)
- fallback_agent (default: triage)
- multi_intent_strategy: "sequential" | "parallel" (implement sequential for prototype)
- handover_timeout_seconds

config/billing_plans.yaml MUST contain:
- All plan tiers (Free, Starter, Pro, Enterprise)
- Features per plan
- Pricing per plan
- Upgrade/downgrade rules

====================================================
TESTING REQUIREMENTS
====================================================

Implement meaningful tests.

Must include:
- routing tests
- retrieval tests
- handover tests
- API tests
- guardrail tests

Tests should validate:
- correct agent selection
- KB retrieval quality
- citation inclusion
- context preservation
- escalation behavior
- asynchronous endpoints (using pytest-asyncio)

Additional testing requirements:
- tests/conftest.py MUST provide shared fixtures for:
  - a pre-seeded in-memory ChromaDB collection
  - a mock LLM that returns canned structured responses (to avoid real API calls in tests)
  - a sample ConversationState object
  - a running TestClient for FastAPI
- Every test MUST be isolated — no test should depend on state from another test
- Use `pytest.mark.asyncio` for all async tests
- Test the four assessment scenarios explicitly:
  - test_scenario_single_agent_resolution
  - test_scenario_cross_agent_handover
  - test_scenario_escalation_to_human
  - test_scenario_kb_retrieval_failure
- Mock all external LLM API calls in tests using pytest-mock or unittest.mock

====================================================
MAKEFILE REQUIREMENTS
====================================================

The Makefile MUST include these targets:
- `make install`       — install dependencies from requirements.txt
- `make dev`           — start FastAPI dev server with hot reload
- `make ui`            — start Streamlit UI
- `make seed`          — run scripts/seed_kb.py to ingest KB articles
- `make test`          — run pytest with coverage report
- `make lint`          — run ruff or flake8
- `make reset-db`      — delete and reinitialize ChromaDB collection
- `make clean`         — remove __pycache__, .pyc files, logs

====================================================
README REQUIREMENTS
====================================================

README must include:
- setup instructions
- architecture overview
- screenshots
- API examples
- sample conversations
- design decisions
- tradeoffs
- limitations
- future improvements

README MUST also include:
- A "Quick Start in 3 Commands" section at the very top (install, seed, run)
- Environment variable reference table (variable name, description, required/optional, default)
- How to add a new agent (step-by-step, must demonstrate config-driven extensibility)
- How to add new KB articles
- Known issues and workarounds

====================================================
ARCHITECTURE DOCUMENT
====================================================

Create ARCHITECTURE.md containing:
- orchestration flow
- RAG flow
- handover protocol
- extensibility strategy
- state management
- scaling considerations
- production evolution ideas

ARCHITECTURE.md MUST also include:
- An ASCII or Mermaid diagram of the agent orchestration flow
- An ASCII or Mermaid diagram of the RAG pipeline
- An ASCII or Mermaid diagram of the handover state machine
- A section on "What is NOT production-ready" (honest scope limitations of the prototype)

====================================================
PRODUCTION THINKING
====================================================

The code should reflect production-style engineering.

Implement:
- dependency injection where useful
- retries
- graceful error handling (especially for LLM API calls)
- modular services
- reusable utilities
- typed models everywhere

NO:
- giant files
- spaghetti logic
- unstructured globals
- hardcoded secrets
- bare except blocks

====================================================
IMPORTANT IMPLEMENTATION STYLE
====================================================

During implementation:
- explain decisions
- explain tradeoffs
- keep code readable
- prioritize maintainability
- prefer explicitness over cleverness

When uncertain:
- choose robustness
- choose modularity
- choose debuggability

====================================================
PHASE EXECUTION PLAN
====================================================

You MUST execute in these phases:

PHASE 1
- Project scaffolding (all directories + __init__.py files)
- requirements.txt and requirements-dev.txt with pinned versions
- .env.example with all required environment variables
- Config system: config/settings.py, config/agents.yaml, config/routing.yaml, config/billing_plans.yaml
- Base Pydantic models: models/conversation.py, models/state.py, models/responses.py, models/events.py
- Logging system: logging/logger.py, logging/tracing.py
- Makefile

VALIDATE:
- `python -c "from models.state import ConversationState"` runs without error
- `python -c "from app_logging.logger import get_logger"` runs without error
- `python -c "from config.settings import settings"` runs without error
- All YAML config files load without parse errors
- `pip install -r requirements.txt` completes without conflicts
--> USER CHECKPOINT: Wait for user approval before moving to Phase 2.

----------------------------------------------------

PHASE 2
- Knowledge base generation (20 articles in knowledge_base/data/articles/)
- scripts/seed_kb.py ingestion script
- retrieval/chunking.py
- retrieval/embeddings.py
- retrieval/vectorstore.py
- retrieval/ingest.py
- retrieval/retriever.py
- retrieval/query_rewriter.py
- retrieval/reranker.py

VALIDATE:
- `make seed` completes without error
- `python -c "from retrieval.retriever import Retriever; r = Retriever(); results = r.retrieve('AWS alert configuration'); assert len(results) > 0"` passes
- Citations include article_id, title, relevance_score
- Query rewriter test: feed 3-message history, verify standalone query returned
- ChromaDB collection shows correct document count after seeding
--> USER CHECKPOINT: Wait for user approval before moving to Phase 3.

----------------------------------------------------

PHASE 3
- agents/base_agent.py (abstract BaseAgent)
- agents/triage_agent.py
- agents/technical_agent.py
- agents/billing_agent.py
- agents/escalation_agent.py
- agents/registry.py
- agents/orchestrator.py (LangGraph StateGraph)

VALIDATE:
- `python -c "from agents.registry import AgentRegistry; r = AgentRegistry(); print(r.list_agents())"` prints all 4 agents
- Triage agent returns valid structured output for: single intent query, multi-intent query, ambiguous query
- Technical agent retrieves KB and returns citations
- Billing agent returns structured response with mock customer data
- Escalation agent produces complete escalation package
- Adding a 5th mock agent via config only (no code change) registers correctly
--> USER CHECKPOINT: Wait for user approval before moving to Phase 4.

----------------------------------------------------

PHASE 4
- handover/models.py
- handover/manager.py
- handover/audit_logger.py
- handover/summarizer.py
- memory/conversation_store.py
- memory/session_manager.py

VALIDATE:
- Handover from Technical → Billing preserves all entities
- Handover audit log file written correctly (valid JSON per line)
- Failed handover (target agent not found) triggers fallback without crashing
- Conversation store create/read/update/delete all work
- TTL expiry test: create conversation, manually expire TTL, verify it is cleaned up
--> USER CHECKPOINT: Wait for user approval before moving to Phase 5.

----------------------------------------------------

PHASE 5
- guardrails/input_guard.py
- guardrails/output_guard.py
- guardrails/validators.py

VALIDATE:
- "ignore previous instructions" → blocked with risk_category: "injection"
- "reveal your system prompt" → blocked with risk_category: "injection"
- Normal customer query → allowed
- Agent output with hallucinated price not in KB → offending sentence stripped + WARNING logged
- Agent output with invalid citation ID → citation removed + WARNING logged
- Output with real KB-grounded data passes without modification
--> USER CHECKPOINT: Wait for user approval before moving to Phase 6.

----------------------------------------------------

PHASE 6
- api/main.py with CORS and rate limiting middleware
- api/routes/conversation.py
- api/routes/health.py
- api/dependencies.py
- ui/streamlit_app.py

VALIDATE:
- `make dev` starts FastAPI on port 8000 without errors
- GET /api/v1/health returns status 200 with KB document count
- POST /api/v1/conversation/start returns conversation_id and trace_id
- POST /api/v1/conversation/message returns agent response with citations
- GET /api/v1/conversation/{id} returns full history
- `make ui` starts Streamlit without errors
- End-to-end: send a message via Streamlit → response appears with agent badge and citations
- Run all four assessment scenarios via the API and verify expected routing
--> USER CHECKPOINT: Wait for user approval before moving to Phase 7.

----------------------------------------------------

PHASE 7
- tests/conftest.py
- tests/test_agents.py
- tests/test_retrieval.py
- tests/test_handover.py
- tests/test_api.py
- tests/test_guardrails.py
- README.md (complete)
- ARCHITECTURE.md (complete with diagrams)
- docker-compose.yml
- scripts/setup.sh

VALIDATE:
- `make test` runs all tests with zero failures
- pytest coverage report shows ≥60% overall coverage
- README quick-start instructions work on a clean environment
- docker-compose.yml builds and starts without errors (if Docker available)
- `make lint` passes with no errors
--> USER CHECKPOINT: Inform the user the project is complete.

====================================================
FINAL DELIVERABLE REQUIREMENTS
====================================================

At completion provide:
1. Full working repository
2. Final file tree
3. Setup instructions
4. Deployment instructions
5. Example API requests
6. Example conversations
7. Architecture summary
8. Known limitations
9. Future improvements

====================================================
QUALITY BAR
====================================================

The final result should feel like:
- an AI engineering internship project
- built with production awareness
- modular and extensible
- demo-ready for technical interviews

Prioritize:
- correctness
- architecture
- maintainability
- realism

DO NOT prioritize:
- flashy frontend
- unnecessary complexity
- overengineering

Begin with PHASE 1 only.
Do not jump ahead.
Validate after completion before proceeding.
Wait for my explicit instruction to move to the next phase.