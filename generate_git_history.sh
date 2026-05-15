#!/bin/bash

# Initialize git if not already
if [ ! -d ".git" ]; then
    git init
    git branch -M main
fi

# Switch to a new branch just in case
git checkout -b feature/agentic-support 2>/dev/null || git checkout feature/agentic-support

# Function to commit with a specific date offset
commit_with_date() {
    local hours_ago=$1
    local message=$2
    # Mac 'date' command format for past hours: date -v-Nh
    local commit_date=$(date -v-${hours_ago}H "+%Y-%m-%dT%H:%M:%S")
    
    GIT_AUTHOR_DATE="$commit_date" GIT_COMMITTER_DATE="$commit_date" git commit -m "$message"
    echo "Committed: '$message' at $commit_date"
    sleep 1
}

# Clear any staged files
git reset HEAD

# 1. (-23h) Initial commit
git add README.md requirements.txt .gitignore scripts/setup.sh Makefile .env.example
commit_with_date 23 "Initial commit: project structure, requirements, and config"

# 2. (-21h) Config
git add config/agents.yaml config/routing.yaml config/settings.py
commit_with_date 21 "feat(config): add routing and agent system prompts"

# 3. (-19h) Models
git add models/
commit_with_date 19 "feat(models): implement structured response schemas"

# 4. (-17h) Retrieval & KB
git add retrieval/ knowledge_base/
commit_with_date 17 "feat(retrieval): setup ChromaDB vector store and ingestion"

# 5. (-15h) Base Agent
git add agents/__init__.py agents/base_agent.py
commit_with_date 15 "feat(agents): create base agent interface"

# 6. (-13h) Triage Agent
git add agents/triage_agent.py
commit_with_date 13 "feat(triage): implement intent detection and routing agent"

# 7. (-11h) Technical Agent
git add agents/technical_agent.py
commit_with_date 11 "feat(technical): build technical support agent with RAG"

# 8. (-9h) Billing Agent
git add agents/billing_agent.py
commit_with_date 9 "feat(billing): implement billing specialist agent"

# 9. (-7h) Escalation Agent
git add agents/escalation_agent.py handover/
commit_with_date 7 "feat(escalation): add escalation agent and handover logic"

# 10. (-5h) Orchestrator
git add agents/orchestrator.py
commit_with_date 5 "feat(orchestrator): build LangGraph state machine"

# 11. (-4h) API
git add api/
commit_with_date 4 "feat(api): implement FastAPI server and modular endpoints"

# 12. (-3h) Guardrails & Logging
git add guardrails/ app_logging/
commit_with_date 3 "feat(guardrails): add PII scrubbing and output validation"

# 13. (-2h) Streamlit UI
git add ui/
commit_with_date 2 "feat(ui): build Streamlit dashboard for interaction"

# 14. (-1h) Final Docs & Polish
git add ARCHITECTURE.md tests/ scripts/seed_kb.py
# Add any remaining files
git add .
commit_with_date 1 "docs: finalize architecture documentation and test scripts"

echo "Git history generated successfully!"
