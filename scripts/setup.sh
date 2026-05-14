#!/bin/bash
# scripts/setup.sh — Environment setup script

echo "Setting up CloudDash development environment..."

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment created."
fi

# 2. Install dependencies
source .venv/bin/activate
pip install -r requirements.txt
echo "Dependencies installed."

# 3. Setup .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env created from .env.example. Please add your GROQ_API_KEY."
fi

# 4. Create necessary directories
mkdir -p logs chroma_db knowledge_base/data/articles
echo "Directories initialized."

# 5. Ingest Knowledge Base
python scripts/seed_kb.py
echo "Knowledge base ingested."

echo "Setup complete! Run 'python api/main.py' to start the server."
