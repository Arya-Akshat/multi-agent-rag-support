"""
config/settings.py — Application-wide settings loaded from environment variables.

Uses pydantic-settings so every field is type-validated at startup.
A single `settings` singleton is imported throughout the codebase so
configuration is centralised and never scattered across modules.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object.

    All fields have sensible defaults so the app can boot without a .env
    file (useful for Phase 1 import-only validation). Fields that genuinely
    require a secret (GROQ_API_KEY) default to an empty string so the
    import succeeds; actual usage will fail with a clear error message if
    the key is missing.
    """

    model_config = SettingsConfigDict(
        # Load from .env if present; do NOT crash if it's missing
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ------------------------------------------------------------------ #
    # LLM (Groq)
    # ------------------------------------------------------------------ #
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(default="llama-3.1-8b-instant", description="Groq LLM model name")

    # ------------------------------------------------------------------ #
    # Vector Store / ChromaDB
    # ------------------------------------------------------------------ #
    chroma_db_path: str = Field(default="./chroma_db", description="ChromaDB persistence path")
    chroma_collection_name: str = Field(
        default="clouddash_kb", description="ChromaDB collection name"
    )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    top_k_retrieval: int = Field(default=5, description="Number of chunks to retrieve")
    min_similarity_score: float = Field(
        default=0.35, description="Minimum cosine similarity threshold"
    )
    query_rewriter_window: int = Field(
        default=5, description="Number of recent messages fed to query rewriter"
    )

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/app.log", description="Rotating log file path")

    # ------------------------------------------------------------------ #
    # API Server
    # ------------------------------------------------------------------ #
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    request_timeout: int = Field(default=60, description="Request timeout in seconds")

    # ------------------------------------------------------------------ #
    # Rate Limiting
    # ------------------------------------------------------------------ #
    rate_limit_rpm: int = Field(default=30, description="Max requests per minute per IP")

    # ------------------------------------------------------------------ #
    # Session / Memory
    # ------------------------------------------------------------------ #
    session_ttl_seconds: int = Field(
        default=3600, description="Conversation session TTL in seconds"
    )

    # ------------------------------------------------------------------ #
    # Streamlit UI
    # ------------------------------------------------------------------ #
    streamlit_api_url: str = Field(
        default="http://localhost:8000", description="API base URL for Streamlit"
    )

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    environment: str = Field(default="development", description="Runtime environment")

    # ------------------------------------------------------------------ #
    # Derived helpers (not env vars)
    # ------------------------------------------------------------------ #
    @property
    def is_production(self) -> bool:
        """Return True when running in a production environment."""
        return self.environment.lower() == "production"

    @property
    def chroma_db_resolved_path(self) -> Path:
        """Resolve the ChromaDB path relative to the current working directory."""
        return Path(self.chroma_db_path).resolve()

    @property
    def log_file_path(self) -> Path:
        """Resolve the log file path and ensure its parent directory exists."""
        p = Path(self.log_file).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere:
#   from config.settings import settings
# ---------------------------------------------------------------------------
settings = Settings()
