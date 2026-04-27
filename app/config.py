"""
app/config.py — 12-factor config via pydantic-settings.

All settings are read from environment variables (or a .env file).
No secrets live in code. Swap LLM providers by changing OPENAI_BASE_URL +
OPENAI_API_KEY alone — the rest of the codebase stays vendor-neutral.

Interview note:
  "How do you keep an LLM app vendor-neutral?"
  → Config + adapter pattern: the client in llm.py reads base_url/api_key from
    Settings and constructs openai.OpenAI(...). Swapping providers = env-var change.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedProvider(str, Enum):
    openai = "openai"
    local = "local"


class LogLevel(str, Enum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible base URL (works with any vendor).",
    )
    openai_api_key: str = Field(
        default="",
        description="API key for the LLM provider.",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="Chat model name.")
    llm_max_tokens: int = Field(default=2048, ge=1)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_provider: EmbedProvider = Field(
        default=EmbedProvider.openai,
        description="'openai' uses the API; 'local' uses sentence-transformers.",
    )
    embed_model: str = Field(
        default="text-embedding-3-small",
        description="Model name for OpenAI-compat embeddings endpoint.",
    )
    embed_local_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model id when embed_provider='local'.",
    )

    # ── Vector DB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))

    # ── SQLite ────────────────────────────────────────────────────────────────
    db_url: str = Field(default="sqlite:///./data/db.sqlite")

    # ── Langfuse ──────────────────────────────────────────────────────────────
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # ── App ───────────────────────────────────────────────────────────────────
    log_level: LogLevel = Field(default=LogLevel.info)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


# Singleton — import and use `settings` everywhere.
settings = Settings()
