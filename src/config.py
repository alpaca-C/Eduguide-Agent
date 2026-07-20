# Configuration — Document QA System
#
# Based on pydantic-settings BaseSettings:
#   - Auto-reads from .env file
#   - Auto-converts types (str → int, str → float, etc.)
#   - Field defaults serve as fallback values
#
# Usage:
#   config = Configuration()                 # reads .env + os.environ
#   config = Configuration(llm_model_id="x") # kwargs override env
#   config = Configuration.from_env(...)     # backward-compat alias

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuration(BaseSettings):
    """Document QA System configuration — auto-loaded from .env + environment."""

    # NOTE: env_file is deliberately NOT set here. The project entry point
    # (main.py) calls load_dotenv() which loads .env into os.environ.
    # BaseSettings reads os.environ by default — this keeps .env loading
    # explicit and test-friendly (tests control env vars, not the file).
    model_config = SettingsConfigDict(
        extra="ignore",  # ignore unknown env vars instead of erroring
    )

    # ── LLM ──────────────────────────────────────────────────────────
    llm_model_id: str = Field(default="deepseek-chat")
    llm_api_key: str = Field(default="sk-placeholder")
    llm_base_url: str = Field(default="https://api.deepseek.com")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=6000)

    # Chapter detection models (fall back to llm_model_id when empty)
    llm_chapter_detect_model: str = Field(default="")
    llm_chapter_review_model: str = Field(default="")
    chapter_detect_timeout: int = Field(default=60)

    # Vision model for image-based PDF TOC detection
    llm_vision_model: str = Field(default="")
    llm_vision_base_url: str = Field(default="")
    llm_vision_api_key: str = Field(default="")

    # ── Document processing ──────────────────────────────────────────
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)
    extract_batch_size: int = Field(default=25)
    extract_chunk_max_chars: int = Field(default=300)
    extract_max_concepts_per_batch: int = Field(default=30)

    # ── Concurrency ──────────────────────────────────────────────────
    extract_concurrency: int = Field(default=3)
    chapter_detect_concurrency: int = Field(default=3)

    # ── Storage ──────────────────────────────────────────────────────
    memory_db_path: str = Field(default="")

    # ── Context Builder (GSSC pipeline) ──────────────────────────────
    context_token_budget: int = Field(default=3000)
    context_hard_limit: int = Field(default=4000)
    context_relevance_weight: float = Field(default=0.6)
    context_recency_weight: float = Field(default=0.4)
    context_min_score: float = Field(default=0.10)

    # ── Monitoring ───────────────────────────────────────────────────
    monitoring_enabled: bool = Field(default=True)

    @classmethod
    def from_env(cls, **overrides) -> "Configuration":
        """
        Backward-compatible constructor.

        Previously this method manually parsed os.environ; now it delegates
        to pydantic-settings BaseSettings which auto-reads .env + os.environ
        and auto-converts types.
        """
        return cls(**overrides)
