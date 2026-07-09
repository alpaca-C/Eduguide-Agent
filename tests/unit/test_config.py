# Unit tests for Configuration

from __future__ import annotations

import os
import pytest

# Force a clean import — tests set env vars before instantiation
from src.config import Configuration


class TestConfiguration:
    """Tests for Configuration.from_env() and field defaults."""

    def test_default_values(self, monkeypatch):
        """All fields should have safe defaults when no env vars are set."""
        # Remove relevant env vars
        for key in ["LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL",
                     "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "CHUNK_SIZE",
                     "CHUNK_OVERLAP", "EXTRACT_BATCH_SIZE"]:
            monkeypatch.delenv(key, raising=False)

        config = Configuration.from_env()
        assert config.llm_model_id == "deepseek-chat"
        assert config.llm_api_key == "sk-placeholder"
        assert config.llm_base_url == "https://api.deepseek.com"
        assert config.llm_temperature == 0.0
        assert config.llm_max_tokens == 6000
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150

    def test_override_from_env(self, monkeypatch):
        """Env vars should override defaults for fields supported by from_env()."""
        monkeypatch.setenv("LLM_MODEL_ID", "gpt-4")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
        monkeypatch.setenv("EXTRACT_CONCURRENCY", "5")

        config = Configuration.from_env()
        assert config.llm_model_id == "gpt-4"
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 4096
        assert config.extract_concurrency == 5

    def test_override_kwargs(self, monkeypatch):
        """Explicit kwargs should take precedence over env vars."""
        monkeypatch.setenv("LLM_MODEL_ID", "from-env")
        config = Configuration.from_env(llm_model_id="from-kwarg")
        assert config.llm_model_id == "from-kwarg"

    def test_concurrency_defaults(self, monkeypatch):
        """Concurrency fields should have reasonable defaults."""
        for key in ["EXTRACT_CONCURRENCY", "CHAPTER_DETECT_CONCURRENCY"]:
            monkeypatch.delenv(key, raising=False)

        config = Configuration.from_env()
        assert config.extract_concurrency == 3
        assert config.chapter_detect_concurrency == 3

    def test_concurrency_from_env(self, monkeypatch):
        """Concurrency fields should be settable from env."""
        monkeypatch.setenv("EXTRACT_CONCURRENCY", "5")
        monkeypatch.setenv("CHAPTER_DETECT_CONCURRENCY", "2")

        config = Configuration.from_env()
        assert config.extract_concurrency == 5
        assert config.chapter_detect_concurrency == 2

    def test_int_parsing(self, monkeypatch):
        """String env values should be correctly parsed as int/float."""
        monkeypatch.setenv("LLM_MAX_TOKENS", "8000")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.3")

        config = Configuration.from_env()
        assert isinstance(config.llm_max_tokens, int)
        assert isinstance(config.llm_temperature, float)
        assert config.llm_max_tokens == 8000
        assert config.llm_temperature == 0.3
