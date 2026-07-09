# Base Agent — unified async interface for all agents

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI

from ..config import Configuration

logger = logging.getLogger(__name__)

# Default max retries for LLM calls with exponential backoff
DEFAULT_LLM_MAX_RETRIES = 2


@dataclass
class AgentInput:
    """Generic input container for agent execution.

    Concrete agents should subclass this with their specific fields.
    """
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Generic output container for agent execution.

    Concrete agents should subclass this with their specific fields.
    """
    success: bool = True
    error: str = ""
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base for all agents in the system.

    Provides:
    - Unified async run() interface
    - Shared LLM factory (ainvoke-first pattern)
    - Built-in retry with exponential backoff for transient failures

    Subclasses must implement:
    - async run(input) -> AgentOutput
    """

    def __init__(self, config: Configuration):
        self._config = config

    # ── Abstract interface ──────────────────────────────────────────────

    @abstractmethod
    async def run(self, input: AgentInput) -> AgentOutput:
        """Execute the agent's core logic. Must be implemented by subclasses."""
        ...

    # ── Shared LLM factory ──────────────────────────────────────────────

    def _make_llm(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        max_retries: int = 1,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance from config with optional overrides.

        All agents should use this factory to ensure consistent configuration
        and to avoid scattering LLM instantiation across the codebase.
        """
        return ChatOpenAI(
            model=model or self._config.llm_model_id,
            api_key=self._config.llm_api_key,
            base_url=self._config.llm_base_url,
            temperature=temperature if temperature is not None else self._config.llm_temperature,
            max_tokens=max_tokens or self._config.llm_max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ── Shared retry helper ─────────────────────────────────────────────

    async def _llm_retry(
        self,
        messages: list,
        llm: ChatOpenAI | None = None,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
    ) -> Any:
        """Call LLM with automatic retry on transient errors.

        Uses ainvoke (async) to avoid blocking the event loop.
        All agents should use this instead of hand-rolling retry loops.
        """
        llm = llm or self._make_llm()
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await llm.ainvoke(messages)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = 1.0 * (attempt + 1)
                    logger.warning(
                        "LLM call failed (attempt %d/%d, retrying in %.1fs): %s",
                        attempt + 1, max_retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]
