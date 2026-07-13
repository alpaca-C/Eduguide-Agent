# Unit tests for BaseAgent (abstract base with LLM factory and retry)

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import (
    BaseAgent, AgentInput, AgentOutput, DEFAULT_LLM_MAX_RETRIES,
)
from src.config import Configuration


# ── Concrete agent for testing abstract class ──────────────────────────

class _TestAgent(BaseAgent):
    """Minimal concrete agent to exercise BaseAgent methods."""

    async def run(self, input: AgentInput) -> AgentOutput:
        return AgentOutput(success=True)


# ── Fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """Create a test agent with a mocked Configuration."""
    config = MagicMock(spec=Configuration)
    config.llm_model_id = "deepseek-chat"
    config.llm_api_key = "sk-test"
    config.llm_base_url = "https://api.test.com"
    config.llm_temperature = 0.0
    config.llm_max_tokens = 6000
    return _TestAgent(config)


# ========================================================================
# AgentInput / AgentOutput
# ========================================================================

class TestAgentInput:
    """Tests for AgentInput dataclass."""

    def test_default_metadata(self):
        inp = AgentInput()
        assert inp.metadata == {}

    def test_custom_metadata(self):
        inp = AgentInput(metadata={"question": "什么是AI"})
        assert inp.metadata["question"] == "什么是AI"


class TestAgentOutput:
    """Tests for AgentOutput dataclass."""

    def test_defaults(self):
        out = AgentOutput()
        assert out.success is True
        assert out.error == ""
        assert out.metadata == {}

    def test_failure_output(self):
        out = AgentOutput(success=False, error="something went wrong")
        assert out.success is False
        assert out.error == "something went wrong"


# ========================================================================
# _make_llm
# ========================================================================

class TestMakeLLM:
    """Tests for BaseAgent._make_llm() factory method."""

    def test_creates_chat_openai_with_defaults(self, agent):
        """Should create ChatOpenAI using config values."""
        with patch("src.agents.base.ChatOpenAI") as mock_cls:
            agent._make_llm()
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["model"] == agent._config.llm_model_id
            assert call_kwargs["api_key"] == agent._config.llm_api_key
            assert call_kwargs["base_url"] == agent._config.llm_base_url

    def test_override_model(self, agent):
        """Explicit model should override config."""
        with patch("src.agents.base.ChatOpenAI") as mock_cls:
            agent._make_llm(model="gpt-4")
            assert mock_cls.call_args.kwargs["model"] == "gpt-4"

    def test_override_temperature(self, agent):
        """Explicit temperature should override config."""
        with patch("src.agents.base.ChatOpenAI") as mock_cls:
            agent._make_llm(temperature=0.9)
            assert mock_cls.call_args.kwargs["temperature"] == 0.9

    def test_override_max_tokens(self, agent):
        """Explicit max_tokens should override config."""
        with patch("src.agents.base.ChatOpenAI") as mock_cls:
            agent._make_llm(max_tokens=8192)
            assert mock_cls.call_args.kwargs["max_tokens"] == 8192

    def test_override_timeout(self, agent):
        """Explicit timeout should be passed through."""
        with patch("src.agents.base.ChatOpenAI") as mock_cls:
            agent._make_llm(timeout=60)
            assert mock_cls.call_args.kwargs["timeout"] == 60

    def test_default_max_retries(self, agent):
        """Default max_retries should be 1 (not LLM retry, SDK retry)."""
        with patch("src.agents.base.ChatOpenAI") as mock_cls:
            agent._make_llm()
            assert mock_cls.call_args.kwargs["max_retries"] == 1


# ========================================================================
# _llm_retry
# ========================================================================

class TestLLMRetry:
    """Tests for BaseAgent._llm_retry() with retry logic."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, agent):
        """Should return result on first successful call."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "success"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await agent._llm_retry(
            [{"role": "user", "content": "hi"}],
            llm=mock_llm,
        )

        assert result.content == "success"
        assert mock_llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure_then_succeed(self, agent):
        """Should retry after transient failure and succeed."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "eventual success"
        mock_llm.ainvoke = AsyncMock(side_effect=[
            Exception("transient error"),
            mock_response,
        ])

        result = await agent._llm_retry(
            [{"role": "user", "content": "hi"}],
            llm=mock_llm,
            max_retries=2,
        )

        assert result.content == "eventual success"
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self, agent):
        """Should raise the last error after all retries fail."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("always fails"))

        with pytest.raises(Exception, match="always fails"):
            await agent._llm_retry(
                [{"role": "user", "content": "hi"}],
                llm=mock_llm,
                max_retries=2,
            )

        # 1 try + 2 retries = 3 attempts
        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_uses_default_llm_when_none_provided(self, agent):
        """Should auto-create an LLM when llm parameter is None."""
        with patch("src.agents.base.ChatOpenAI") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(
                return_value=MagicMock(content="auto")
            )
            mock_cls.return_value = mock_instance

            result = await agent._llm_retry(
                [{"role": "user", "content": "hi"}],
                llm=None,
                max_retries=0,
            )

            assert result.content == "auto"

    @pytest.mark.asyncio
    async def test_zero_retries_calls_once(self, agent):
        """With max_retries=0, should call exactly once."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="ok")
        )

        await agent._llm_retry(
            [{"role": "user", "content": "hi"}],
            llm=mock_llm,
            max_retries=0,
        )

        assert mock_llm.ainvoke.call_count == 1
