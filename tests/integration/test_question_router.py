# Integration tests for QuestionRouter
#
# Tests difficulty classification and decomposition with mocked LLM.

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from tests.integration.conftest import (
    MockLLMResponse,
    ROUTER_TRIVIAL_JSON,
    ROUTER_MODERATE_JSON,
    ROUTER_COMPLEX_JSON,
)


def _patch_router(router, response_text: str):
    """Replace _llm_retry with a mock that returns `response_text`."""
    async def _fake_retry(messages, llm=None, max_retries=2):
        return MockLLMResponse(response_text)
    router._llm_retry = _fake_retry


class TestQuestionRouterIntegration:
    """Integration tests for QuestionRouter → difficulty classification pipeline."""

    @pytest.mark.asyncio
    async def test_classify_trivial_question(self, mock_config):
        """Greetings should be classified as trivial."""
        from src.agents.qa.router import QuestionRouter
        from src.agents.base import AgentInput

        router = QuestionRouter(mock_config)
        _patch_router(router, ROUTER_TRIVIAL_JSON)

        output = await router.run(
            AgentInput(metadata={"question": "你好", "chat_history": ""})
        )

        assert output.success is True
        assert output.metadata["difficulty"] == "trivial"
        assert output.metadata["target_docs"] == []

    @pytest.mark.asyncio
    async def test_classify_moderate_question(self, mock_config):
        """Single-concept lookup should be classified as moderate."""
        from src.agents.qa.router import QuestionRouter
        from src.agents.base import AgentInput

        router = QuestionRouter(mock_config)
        _patch_router(router, ROUTER_MODERATE_JSON)

        output = await router.run(
            AgentInput(metadata={"question": "什么是库仑定律？", "chat_history": ""})
        )

        assert output.success is True
        assert output.metadata["difficulty"] == "moderate"
        assert "库仑定律" in str(output.metadata.get("decomposition", []))

    @pytest.mark.asyncio
    async def test_classify_complex_question(self, mock_config):
        """Multi-document cross-comparison should be classified as complex."""
        from src.agents.qa.router import QuestionRouter
        from src.agents.base import AgentInput

        router = QuestionRouter(mock_config)
        _patch_router(router, ROUTER_COMPLEX_JSON)

        output = await router.run(
            AgentInput(metadata={
                "question": "麦克斯韦方程组和薛定谔方程有什么联系？",
                "chat_history": "",
            })
        )

        assert output.success is True
        assert output.metadata["difficulty"] == "complex"
        assert len(output.metadata.get("decomposition", [])) >= 2
        assert len(output.metadata.get("target_docs", [])) >= 1

    @pytest.mark.asyncio
    async def test_router_failure_defaults_moderate(self, mock_config):
        """When _llm_retry raises an exception, router should default to moderate (graceful)."""
        from src.agents.qa.router import QuestionRouter
        from src.agents.base import AgentInput

        router = QuestionRouter(mock_config)

        async def _failing_retry(messages, llm=None, max_retries=2):
            raise RuntimeError("Simulated LLM connection failure")
        router._llm_retry = _failing_retry

        output = await router.run(
            AgentInput(metadata={"question": "随便什么问题", "chat_history": ""})
        )

        assert output.success is True
        assert output.metadata["difficulty"] == "moderate"  # safe default
        assert "failed" in str(output.metadata.get("reason", "")).lower()
