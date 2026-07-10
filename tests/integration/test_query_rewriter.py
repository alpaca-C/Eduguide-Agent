# Integration tests for QueryRewriter
#
# Tests the full rewrite pipeline with mocked LLM.

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from tests.integration.conftest import MockLLMResponse, REWRITER_OUTPUT


def _patch_rewriter(rewriter, response_text: str):
    """Replace _llm_retry with a mock that returns `response_text`."""
    async def _fake_retry(messages, llm=None, max_retries=2):
        return MockLLMResponse(response_text)
    rewriter._llm_retry = _fake_retry


class TestQueryRewriterIntegration:
    """Integration tests for QueryRewriter → keyword extraction pipeline."""

    @pytest.mark.asyncio
    async def test_rewrite_returns_keyword_list(self, mock_config):
        """A natural language question should be rewritten into multiple keywords."""
        from src.agents.qa.query_rewriter import QueryRewriter

        rewriter = QueryRewriter(mock_config)
        _patch_rewriter(rewriter, REWRITER_OUTPUT)

        result = await rewriter.rewrite("什么是库仑定律？")

        assert isinstance(result, list)
        assert len(result) >= 3, f"Expected >=3 keywords, got {len(result)}: {result}"
        assert "库仑定律" in result
        # Should include English term
        assert any("Coulomb" in kw for kw in result)

    @pytest.mark.asyncio
    async def test_rewrite_graceful_fallback(self, mock_config):
        """When LLM returns empty output, should fall back to original question."""
        from src.agents.qa.query_rewriter import QueryRewriter

        rewriter = QueryRewriter(mock_config)
        _patch_rewriter(rewriter, "")  # empty LLM output

        result = await rewriter.rewrite("复杂问题不易解析")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "复杂问题不易解析"  # fallback

    @pytest.mark.asyncio
    async def test_run_method_returns_agent_output(self, mock_config):
        """The BaseAgent run() interface should return AgentOutput with metadata."""
        from src.agents.qa.query_rewriter import QueryRewriter
        from src.agents.base import AgentInput

        rewriter = QueryRewriter(mock_config)
        _patch_rewriter(rewriter, REWRITER_OUTPUT)

        output = await rewriter.run(AgentInput(metadata={"question": "麦克斯韦方程组是什么？"}))

        assert output.success is True
        assert "queries" in output.metadata
        assert len(output.metadata["queries"]) >= 3
