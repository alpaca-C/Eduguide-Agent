# Unit tests for Reflector (answer quality reviewer)

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.qa.reflector import Reflector
from src.agents.base import AgentInput
from src.config import Configuration


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Create a mocked Configuration."""
    config = MagicMock(spec=Configuration)
    config.llm_model_id = "deepseek-chat"
    config.llm_api_key = "sk-test"
    config.llm_base_url = "https://api.test.com"
    config.llm_temperature = 0.0
    config.llm_max_tokens = 6000
    return config


@pytest.fixture
def reflector(config):
    """Create a Reflector with mocked LLM."""
    agent = Reflector(config)
    # Replace the internal LLM with a mock
    agent._llm = MagicMock()
    return agent


# ── Helper ─────────────────────────────────────────────────────────────

def _make_llm_response(json_data: dict) -> MagicMock:
    """Create a mock LLM response containing JSON text."""
    resp = MagicMock()
    resp.content = json.dumps(json_data, ensure_ascii=False)
    return resp


# ========================================================================
# _parse_json
# ========================================================================

class TestParseJson:
    """Tests for Reflector._parse_json() static method."""

    def test_parse_valid_json_in_text(self):
        """Should extract JSON from text containing additional content."""
        text = 'Some prefix text\n{"verdict": "SUFFICIENT", "reason": "OK"}\nsuffix'
        result = Reflector._parse_json(text)
        assert result["verdict"] == "SUFFICIENT"
        assert result["reason"] == "OK"

    def test_parse_pure_json(self):
        """Should parse pure JSON string."""
        text = '{"verdict": "INSUFFICIENT", "missing": ["gap1"]}'
        result = Reflector._parse_json(text)
        assert result["verdict"] == "INSUFFICIENT"
        assert result["missing"] == ["gap1"]

    def test_parse_malformed_json_falls_back(self):
        """Malformed JSON should return SUFFICIENT fallback."""
        result = Reflector._parse_json("not json at all {broken")
        assert result["verdict"] == "SUFFICIENT"
        assert "JSON parse failed" in result["reason"]

    def test_parse_no_braces_falls_back(self):
        """Text with no braces should return SUFFICIENT fallback."""
        result = Reflector._parse_json("just plain text, no JSON here")
        assert result["verdict"] == "SUFFICIENT"

    def test_parse_with_newlines_in_json(self):
        """Should handle JSON with embedded newlines."""
        text = '''{"verdict": "INSUFFICIENT",
        "missing": ["item1", "item2"],
        "reason": "回答不完整"}'''
        result = Reflector._parse_json(text)
        assert result["verdict"] == "INSUFFICIENT"
        assert len(result["missing"]) == 2

    def test_parse_returns_all_expected_keys(self):
        """The returned dict should have verdict, missing, suggested_queries, issues, reason."""
        text = '{"verdict": "SUFFICIENT"}'
        result = Reflector._parse_json(text)
        # Even incomplete JSON gets defaults from the fallback
        assert "verdict" in result


# ========================================================================
# review
# ========================================================================

class TestReview:
    """Tests for Reflector.review() method."""

    @pytest.mark.asyncio
    async def test_review_sufficient(self, reflector):
        """Review should return SUFFICIENT verdict when answer is good."""
        reflector._llm_retry = AsyncMock()
        reflector._llm_retry.return_value = _make_llm_response({
            "verdict": "SUFFICIENT",
            "missing": [],
            "suggested_queries": [],
            "issues": [],
            "reason": "回答完整准确",
        })

        result = await reflector.review(
            question="什么是库仑定律",
            answer="库仑定律描述了...",
            observations="搜索结果：库仑定律...",
        )

        assert result["verdict"] == "SUFFICIENT"
        assert result["missing"] == []
        assert result["reason"] == "回答完整准确"
        reflector._llm_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_insufficient(self, reflector):
        """Review should return INSUFFICIENT with gaps when answer is lacking."""
        reflector._llm_retry = AsyncMock()
        reflector._llm_retry.return_value = _make_llm_response({
            "verdict": "INSUFFICIENT",
            "missing": ["未提及库仑力的矢量性"],
            "suggested_queries": ["库仑力 矢量叠加"],
            "issues": ["回答缺少方向信息"],
            "reason": "库仑力是矢量，回答未提及方向",
        })

        result = await reflector.review(
            question="库仑定律的矢量形式是什么",
            answer="F=k·q1q2/r²",
            observations="搜索结果不完整",
        )

        assert result["verdict"] == "INSUFFICIENT"
        assert "未提及库仑力的矢量性" in result["missing"]
        assert "库仑力 矢量叠加" in result["suggested_queries"]

    @pytest.mark.asyncio
    async def test_review_with_history_context(self, reflector):
        """Review should include history_ctx in the prompt when provided."""
        reflector._llm_retry = AsyncMock()
        reflector._llm_retry.return_value = _make_llm_response({
            "verdict": "SUFFICIENT",
            "missing": [],
            "suggested_queries": [],
            "issues": [],
            "reason": "OK",
        })

        await reflector.review(
            question="Q", answer="A", observations="O",
            history_ctx="Previous conversation context",
        )

        # Verify history_ctx was included in the system message
        call_args = reflector._llm_retry.call_args[0][0]
        system_content = call_args[0].content
        assert "Previous conversation context" in system_content

    @pytest.mark.asyncio
    async def test_review_handles_llm_response_without_content(self, reflector):
        """Should handle LLM response objects without .content attribute."""
        reflector._llm_retry = AsyncMock()
        # Create a mock that has no 'content' but has a useful __str__
        resp = MagicMock()
        del resp.content  # Remove .content so hasattr check fails
        resp.__str__.return_value = '{"verdict": "SUFFICIENT"}'
        reflector._llm_retry.return_value = resp

        result = await reflector.review(
            question="Q", answer="A", observations="O",
        )

        assert result["verdict"] == "SUFFICIENT"


# ========================================================================
# run (AgentInput → AgentOutput)
# ========================================================================

class TestRun:
    """Tests for Reflector.run() method."""

    @pytest.mark.asyncio
    async def test_run_success(self, reflector):
        """run() should extract metadata from AgentInput and return AgentOutput."""
        reflector._llm_retry = MagicMock()
        reflector._llm_retry.return_value = _make_llm_response({
            "verdict": "SUFFICIENT",
            "missing": [],
            "suggested_queries": [],
            "issues": [],
            "reason": "OK",
        })

        inp = AgentInput(metadata={
            "question": "什么是AI",
            "answer": "AI是人工智能...",
            "observations": "搜索结果...",
        })

        result = await reflector.run(inp)

        assert result.success is True
        assert result.metadata["verdict"] == "SUFFICIENT"

    @pytest.mark.asyncio
    async def test_run_failure_returns_sufficient_fallback(self, reflector):
        """When review crashes, run() should return SUFFICIENT to avoid infinite loop."""
        reflector._llm_retry = AsyncMock(side_effect=Exception("LLM crashed"))

        inp = AgentInput(metadata={
            "question": "Q", "answer": "A", "observations": "O",
        })

        result = await reflector.run(inp)

        assert result.success is True  # Not False — safe fallback
        assert result.metadata["verdict"] == "SUFFICIENT"
        assert "Review failed" in result.metadata["reason"]
