# Extended unit tests for Planner edge cases

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.qa.planner import Planner
from src.config import Configuration


# ── Fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def planner():
    """Create a Planner with mocked LLM."""
    config = MagicMock(spec=Configuration)
    config.llm_model_id = "deepseek-chat"
    config.llm_api_key = "sk-test"
    config.llm_base_url = "https://api.test.com"
    config.llm_temperature = 0.0
    config.llm_max_tokens = 6000

    agent = Planner(config)
    agent._llm = MagicMock()
    return agent


# ========================================================================
# _extract_json (replaces old _parse_json — returns None on failure, not {})
# ========================================================================

class TestParseJson:
    def test_parse_valid_json(self):
        result = Planner._extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_in_text(self):
        result = Planner._extract_json('prefix\n{"sub_questions": []}\nsuffix')
        assert result == {"sub_questions": []}

    def test_parse_with_newlines(self):
        text = '''{"sub_questions": [
            {"id": 1, "question": "Q1"}
        ]}'''
        result = Planner._extract_json(text)
        assert len(result["sub_questions"]) == 1

    def test_parse_nested_braces(self):
        """Brace-counting fallback handles nested {} (e.g. math formulas)."""
        text = '{"desc": "equation ${E = mc^2}$ in text"}'
        result = Planner._extract_json(text)
        assert result == {"desc": "equation ${E = mc^2}$ in text"}

    def test_parse_malformed_returns_none(self):
        assert Planner._extract_json("not json") is None

    def test_parse_no_braces_returns_none(self):
        assert Planner._extract_json("no braces") is None

    def test_parse_broken_json_returns_none(self):
        assert Planner._extract_json('{"broken": ') is None

    def test_parse_empty_string(self):
        assert Planner._extract_json("") is None


# ========================================================================
# _get_doc_list
# ========================================================================

class TestGetDocList:
    def test_returns_doc_names(self):
        with patch("src.agents.qa.planner.get_doc_names",
                   return_value=["a.pdf", "b.pdf"]):
            result = Planner._get_doc_list()
            assert result == ["a.pdf", "b.pdf"]

    def test_fallback_on_error(self):
        with patch("src.agents.qa.planner.get_doc_names",
                   side_effect=Exception("unavailable")):
            result = Planner._get_doc_list()
            assert result == []


# ========================================================================
# plan
# ========================================================================

class TestPlan:
    @pytest.mark.asyncio
    async def test_plan_with_seed_decomposition(self, planner):
        """Plan should include seed decomposition from Router."""
        planner._get_doc_list = MagicMock(return_value=["book.pdf"])
        mock_resp = MagicMock()
        mock_resp.content = '{"sub_questions": [{"id": 1, "question": "什么是梯度下降", "keywords": ["梯度下降"], "tool": "rag_search", "depends_on": []}]}'
        planner._llm_retry = AsyncMock(return_value=mock_resp)

        result = await planner.plan(
            question="解释梯度下降",
            seed_decomposition=["什么是梯度下降", "梯度下降的变体"],
        )

        assert len(result) > 0
        # seed_decomposition should be included in prompt
        call_args = planner._llm_retry.call_args[0][0]
        prompt_text = call_args[0].content
        assert "什么是梯度下降" in prompt_text

    @pytest.mark.asyncio
    async def test_plan_with_feedback(self, planner):
        """Plan should incorporate reviewer feedback."""
        planner._get_doc_list = MagicMock(return_value=[])
        mock_resp = MagicMock()
        mock_resp.content = '{"sub_questions": []}'
        planner._llm_retry = AsyncMock(return_value=mock_resp)

        result = await planner.plan(
            question="test",
            feedback="上一轮的回答缺少细节",
        )

        assert result == []
        call_args = planner._llm_retry.call_args[0][0]
        prompt_text = call_args[0].content
        assert "缺少细节" in prompt_text

    @pytest.mark.asyncio
    async def test_plan_json_parse_failure(self, planner):
        """When LLM returns invalid JSON, plan() should return empty list."""
        planner._get_doc_list = MagicMock(return_value=[])
        mock_resp = MagicMock()
        mock_resp.content = "garbage, not json at all"
        planner._llm_retry = AsyncMock(return_value=mock_resp)

        result = await planner.plan(question="test")
        assert result == []

    @pytest.mark.asyncio
    async def test_plan_with_no_sub_questions_key(self, planner):
        """When JSON has no 'sub_questions' key, return empty list."""
        planner._get_doc_list = MagicMock(return_value=[])
        mock_resp = MagicMock()
        mock_resp.content = '{"other_field": "value"}'
        planner._llm_retry = AsyncMock(return_value=mock_resp)

        result = await planner.plan(question="test")
        assert result == []


# ========================================================================
# solve
# ========================================================================

class TestSolve:
    @pytest.mark.asyncio
    async def test_solve_basic(self, planner):
        """Solve should synthesize observations into an answer."""
        mock_resp = MagicMock()
        mock_resp.content = "综合答案：梯度下降是机器学习中的核心优化算法。"
        planner._llm_retry = AsyncMock(return_value=mock_resp)

        result = await planner.solve(
            question="什么是梯度下降",
            observations="搜索结果：梯度下降是...",
        )

        assert "梯度下降" in result
        planner._llm_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_with_history_context(self, planner):
        """History context should be included in the prompt."""
        mock_resp = MagicMock()
        mock_resp.content = "答案"
        planner._llm_retry = AsyncMock(return_value=mock_resp)

        await planner.solve(
            question="Q",
            observations="O",
            history_ctx="之前讨论了电场的基本概念",
        )

        call_args = planner._llm_retry.call_args[0][0]
        prompt_text = call_args[0].content
        assert "电场的基本概念" in prompt_text

    @pytest.mark.asyncio
    async def test_solve_handles_response_without_content(self, planner):
        """Should handle response objects without .content via str()."""
        # Create a mock that has no .content attribute
        mock_resp = MagicMock()
        # Delete the content attribute so hasattr(resp, "content") returns False
        del mock_resp.content
        # Configure __str__ to return a value
        mock_resp.__str__ = MagicMock(return_value="fallback answer")
        planner._llm_retry = AsyncMock(return_value=mock_resp)

        result = await planner.solve(question="Q", observations="O")
        assert result == "fallback answer"
