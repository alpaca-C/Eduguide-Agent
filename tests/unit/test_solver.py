# Unit tests for DirectSolver (think → act → synthesize)

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.qa.solver import DirectSolver
from src.agents.base import AgentInput
from src.config import Configuration
from src.tools import ToolResult


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
def solver(config):
    """Create a DirectSolver with mocked LLM, tools, and rewriter."""
    agent = DirectSolver(config)
    # Replace internal components with mocks
    agent._llm = MagicMock()
    return agent


# ── Helpers ────────────────────────────────────────────────────────────

def _make_llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


# ========================================================================
# _tool_descs
# ========================================================================

class TestToolDescs:
    """Tests for DirectSolver._tool_descs()."""

    def test_formats_allowed_tools(self, solver):
        """Should format descriptions for allowed tools only."""
        solver._tools = {
            "rag_search": {"name": "rag_search", "description": "搜索本地文档"},
            "web_search": {"name": "web_search", "description": "搜索互联网"},
        }
        result = solver._tool_descs({"rag_search"})
        assert "rag_search" in result
        assert "搜索本地文档" in result
        assert "web_search" not in result

    def test_empty_allowed_set(self, solver):
        """Empty allowed set should produce empty string."""
        solver._tools = {"rag_search": {"name": "r", "description": "d"}}
        result = solver._tool_descs(set())
        assert result == ""


# ========================================================================
# _build_context
# ========================================================================

class TestBuildContext:
    """Tests for DirectSolver._build_context()."""

    def test_basic_context_without_history(self, solver):
        """Without chat history, should only include the question."""
        ctx = solver._build_context("什么是梯度下降", chat_history=None)
        assert "什么是梯度下降" in ctx
        assert "对话历史" not in ctx

    def test_context_with_chat_history(self, solver):
        """Should include last 10 messages from chat history."""
        history = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ]
        ctx = solver._build_context("Q2", chat_history=history)
        assert "学生" in ctx
        assert "答疑老师" in ctx
        assert "Q1" in ctx
        assert "A1" in ctx

    def test_context_truncates_long_messages(self, solver):
        """Long chat messages should be truncated to 300 chars."""
        long_msg = "X" * 500
        history = [{"role": "user", "content": long_msg}]
        ctx = solver._build_context("Q", chat_history=history)
        # The truncated message should appear (300 chars)
        assert long_msg[:300] in ctx
        assert long_msg not in ctx  # Full 500 chars truncated

    def test_context_trims_to_last_10(self, solver):
        """Only the last 10 messages should be included."""
        history = [{"role": "user", "content": f"MSG_{i:03d}"} for i in range(15)]
        ctx = solver._build_context("Q", chat_history=history)
        # MSG_000 to MSG_004 should NOT appear (first 5 trimmed)
        assert "MSG_000" not in ctx
        assert "MSG_001" not in ctx
        assert "MSG_004" not in ctx
        # MSG_005 to MSG_014 should appear (last 10)
        assert "MSG_005" in ctx
        assert "MSG_014" in ctx


# ========================================================================
# _parse_tool_calls_from_text
# ========================================================================

class TestParseToolCallsFromText:
    """Tests for DirectSolver._parse_tool_calls_from_text()."""

    def test_parses_rag_search_colon_format(self, solver):
        """Should parse rag_search: 查询内容 format."""
        text = "rag_search: 什么是机器学习"
        solver._tools = {"rag_search": {}}
        calls = solver._parse_tool_calls_from_text(text, {"rag_search"})
        assert len(calls) == 1
        assert calls[0]["name"] == "rag_search"
        assert calls[0]["args"]["query"] == "什么是机器学习"

    def test_parses_rag_search_chinese_colon(self, solver):
        """Should parse rag_search：查询内容 (full-width colon)."""
        text = "rag_search：深度学习基础"
        solver._tools = {"rag_search": {}}
        calls = solver._parse_tool_calls_from_text(text, {"rag_search"})
        assert len(calls) == 1
        assert calls[0]["args"]["query"] == "深度学习基础"

    def test_parses_multiple_tools(self, solver):
        """Should parse multiple tool calls from text."""
        text = (
            "rag_search: 梯度下降\n"
            "rag_search: 随机梯度下降"
        )
        solver._tools = {"rag_search": {}}
        calls = solver._parse_tool_calls_from_text(text, {"rag_search"})
        assert len(calls) == 2
        queries = {c["args"]["query"] for c in calls}
        assert queries == {"梯度下降", "随机梯度下降"}

    def test_ignores_tools_not_in_allowed(self, solver):
        """Should not parse tools that are not in the allowed set."""
        text = "web_search: 最新新闻"
        solver._tools = {"rag_search": {}, "web_search": {}}
        calls = solver._parse_tool_calls_from_text(text, {"rag_search"})
        # web_search is not allowed, so it should be ignored
        assert len(calls) == 0

    def test_empty_text_produces_no_calls(self, solver):
        """Empty text should produce no tool calls."""
        solver._tools = {"rag_search": {}}
        calls = solver._parse_tool_calls_from_text("", {"rag_search"})
        assert calls == []

    def test_generates_unique_call_ids(self, solver):
        """Each parsed call should have a unique id."""
        text = "rag_search: query1\nrag_search: query2"
        solver._tools = {"rag_search": {}}
        calls = solver._parse_tool_calls_from_text(text, {"rag_search"})
        ids = {c["id"] for c in calls}
        assert len(ids) == 2  # All unique


# ========================================================================
# _execute_tools
# ========================================================================

class TestExecuteTools:
    """Tests for DirectSolver._execute_tools()."""

    @pytest.mark.asyncio
    async def test_executes_rag_search(self, solver):
        """Should execute rag_search tool with filter_docs."""
        mock_func = AsyncMock(return_value=ToolResult(
            tool_name="rag_search", query="test",
            content="search result content",
        ))
        solver._tools = {
            "rag_search": {"name": "rag_search", "func": mock_func},
        }

        results = await solver._execute_tools(
            [{"name": "rag_search", "args": {"query": "test query"}}],
            doc_filter={"doc1.pdf"},
        )

        assert len(results) == 1
        assert results[0].content == "search result content"
        mock_func.assert_called_once_with(query="test query", filter_docs={"doc1.pdf"})

    @pytest.mark.asyncio
    async def test_executes_non_rag_tool(self, solver):
        """Should execute non-rag tools without filter_docs."""
        mock_func = AsyncMock(return_value=ToolResult(
            tool_name="web_search", query="test", content="web results",
        ))
        solver._tools = {
            "web_search": {"name": "web_search", "func": mock_func},
        }

        results = await solver._execute_tools(
            [{"name": "web_search", "args": {"query": "news"}}],
            doc_filter=None,
        )

        assert len(results) == 1
        # Non-rag tools should NOT receive filter_docs
        mock_func.assert_called_once_with(query="news")

    @pytest.mark.asyncio
    async def test_skips_unknown_tools(self, solver):
        """Unknown tool names should be skipped."""
        solver._tools = {}
        results = await solver._execute_tools(
            [{"name": "nonexistent", "args": {"query": "x"}}],
            doc_filter=None,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_tool_calls(self, solver):
        """Empty list should return empty results."""
        results = await solver._execute_tools([], doc_filter=None)
        assert results == []

    @pytest.mark.asyncio
    async def test_handles_tool_exception(self, solver):
        """Tool exceptions should be caught and wrapped in error ToolResult."""
        mock_func = AsyncMock(side_effect=Exception("tool crashed"))
        solver._tools = {
            "rag_search": {"name": "rag_search", "func": mock_func},
        }

        results = await solver._execute_tools(
            [{"name": "rag_search", "args": {"query": "test"}}],
            doc_filter=None,
        )

        assert len(results) == 1
        assert results[0].tool_name == "error"
        assert "工具执行出错" in results[0].content


# ========================================================================
# _answer (full pipeline)
# ========================================================================

class TestAnswer:
    """Tests for DirectSolver._answer() — the full pipeline."""

    @pytest.mark.asyncio
    async def test_answer_with_results(self, solver):
        """Full pipeline with successful search and synthesis."""
        # Mock rewriter
        solver._rewriter.rewrite = AsyncMock(
            return_value=["梯度下降", "优化算法"]
        )

        # Mock rag_search tool
        rag_mock = AsyncMock(return_value=ToolResult(
            tool_name="rag_search", query="test",
            content="some search result",
        ))
        solver._tools = {"rag_search": {"name": "rag_search", "func": rag_mock}}

        # Mock LLM synthesis
        solver._llm_retry = AsyncMock(
            return_value=_make_llm_response("综合答案：梯度下降是...")
        )

        result = await solver._answer("什么是梯度下降", doc_filter=None, chat_history=None)

        assert result["route"] == "done"
        assert "梯度下降" in result["reply"]
        assert len(result["tool_calls"]) == 2
        # rewriter should have been called
        solver._rewriter.rewrite.assert_called_once_with("什么是梯度下降")

    @pytest.mark.asyncio
    async def test_answer_no_results_but_synthesis_ok(self, solver):
        """Even without search results, synthesis should still produce an answer."""
        solver._rewriter.rewrite = AsyncMock(return_value=["query1"])
        solver._tools = {}  # No tools registered

        solver._llm_retry = AsyncMock(
            return_value=_make_llm_response("基于我的知识，梯度下降是...")
        )

        result = await solver._answer("什么是梯度下降", doc_filter=None, chat_history=None)

        assert result["route"] == "done"  # Has reply text
        assert "梯度下降" in result["reply"]

    @pytest.mark.asyncio
    async def test_answer_escalates_when_empty(self, solver):
        """When both observations and reply are empty, escalate to Planner."""
        solver._rewriter.rewrite = AsyncMock(return_value=["query1"])

        # Tool raises exception → observations stays empty
        solver._tools = {
            "rag_search": {
                "name": "rag_search",
                "func": AsyncMock(side_effect=Exception("search failed")),
            }
        }

        # Synthesis returns empty
        solver._llm_retry = AsyncMock(
            return_value=_make_llm_response("")
        )

        result = await solver._answer("复杂问题", doc_filter=None, chat_history=None)

        assert result["route"] == "escalate"

    @pytest.mark.asyncio
    async def test_answer_handles_synthesis_failure(self, solver):
        """When LLM synthesis fails, should return error reply."""
        solver._rewriter.rewrite = AsyncMock(return_value=["query1"])
        solver._tools = {
            "rag_search": {
                "name": "rag_search",
                "func": AsyncMock(return_value=ToolResult(
                    tool_name="rag_search", query="test", content="results",
                )),
            }
        }
        solver._llm_retry = AsyncMock(side_effect=Exception("LLM down"))

        result = await solver._answer("question", doc_filter=None, chat_history=None)

        assert "回答生成失败" in result["reply"]


# ========================================================================
# run (AgentInput → AgentOutput)
# ========================================================================

class TestSolverRun:
    """Tests for DirectSolver.run() method."""

    @pytest.mark.asyncio
    async def test_run_success(self, solver):
        """run() should extract question from AgentInput and return AgentOutput."""
        solver._rewriter.rewrite = AsyncMock(return_value=["q1"])
        solver._tools = {}
        solver._llm_retry = AsyncMock(
            return_value=_make_llm_response("答案")
        )

        inp = AgentInput(metadata={
            "question": "什么是机器学习",
            "doc_filter": None,
            "chat_history": None,
        })

        result = await solver.run(inp)

        assert result.success is True
        assert "答案" in result.metadata["reply"]

    @pytest.mark.asyncio
    async def test_run_failure(self, solver):
        """When _answer raises, run() should return AgentOutput with error."""
        solver._rewriter.rewrite = AsyncMock(side_effect=Exception("rewriter down"))

        inp = AgentInput(metadata={"question": "Q"})

        result = await solver.run(inp)

        assert result.success is False
        assert "rewriter down" in result.error
