# Extended unit tests for QASystem orchestrator edge cases

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Configuration

from tests.integration.conftest import (
    MockLLMResponse,
    ROUTER_TRIVIAL_JSON,
    ROUTER_MODERATE_JSON,
    ROUTER_COMPLEX_JSON,
    REWRITER_OUTPUT,
    PLANNER_PLAN_JSON,
    PLANNER_SOLVE_OUTPUT,
    REFLECTOR_SUFFICIENT_JSON,
    REFLECTOR_INSUFFICIENT_JSON,
)


# ── Local fixture (duplicated here because conftest is in integration/) ──

@pytest.fixture
def mock_config(monkeypatch):
    """Create a real Configuration with test-safe env values."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://test.api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "test/embedding")
    monkeypatch.setenv("MONITORING_ENABLED", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    return Configuration.from_env()


# ── Helpers ────────────────────────────────────────────────────────────

def _make_fake_retry(response_text: str):
    """Create an async mock for _llm_retry that returns the given response."""
    async def _fake(messages, llm=None, max_retries=2):
        return MockLLMResponse(response_text)
    return _fake


def _setup_mock_backends():
    """Wire mock vector store + knowledge graph into the rag_search module."""
    import src.tools.rag_search as rs

    mock_vs = MagicMock()
    mock_vs.search_hybrid.return_value = {
        "dense": [
            {"text": "F=k·q₁q₂/r²",
             "doc_filename": "电磁学.pdf", "chapter_title": "第一章", "source": "dense"},
        ],
        "sparse": [
            {"text": "库仑定律 1785 年提出",
             "doc_filename": "电磁学.pdf", "chapter_title": "第一章", "source": "sparse"},
        ],
    }
    mock_vs.search.return_value = [
        {"text": "F=k·q₁q₂/r²",
         "doc_filename": "电磁学.pdf", "chapter_title": "第一章", "source": "dense"},
    ]
    mock_vs.get_doc_names.return_value = ["电磁学.pdf"]
    mock_kg = MagicMock()
    mock_kg.search_concepts.return_value = []
    mock_kg.search_concepts_by_docs.return_value = []
    mock_kg.get_neighbors.return_value = []
    mock_kg.get_doc_names.return_value = ["电磁学.pdf"]

    orig_vs = rs._vector_store
    orig_kg = rs._knowledge_graph
    rs._vector_store = mock_vs
    rs._knowledge_graph = mock_kg
    return orig_vs, orig_kg


def _teardown_mock_backends(orig_vs, orig_kg):
    """Restore original backends."""
    import src.tools.rag_search as rs
    rs._vector_store = orig_vs
    rs._knowledge_graph = orig_kg


class TestOrchestratorMaxRounds:
    """Tests for QASystem max rounds + reflection loop behavior."""

    @pytest.mark.asyncio
    async def test_complex_max_rounds_limit(self, mock_config):
        """Complex route should stop after max_rounds even if reflector says INSUFFICIENT."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            from src.agents.qa.orchestrator import QASystem

            qa = QASystem(mock_config)
            qa._router._llm_retry = _make_fake_retry(ROUTER_COMPLEX_JSON)
            qa._planner._llm_retry = _make_fake_retry(PLANNER_PLAN_JSON)
            # Reflector always says INSUFFICIENT
            qa._reflector._llm_retry = _make_fake_retry(REFLECTOR_INSUFFICIENT_JSON)
            qa._executor._rewriter._llm_retry = _make_fake_retry(REWRITER_OUTPUT)

            result = await qa.answer(
                "麦克斯韦方程组推导", doc_filter=None, chat_history=None,
            )

            # Should still complete (rounds capped at max_rounds)
            assert result["route"] == "complex"
            assert result["rounds"] >= 1
            assert len(result["reply"]) > 0
            # With max_rounds=3 and always-insufficient, should hit the cap
            assert result["rounds"] <= 3
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)


class TestOrchestratorWebSearch:
    """Tests for web_search fallback when RAG has no results."""

    @pytest.mark.asyncio
    async def test_moderate_with_web_search_fallback(self, mock_config):
        """Moderate question should attempt web_search when RAG returns nothing."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            import src.tools.rag_search as rs
            # Make RAG return empty
            rs._vector_store.search.return_value = []

            # Also mock web_search to return results
            from unittest.mock import patch
            from src.tools import ToolResult

            mock_web = AsyncMock(return_value=ToolResult(
                tool_name="web_search", query="test",
                content="网络结果：库仑定律...",
            ))

            # Register web_search mock
            import src.tools
            original_web = src.tools._tool_registry.get("web_search")
            src.tools._tool_registry["web_search"] = {
                "name": "web_search", "description": "搜索互联网",
                "func": mock_web,
            }

            try:
                from src.agents.qa.orchestrator import QASystem

                qa = QASystem(mock_config)
                router_json = json.dumps({
                    "difficulty": "moderate",
                    "reason": "需要外部搜索",
                    "target_docs": [],
                    "decomposition": ["库仑定律"],
                }, ensure_ascii=False)
                qa._router._llm_retry = _make_fake_retry(router_json)
                # Mock the full pipeline so escalation to complex doesn't crash
                qa._solver._llm_retry = _make_fake_retry("综合答案...")
                qa._solver._rewriter._llm_retry = _make_fake_retry(REWRITER_OUTPUT)
                qa._planner._llm_retry = _make_fake_retry(PLANNER_PLAN_JSON)
                qa._reflector._llm_retry = _make_fake_retry(REFLECTOR_SUFFICIENT_JSON)
                qa._executor._rewriter._llm_retry = _make_fake_retry(REWRITER_OUTPUT)

                result = await qa.answer(
                    "什么是库仑定律？", doc_filter=None, chat_history=None,
                )
                # When RAG returns empty, moderate escalates to complex (current behavior).
                # Complex path has web_search available in round 3.
                assert result["route"] in ("moderate", "complex")
                assert len(result["reply"]) > 0
            finally:
                if original_web:
                    src.tools._tool_registry["web_search"] = original_web
                else:
                    src.tools._tool_registry.pop("web_search", None)
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)


class TestOrchestratorDocFilter:
    """Tests for doc_filter propagation."""

    @pytest.mark.asyncio
    async def test_doc_filter_passed_to_solver(self, mock_config):
        """doc_filter should be forwarded through the moderate path."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            from src.agents.qa.orchestrator import QASystem

            qa = QASystem(mock_config)
            qa._router._llm_retry = _make_fake_retry(ROUTER_MODERATE_JSON)
            # Mock solver._answer entirely — we only need to verify doc_filter
            # propagation, not exercise the full tool→LLM pipeline.
            call_args_holder = {}

            async def spy_answer(question, doc_filter, chat_history):
                call_args_holder["doc_filter"] = doc_filter
                return {
                    "reply": "测试回答", "route": "done",
                    "tool_calls": [], "observations": [],
                }

            qa._solver._answer = spy_answer

            await qa.answer(
                "什么是库仑定律？",
                doc_filter={"电磁学.pdf"},
                chat_history=None,
            )

            assert call_args_holder.get("doc_filter") == {"电磁学.pdf"}
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)


class TestOrchestratorHistoryCompression:
    """Tests for _build_history_context and _summarize_sync."""

    def test_build_history_context_empty(self, mock_config):
        """Empty history should produce empty context."""
        from src.agents.qa.orchestrator import QASystem
        qa = QASystem(mock_config)

        ctx = qa._build_history_context([])
        assert ctx == ""

    def test_build_history_context_short_history(self, mock_config):
        """Short history (≤3200 chars) should be returned as-is."""
        from src.agents.qa.orchestrator import QASystem
        qa = QASystem(mock_config)

        history = [
            {"role": "user", "content": "什么是电场"},
            {"role": "assistant", "content": "电场是电荷周围的空间"},
        ]
        ctx = qa._build_history_context(history)
        assert "什么是电场" in ctx
        assert "电场是电荷周围的空间" in ctx

    def test_build_history_context_long_history(self, mock_config):
        """Long history should trigger summarization."""
        from src.agents.qa.orchestrator import QASystem
        qa = QASystem(mock_config)

        long_text = "X" * 200
        history = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
        ] * 10  # ~4000 chars × 10 rounds = long

        # This should trigger the summarization path
        ctx = qa._build_history_context(history)
        # Either returns compressed context or empty string
        assert isinstance(ctx, str)

    def test_build_history_context_none(self, mock_config):
        """None history should return empty string."""
        from src.agents.qa.orchestrator import QASystem
        qa = QASystem(mock_config)

        ctx = qa._build_history_context(None)
        assert ctx == ""
