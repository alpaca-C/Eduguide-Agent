# Integration tests for QASystem Orchestrator
#
# Tests the full 3-tier routing pipeline with mocked LLM and backend services.

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from tests.integration.conftest import (
    MockLLMResponse,
    ROUTER_TRIVIAL_JSON,
    ROUTER_MODERATE_JSON,
    ROUTER_COMPLEX_JSON,
    REWRITER_OUTPUT,
    PLANNER_PLAN_JSON,
    PLANNER_SOLVE_OUTPUT,
    REFLECTOR_SUFFICIENT_JSON,
)


# ── Mock helpers ──────────────────────────────────────────────────

def _make_fake_retry(response_text: str):
    """Create an async mock for _llm_retry that returns the given response."""
    async def _fake(messages, llm=None, max_retries=2):
        return MockLLMResponse(response_text)
    return _fake


def _patch_all_sub_agents(qa, router_json, rewriter_text, planner_json, solver_text, reflector_json):
    """Replace _llm_retry on all sub-agents inside a QASystem instance."""
    qa._router._llm_retry = _make_fake_retry(router_json)
    qa._solver._llm_retry = _make_fake_retry("")  # synthesis — not used in moderate path
    qa._solver._rewriter._llm_retry = _make_fake_retry(rewriter_text)
    qa._planner._llm_retry = _make_fake_retry(planner_json)  # used for plan()
    qa._reflector._llm_retry = _make_fake_retry(reflector_json)


def _setup_mock_backends():
    """Wire mock vector store + knowledge graph into the rag_search module."""
    import src.tools.rag_search as rs

    mock_vs = MagicMock()
    mock_vs.search_hybrid.return_value = {
        "dense": [
            {"text": "库仑定律：F=k·q₁q₂/r²",
             "doc_filename": "电磁学.pdf", "chapter_title": "第一章 静电场", "source": "dense"},
        ],
        "sparse": [
            {"text": "库仑定律 1785 年由库仑通过扭秤实验得出。",
             "doc_filename": "电磁学.pdf", "chapter_title": "第一章 静电场", "source": "sparse"},
        ],
    }
    mock_vs.search.return_value = [
        {"text": "库仑定律：F=k·q₁q₂/r²",
         "doc_filename": "电磁学.pdf", "chapter_title": "第一章 静电场", "source": "dense"},
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


class TestOrchestratorTrivial:
    """Tests for trivial (greeting / chitchat) routing."""

    @pytest.mark.asyncio
    async def test_trivial_question_returns_direct_answer(self, mock_config):
        """A greeting should route to direct_answer, 0 rounds, no tools."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            from src.agents.qa.orchestrator import QASystem

            qa = QASystem(mock_config)
            qa._router._llm_retry = _make_fake_retry(ROUTER_TRIVIAL_JSON)

            result = await qa.answer("你好", doc_filter=None, chat_history=None)

            assert result["route"] == "trivial", f"Expected trivial, got {result['route']}"
            assert result["rounds"] == 0
            assert result["tool_calls"] == []
            assert len(result["reply"]) > 0
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)


class TestOrchestratorModerate:
    """Tests for moderate (single-concept) routing."""

    @pytest.mark.asyncio
    async def test_moderate_question_uses_direct_solver(self, mock_config):
        """A single-concept question should route to DirectSolver with tool calls."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            from src.agents.qa.orchestrator import QASystem

            qa = QASystem(mock_config)
            qa._router._llm_retry = _make_fake_retry(ROUTER_MODERATE_JSON)
            qa._solver._llm_retry = _make_fake_retry("综合答案：库仑定律是...")
            qa._solver._rewriter._llm_retry = _make_fake_retry(REWRITER_OUTPUT)

            result = await qa.answer(
                "什么是库仑定律？", doc_filter=None, chat_history=None,
            )

            assert result["route"] == "moderate", f"Expected moderate, got {result['route']}"
            assert result["rounds"] >= 1
            assert len(result["tool_calls"]) > 0
            # Should have called rag_search
            assert any(tc["tool"] == "rag_search" for tc in result["tool_calls"])
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)


class TestOrchestratorComplex:
    """Tests for complex (multi-step reasoning) routing."""

    @pytest.mark.asyncio
    async def test_complex_question_plans_and_solves(self, mock_config):
        """A complex question should go through Planner→Executor→Solver→Reflector."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            from src.agents.qa.orchestrator import QASystem

            qa = QASystem(mock_config)
            qa._router._llm_retry = _make_fake_retry(ROUTER_COMPLEX_JSON)
            qa._planner._llm_retry = _make_fake_retry(PLANNER_PLAN_JSON)
            qa._reflector._llm_retry = _make_fake_retry(REFLECTOR_SUFFICIENT_JSON)
            # Executor uses rewriter internally
            qa._executor._rewriter._llm_retry = _make_fake_retry(REWRITER_OUTPUT)

            result = await qa.answer(
                "麦克斯韦方程组如何从库仑定律和高斯定理推导？",
                doc_filter=None, chat_history=None,
            )

            assert result["route"] == "complex", f"Expected complex, got {result['route']}"
            assert result["rounds"] >= 1
            assert len(result["reply"]) > 0
            assert len(result["tool_calls"]) > 0
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)


class TestOrchestratorEdgeCases:
    """Edge case tests for orchestrator behavior."""

    @pytest.mark.asyncio
    async def test_router_graceful_degradation(self, mock_config):
        """When router returns malformed JSON, orchestrator should still work."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            from src.agents.qa.orchestrator import QASystem

            qa = QASystem(mock_config)
            qa._router._llm_retry = _make_fake_retry("bad json {{")
            qa._solver._llm_retry = _make_fake_retry("兜底答案")
            qa._solver._rewriter._llm_retry = _make_fake_retry(REWRITER_OUTPUT)

            result = await qa.answer(
                "随机问题", doc_filter=None, chat_history=None,
            )

            assert result["route"] in ("moderate", "complex_fallback"), \
                f"Expected moderate or complex_fallback, got {result['route']}"
            assert "reply" in result
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)

    @pytest.mark.asyncio
    async def test_chat_history_injection(self, mock_config):
        """Chat history should be injected into sub-agents without errors."""
        orig_vs, orig_kg = _setup_mock_backends()
        try:
            from src.agents.qa.orchestrator import QASystem

            qa = QASystem(mock_config)
            qa._router._llm_retry = _make_fake_retry(ROUTER_TRIVIAL_JSON)

            history = [
                {"role": "user", "content": "什么是电场？"},
                {"role": "assistant", "content": "电场是电荷周围空间存在的一种特殊物质..."},
                {"role": "user", "content": "谢谢"},
            ]
            result = await qa.answer(
                "谢谢老师", doc_filter=None, chat_history=history,
            )

            assert result["route"] == "trivial"
            assert len(result["reply"]) > 0
        finally:
            _teardown_mock_backends(orig_vs, orig_kg)
