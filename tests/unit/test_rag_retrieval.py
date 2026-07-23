"""Unit tests for src.tools.rag_retrieval — two-tier retrieval strategy.

Tests the RAGRetrievalStrategy class (state machine for fast↔full escalation)
and the tool wrapper that registers it as "rag_skill".

Each test creates a fresh strategy instance to avoid global state leakage.
Module-level side effects (tool registration, global singleton) are tested
separately with explicit cleanup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.tools import ToolResult, register_tool, _tool_registry


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def strategy():
    """Fresh strategy with clean internal state for each test."""
    from src.tools.rag_retrieval import RAGRetrievalStrategy
    return RAGRetrievalStrategy()


@pytest.fixture
def mock_rag_funcs():
    """Mock rag_search and rag_fullsearch to avoid real ChromaDB/embedding."""
    fast = AsyncMock(return_value=ToolResult(
        tool_name="rag_search", query="test",
        content="fast result content",
    ))
    full = AsyncMock(return_value=ToolResult(
        tool_name="rag_fullsearch", query="test",
        content="full result content",
    ))
    return fast, full


# ═══════════════════════════════════════════════════════════════════════
# _should_escalate — pure logic, no mocks needed
# ═══════════════════════════════════════════════════════════════════════

class TestShouldEscalate:
    """Tests for the escalation decision logic."""

    def test_exact_match(self, strategy):
        strategy.mark_unsatisfied("高斯定理")
        assert strategy._should_escalate("高斯定理") is True

    def test_case_insensitive_match(self, strategy):
        strategy.mark_unsatisfied("GAUSS LAW")
        assert strategy._should_escalate("gauss law") is True

    def test_whitespace_insensitive(self, strategy):
        strategy.mark_unsatisfied("  库仑定律  ")
        assert strategy._should_escalate("库仑定律") is True

    def test_substring_match_query_is_subset(self, strategy):
        strategy.mark_unsatisfied("高斯定理推导")
        assert strategy._should_escalate("高斯定理") is True  # "高斯定理" in "高斯定理推导"

    def test_substring_match_unsatisfied_is_subset(self, strategy):
        strategy.mark_unsatisfied("高斯定理")
        assert strategy._should_escalate("高斯定理推导步骤") is True  # "高斯定理" in "高斯定理推导步骤"

    def test_no_match_different_queries(self, strategy):
        strategy.mark_unsatisfied("库仑定律")
        assert strategy._should_escalate("高斯定理") is False

    def test_no_match_empty_unsatisfied(self, strategy):
        assert strategy._should_escalate("anything") is False

    def test_no_query_registered_returns_false(self, strategy):
        """No queries marked → should never escalate."""
        assert strategy._should_escalate("anything") is False


# ═══════════════════════════════════════════════════════════════════════
# search() routing
# ═══════════════════════════════════════════════════════════════════════

class TestSearchRouting:
    """Tests for search() dispatching to fast vs full."""

    @pytest.mark.asyncio
    async def test_default_uses_fast_path(self, strategy, mock_rag_funcs):
        fast, _ = mock_rag_funcs
        with patch("src.tools.rag_search.rag_search", fast):
            result = await strategy.search("什么是库仑定律")
        assert result.content == "fast result content"
        fast.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_full_skips_fast_path(self, strategy, mock_rag_funcs):
        _, full = mock_rag_funcs
        with patch("src.tools.rag_search.rag_fullsearch", full):
            result = await strategy.search("test", force_full=True)
        assert result.content == "full result content"
        full.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_escalates_after_mark_unsatisfied(self, strategy, mock_rag_funcs):
        """After mark_unsatisfied(), same query should use full path."""
        fast, full = mock_rag_funcs
        strategy.mark_unsatisfied("高斯定理推导")

        with patch("src.tools.rag_search.rag_fullsearch", full):
            result = await strategy.search("高斯定理推导")
        assert result.content == "full result content"
        full.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_escalates_fuzzy_match(self, strategy, mock_rag_funcs):
        """Substring of unsatisfied query should also escalate."""
        _, full = mock_rag_funcs
        strategy.mark_unsatisfied("高斯定理及其推导过程")

        with patch("src.tools.rag_search.rag_fullsearch", full):
            result = await strategy.search("高斯定理")
        assert result.content == "full result content"

    @pytest.mark.asyncio
    async def test_unrelated_query_stays_fast(self, strategy, mock_rag_funcs):
        """Only marked queries escalate; others stay fast."""
        fast, _ = mock_rag_funcs
        strategy.mark_unsatisfied("高斯定理")

        with patch("src.tools.rag_search.rag_search", fast):
            result = await strategy.search("库仑定律")
        assert result.content == "fast result content"

    @pytest.mark.asyncio
    async def test_passes_top_k_and_filter(self, strategy, mock_rag_funcs):
        fast, _ = mock_rag_funcs
        with patch("src.tools.rag_search.rag_search", fast):
            await strategy.search("test", top_k=8, filter_docs={"doc1.pdf"})
        call_kw = fast.call_args.kwargs
        assert call_kw["top_k"] == 8
        assert call_kw["filter_docs"] == {"doc1.pdf"}


# ═══════════════════════════════════════════════════════════════════════
# mark_unsatisfied / mark_satisfied / reset
# ═══════════════════════════════════════════════════════════════════════

class TestStateManagement:
    """Tests for the unsatisfied state machine."""

    def test_mark_unsatisfied_adds_query(self, strategy):
        strategy.mark_unsatisfied("高斯定理")
        assert "高斯定理" in strategy._unsatisfied

    def test_mark_unsatisfied_normalizes(self, strategy):
        """Query is lowercased and stripped."""
        strategy.mark_unsatisfied("  库仑定律  ")
        assert "库仑定律" in strategy._unsatisfied

    def test_mark_unsatisfied_with_session(self, strategy):
        strategy.mark_unsatisfied("query1", session_id="sess-abc")
        assert "sess-abc" in strategy._escalate_session

    def test_mark_unsatisfied_without_session(self, strategy):
        strategy.mark_unsatisfied("query1")
        assert len(strategy._escalate_session) == 0

    def test_mark_satisfied_removes_query(self, strategy):
        strategy.mark_unsatisfied("高斯定理")
        strategy.mark_satisfied("高斯定理")
        assert "高斯定理" not in strategy._unsatisfied

    def test_mark_satisfied_does_not_affect_sessions(self, strategy):
        strategy.mark_unsatisfied("q", session_id="s1")
        strategy.mark_satisfied("q")
        # Session tracking is separate from query tracking
        assert "s1" in strategy._escalate_session

    def test_reset_clears_all_state(self, strategy):
        strategy.mark_unsatisfied("q1")
        strategy.mark_unsatisfied("q2", session_id="s2")
        strategy.reset()
        assert len(strategy._unsatisfied) == 0
        assert len(strategy._escalate_session) == 0

    def test_mark_satisfied_nonexistent_no_error(self, strategy):
        """Removing a query that was never added should not raise."""
        strategy.mark_satisfied("never added")


# ═══════════════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════════════

class TestGlobalSingleton:
    """Tests for get_rag_strategy() singleton factory."""

    def test_returns_same_instance(self):
        from src.tools.rag_retrieval import get_rag_strategy
        s1 = get_rag_strategy()
        s2 = get_rag_strategy()
        assert s1 is s2

    def test_reset_global_between_tests(self):
        """Ensure global singleton can be reset for test isolation."""
        from src.tools.rag_retrieval import get_rag_strategy, _strategy_instance
        # Reset global state
        import src.tools.rag_retrieval as rrm
        rrm._strategy_instance = None

        s = get_rag_strategy()
        assert s is not None
        s.mark_unsatisfied("test")
        assert "test" in s._unsatisfied

        # Clean up — leave singleton clean
        rrm._strategy_instance = None


# ═══════════════════════════════════════════════════════════════════════
# Tool registration
# ═══════════════════════════════════════════════════════════════════════

class TestToolRegistration:
    """Tests for the "rag_skill" tool that wraps RAGRetrievalStrategy."""

    def test_rag_skill_is_registered(self):
        """Module import registers "rag_skill" in the tool registry."""
        assert "rag_skill" in _tool_registry

    def test_rag_skill_func_is_callable(self):
        assert callable(_tool_registry["rag_skill"]["func"])

    def test_rag_skill_has_description(self):
        desc = _tool_registry["rag_skill"]["description"]
        assert "文档检索" in desc or "检索" in desc
