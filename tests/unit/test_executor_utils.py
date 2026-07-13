# Unit tests for Executor pure utility functions

from __future__ import annotations

import pytest

from src.agents.qa.executor import Executor
from src.tools import ToolResult


# ========================================================================
# _topological_rounds
# ========================================================================

class TestTopologicalRounds:
    def test_no_dependencies_single_round(self):
        """All items without depends_on → single round."""
        plan = [
            {"id": 1, "question": "Q1"},
            {"id": 2, "question": "Q2"},
            {"id": 3, "question": "Q3"},
        ]
        rounds = Executor._topological_rounds(plan)
        assert len(rounds) == 1
        assert set(rounds[0]) == {1, 2, 3}

    def test_empty_plan_single_round(self):
        """Empty plan → single empty round."""
        rounds = Executor._topological_rounds([])
        assert rounds == [[]]

    def test_linear_dependencies(self):
        """A depends on nothing, B depends on A, C depends on B."""
        plan = [
            {"id": 1, "depends_on": []},
            {"id": 2, "depends_on": [1]},
            {"id": 3, "depends_on": [2]},
        ]
        rounds = Executor._topological_rounds(plan)
        # Should be 3 sequential rounds
        assert len(rounds) == 3
        assert rounds[0] == [1]
        assert rounds[1] == [2]
        assert rounds[2] == [3]

    def test_diamond_dependencies(self):
        """A → B, A → C, B → D, C → D."""
        plan = [
            {"id": 1, "depends_on": []},
            {"id": 2, "depends_on": [1]},
            {"id": 3, "depends_on": [1]},
            {"id": 4, "depends_on": [2, 3]},
        ]
        rounds = Executor._topological_rounds(plan)
        # Round 0: {1}, Round 1: {2, 3}, Round 2: {4}
        assert len(rounds) == 3
        assert set(rounds[0]) == {1}
        assert set(rounds[1]) == {2, 3}
        assert set(rounds[2]) == {4}

    def test_independent_groups_in_same_round(self):
        """Multiple independent chains should run in parallel within rounds."""
        plan = [
            {"id": 1, "depends_on": []},
            {"id": 2, "depends_on": [1]},
            {"id": 3, "depends_on": []},      # Independent of chain 1-2
            {"id": 4, "depends_on": [3]},
        ]
        rounds = Executor._topological_rounds(plan)
        assert len(rounds) == 2
        assert set(rounds[0]) == {1, 3}
        assert set(rounds[1]) == {2, 4}

    def test_circular_dependency_handled(self):
        """Circular dependency should be gracefully handled (flushed as final round)."""
        plan = [
            {"id": 1, "depends_on": [2]},
            {"id": 2, "depends_on": [1]},
        ]
        rounds = Executor._topological_rounds(plan)
        # Should not infinite loop — should flush remaining as final round
        assert len(rounds) >= 1
        all_ids = [sid for rnd in rounds for sid in rnd]
        assert 1 in all_ids
        assert 2 in all_ids

    def test_missing_dependency(self):
        """Sub-question depends on a non-existent ID → handled gracefully."""
        plan = [
            {"id": 1, "depends_on": [99]},  # 99 doesn't exist
        ]
        rounds = Executor._topological_rounds(plan)
        # Should still complete without infinite loop
        assert 1 in [sid for rnd in rounds for sid in rnd]


# ========================================================================
# _summarize_for_context
# ========================================================================

class TestSummarizeForContext:
    def test_basic_summary(self):
        results = [
            ToolResult(tool_name="rag", query="q",
                       content="库仑定律是电磁学的基本定律。"),
        ]
        summary = Executor._summarize_for_context("什么是库仑定律", results)
        assert "Q: 什么是库仑定律" in summary
        assert "库仑定律" in summary

    def test_empty_results(self):
        summary = Executor._summarize_for_context("Q", [])
        assert summary == "Q: Q"

    def test_truncates_to_500_chars(self):
        """Summary should not exceed 500 characters."""
        long_result = ToolResult(tool_name="r", query="q", content="X" * 1000)
        summary = Executor._summarize_for_context("Q", [long_result])
        assert len(summary) <= 500

    def test_multiple_results(self):
        results = [
            ToolResult(tool_name="r1", query="q1", content="result A"),
            ToolResult(tool_name="r2", query="q2", content="result B"),
        ]
        summary = Executor._summarize_for_context("Q", results)
        assert "result A" in summary
        assert "result B" in summary

    def test_result_without_content(self):
        """Results without .content should be skipped gracefully."""
        # Use a regular object without content attribute
        class NoContentResult:
            pass
        summary = Executor._summarize_for_context("Q", [NoContentResult()])
        assert summary == "Q: Q"
