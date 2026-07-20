"""Unit tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mrr,
    average_precision,
    compute_all_metrics,
)


class TestRecallAtK:
    def test_all_relevant_found(self):
        relevant = {"d1", "d2", "d3"}
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        assert recall_at_k(relevant, retrieved, k=3) == 1.0

    def test_half_found(self):
        relevant = {"d1", "d2", "d3", "d4"}
        retrieved = ["d1", "d2", "d5", "d6", "d7"]
        assert recall_at_k(relevant, retrieved, k=5) == 0.5

    def test_none_found(self):
        relevant = {"d1", "d2"}
        retrieved = ["d3", "d4", "d5"]
        assert recall_at_k(relevant, retrieved, k=3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(set(), ["d1"], k=1) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        relevant = {"d1", "d2"}
        retrieved = ["d1", "d2", "d3"]
        assert precision_at_k(relevant, retrieved, k=2) == 1.0

    def test_half_relevant(self):
        relevant = {"d1", "d3"}
        retrieved = ["d1", "d2", "d3", "d4"]
        assert precision_at_k(relevant, retrieved, k=4) == 0.5


class TestNDCG:
    def test_perfect_ranking(self):
        relevant = {"d1": 3, "d2": 2, "d3": 1}
        retrieved = ["d1", "d2", "d3"]
        assert ndcg_at_k(relevant, retrieved, k=3) == pytest.approx(1.0, abs=0.001)

    def test_reversed_ranking(self):
        relevant = {"d1": 1, "d2": 2, "d3": 3}
        retrieved = ["d1", "d2", "d3"]  # worst first, best last
        ndcg = ndcg_at_k(relevant, retrieved, k=3)
        assert ndcg < 1.0


class TestMRR:
    def test_first_position(self):
        assert mrr({"d1"}, ["d1", "d2", "d3"]) == 1.0

    def test_third_position(self):
        assert mrr({"d3"}, ["d1", "d2", "d3"]) == pytest.approx(1.0 / 3)

    def test_not_found(self):
        assert mrr({"d9"}, ["d1", "d2", "d3"]) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision({"d1", "d2"}, ["d1", "d2"]) == 1.0

    def test_imperfect(self):
        ap = average_precision({"d1", "d3"}, ["d1", "d2", "d3", "d4"])
        assert ap < 1.0
        assert ap > 0.0


class TestComputeAllMetrics:
    def test_basic(self):
        queries = {"q1": {"d1", "d2"}, "q2": {"d3"}}
        retrieved = {"q1": ["d1", "d3", "d2"], "q2": ["d5", "d6"]}
        metrics = compute_all_metrics(queries, retrieved, k_values=(1, 3))

        assert metrics["num_queries"] == 2
        assert metrics["Recall@1"] > 0
        assert metrics["Recall@1"] < 1.0
        assert 0 <= metrics["MRR"] <= 1.0
        assert 0 <= metrics["MAP"] <= 1.0
