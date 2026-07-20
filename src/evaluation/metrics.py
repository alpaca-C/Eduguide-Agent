"""
Retrieval evaluation metrics.

Standard IR metrics used by BEIR / TREC / industry:
  - Recall@K    : fraction of relevant docs retrieved in top-K
  - Precision@K : fraction of top-K that are relevant
  - NDCG@K      : position-weighted relevance (higher rank = more weight)
  - MRR         : mean reciprocal rank of first relevant doc
  - MAP         : mean average precision across queries
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def recall_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    """R@K: how many relevant docs appear in top-K?"""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / len(relevant)


def precision_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    """P@K: what fraction of top-K results are relevant?"""
    if k == 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / k


def ndcg_at_k(
    relevant: dict[str, int],  # doc_id → relevance score (1, 2, 3...)
    retrieved: list[str],
    k: int,
) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    DCG = sum(relevance_i / log2(rank_i + 1))
    NDCG = DCG / IDCG (where IDCG is DCG of ideal ranking)
    """
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevant.get(doc_id, 0)
        if rel > 0:
            dcg += rel / np.log2(i + 2)  # i+2 because ranks start at 1

    # IDCG (ideal: all relevant docs ranked by relevance score descending)
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def mrr(
    relevant: set[str],
    retrieved: list[str],
) -> float:
    """
    MRR: Mean Reciprocal Rank.

    For each query: reciprocal_rank = 1 / rank_of_first_relevant
    Rank starts at 1.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    relevant: set[str],
    retrieved: list[str],
) -> float:
    """
    AP: Average Precision.

    AP = sum(P@k * rel(k)) / num_relevant
    where rel(k) = 1 if doc at rank k is relevant, else 0
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_prec += hits / (i + 1)

    return sum_prec / len(relevant)


def compute_all_metrics(
    queries: dict[str, set[str]],          # query_id → {relevant_doc_ids}
    retrieved: dict[str, list[str]],        # query_id → [ranked_doc_ids]
    k_values: Sequence[int] = (1, 3, 5, 10),
    relevance_grades: dict[str, dict[str, int]] | None = None,
) -> dict:
    """
    Compute all standard IR metrics across queries.

    Args:
        queries: query_id → set of relevant document ids
        retrieved: query_id → ordered list of retrieved document ids
        k_values: K values for @K metrics
        relevance_grades: query_id → {doc_id → relevance_score (1,2,3)}
                          If provided, NDCG is also computed.

    Returns:
        Dict with mean metrics across all queries.
    """
    results: dict[str, float] = {}

    for k in k_values:
        recalls = []
        precisions = []
        ndcgs = []

        for qid, relevant_docs in queries.items():
            ret = retrieved.get(qid, [])
            recalls.append(recall_at_k(relevant_docs, ret, k))
            precisions.append(precision_at_k(relevant_docs, ret, k))

            if relevance_grades and qid in relevance_grades:
                ndcgs.append(ndcg_at_k(relevance_grades[qid], ret, k))

        results[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"Precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        if ndcgs:
            results[f"NDCG@{k}"] = float(np.mean(ndcgs))

    # MRR
    mrrs = []
    for qid, relevant_docs in queries.items():
        ret = retrieved.get(qid, [])
        mrrs.append(mrr(relevant_docs, ret))
    results["MRR"] = float(np.mean(mrrs)) if mrrs else 0.0

    # MAP
    maps = []
    for qid, relevant_docs in queries.items():
        ret = retrieved.get(qid, [])
        maps.append(average_precision(relevant_docs, ret))
    results["MAP"] = float(np.mean(maps)) if maps else 0.0

    # Query-level stats
    num_empty = sum(1 for v in retrieved.values() if not v)
    results["num_queries"] = len(queries)
    results["empty_results"] = num_empty

    return results


def format_metrics_table(metrics: dict) -> str:
    """Pretty-print metrics as an aligned table."""
    lines = []
    lines.append("=" * 55)
    lines.append(f"{'Metric':<20} {'Value':>10}")
    lines.append("-" * 55)

    metric_order = [
        "num_queries", "empty_results",
        "Recall@1", "Recall@3", "Recall@5", "Recall@10",
        "Precision@1", "Precision@3", "Precision@5", "Precision@10",
        "NDCG@5", "NDCG@10",
        "MRR", "MAP",
    ]

    for name in metric_order:
        if name in metrics:
            val = metrics[name]
            if isinstance(val, float):
                lines.append(f"{name:<20} {val:>10.4f}")
            else:
                lines.append(f"{name:<20} {val:>10}")

    lines.append("=" * 55)
    return "\n".join(lines)
