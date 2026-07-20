"""
Retrieval evaluator — runs evaluation on a dataset and reports metrics.

Supports:
  - Single strategy evaluation
  - Strategy comparison (ablation): dense vs sparse vs hybrid
  - Pretty-printed report
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from .metrics import compute_all_metrics, format_metrics_table

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Run retrieval evaluation and compare strategies."""

    def __init__(self, retriever=None):
        """
        Args:
            retriever: Object with a search(corpus, queries, top_k) method.
                       If None, uses DummyRetriever for testing.
        """
        if retriever is None:
            from .retriever import DummyRetriever
            retriever = DummyRetriever()
        self.retriever = retriever

    def evaluate(
        self,
        corpus: dict[str, dict],
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        top_k: int = 10,
    ) -> dict:
        """
        Run evaluation.

        Args:
            corpus: {doc_id: {"title": ..., "text": ...}}
            queries: {query_id: query_text}
            qrels: {query_id: {doc_id: relevance_score}}
            top_k: Number of results to retrieve per query.

        Returns:
            Metrics dict from compute_all_metrics().
        """
        logger.info("Running evaluation: %d queries, top_k=%d", len(queries), top_k)

        t0 = time.time()
        results = self.retriever.search(corpus, queries, top_k=top_k)
        elapsed = time.time() - t0
        logger.info("Search completed in %.1fs", elapsed)

        # Normalize chunk IDs: strip _v1, _v2 suffixes so that
        # Ch1_14 and Ch1_14_v1 are treated as the same chunk.
        import re as _re
        def _norm_id(cid: str) -> str:
            return _re.sub(r'_v\d+$', '', cid)

        # Convert BEIR result format {qid: {doc_id: score}} → ranked list
        retrieved_ranked: dict[str, list[str]] = {}
        for qid, scored in results.items():
            sorted_docs = sorted(scored.items(), key=lambda x: x[1], reverse=True)
            # Normalize + deduplicate (keep first occurrence = highest score)
            seen = set()
            ranked = []
            for doc_id, _ in sorted_docs:
                nid = _norm_id(doc_id)
                if nid not in seen:
                    seen.add(nid)
                    ranked.append(nid)
            retrieved_ranked[qid] = ranked

        # Build relevance dicts (also normalize)
        relevant_docs: dict[str, set[str]] = {
            qid: {_norm_id(doc_id) for doc_id in qrels.get(qid, {}).keys()}
            for qid in queries
        }

        metrics = compute_all_metrics(
            queries=relevant_docs,
            retrieved=retrieved_ranked,
            k_values=(1, 5, 10),
            relevance_grades=qrels,
        )

        metrics["search_time_s"] = round(elapsed, 1)
        metrics["queries_per_second"] = round(len(queries) / elapsed, 1) if elapsed > 0 else 0

        return metrics

    def compare_strategies(
        self,
        corpus: dict[str, dict],
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
        strategies: list[str] | None = None,
        top_k: int = 10,
        retriever_cls=None,   # Optional: custom retriever class
        retriever_kwargs: dict | None = None,  # Passed to retriever constructor
    ) -> dict[str, dict]:
        """
        Ablation study: compare multiple retrieval strategies.

        Args:
            corpus, queries, qrels: BEIR-format data.
            strategies: List of strategy names. Default: ["dense", "sparse", "hybrid"].
            top_k: Results per query.
            retriever_cls: Optional retriever class (default: ProjectRetriever).
            retriever_kwargs: Optional kwargs for the retriever constructor
                              (e.g. {"doc_filter": "beir:scifact"}).

        Returns:
            {strategy_name: metrics_dict}
        """
        if strategies is None:
            strategies = ["dense", "sparse", "hybrid"]

        if retriever_cls is None:
            from .retriever import ProjectRetriever
            retriever_cls = ProjectRetriever

        if retriever_kwargs is None:
            retriever_kwargs = {}

        results = {}
        for strategy in strategies:
            logger.info("Evaluating strategy: %s", strategy)
            self.retriever = retriever_cls(strategy=strategy, **retriever_kwargs)
            try:
                metrics = self.evaluate(corpus, queries, qrels, top_k)
                results[strategy] = metrics
            except Exception as e:
                logger.error("Strategy '%s' failed: %s", strategy, e)
                results[strategy] = {"error": str(e)}

        return results

    def format_comparison_report(
        self,
        comparison: dict[str, dict],
        output_path: str = "",
    ) -> str:
        """
        Generate a human-readable comparison report.

        Args:
            comparison: Output from compare_strategies().
            output_path: If provided, save JSON report to this path.

        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("RAG Retrieval Evaluation — Strategy Comparison")
        lines.append("=" * 70)

        # Collect all metric names
        metric_names = [
            "num_queries", "empty_results",
            "Recall@1", "Recall@5", "Recall@10",
            "Precision@1", "Precision@5", "Precision@10",
            "NDCG@5", "NDCG@10",
            "MRR", "MAP",
            "search_time_s",
        ]

        # Build table
        strategies = list(comparison.keys())
        header = f"{'Metric':<20}"
        for s in strategies:
            header += f" {s:>12}"
        lines.append(header)
        lines.append("-" * (20 + 13 * len(strategies)))

        for metric in metric_names:
            row = f"{metric:<20}"
            for s in strategies:
                val = comparison[s].get(metric, "N/A")
                if isinstance(val, float):
                    row += f" {val:>12.4f}"
                else:
                    row += f" {str(val):>12}"
            lines.append(row)

        lines.append("=" * (20 + 13 * len(strategies)))

        # Highlight best values
        lines.append("\nBest per metric:")
        for metric in metric_names:
            best_strategy = None
            best_val = None
            for s in strategies:
                val = comparison[s].get(metric)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if best_val is None or val > best_val:
                        best_val = val
                        best_strategy = s
            if best_strategy:
                lines.append(f"  {metric:<20} → {best_strategy} ({best_val:.4f})")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\n[OK] Report saved to: {output_path}")

        return report
