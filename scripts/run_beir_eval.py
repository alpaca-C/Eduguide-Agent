#!/usr/bin/env python
"""
BEIR benchmark evaluation runner.

Usage:
  # Test the eval pipeline with a dummy retriever (no project context needed)
  python scripts/run_beir_eval.py --dummy

  # Run on a BEIR dataset (requires BEIR data download)
  python scripts/run_beir_eval.py --dataset nfcorpus --top-k 10

  # Compare retrieval strategies (ablation study)
  python scripts/run_beir_eval.py --dataset nfcorpus --compare

  # Run on a custom dataset
  python scripts/run_beir_eval.py --custom data/eval/sample_questions.json

  # Generate sample custom dataset template
  python scripts/run_beir_eval.py --create-sample
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("beir_eval")


def run_beir_benchmark(
    dataset: str = "nfcorpus",
    top_k: int = 10,
    compare: bool = False,
    output: str = "",
):
    """Run a BEIR benchmark evaluation.

    Uses ProjectRetriever with doc_filter — requires the BEIR dataset to be
    pre-indexed into ChromaDB via `scripts/index_beir.py`.
    """
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from src.context import init_context
    from src.config import Configuration
    init_context(Configuration())

    from src.evaluation.datasets import load_beir_dataset
    from src.evaluation.evaluator import RetrievalEvaluator
    from src.evaluation.retriever import ProjectRetriever

    # Load BEIR data
    print(f"\nLoading BEIR dataset: {dataset}")
    try:
        corpus, queries, qrels = load_beir_dataset(name=dataset, split="test")
    except FileNotFoundError:
        print(f"\n[ERROR] BEIR dataset '{dataset}' not found locally.")
        return 1

    # Check if BEIR data is indexed — if not, auto-index
    beir_doc_filter = f"beir:{dataset}"
    from src.context import get_context
    ctx = get_context()
    if not ctx.vector_store.has_document(beir_doc_filter):
        print(f"\nBEIR/{dataset} is NOT indexed yet. Auto-indexing...")
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.index_beir import index_beir
        result = index_beir(dataset=dataset)
        print(f"Indexed: {result['chunks_indexed']} chunks")
    else:
        print(f"BEIR/{dataset} already indexed")

    evaluator = RetrievalEvaluator()

    if compare:
        # 4-way ablation: dummy, dense, sparse, hybrid
        from src.evaluation.retriever import DummyRetriever
        print(f"Running 4-way ablation: dummy vs dense vs sparse vs hybrid (BEIR/{dataset})\n")

        # Strategy 0: dummy (Jaccard text overlap — true baseline)
        logger.info("Evaluating strategy: dummy")
        evaluator.retriever = DummyRetriever()
        dummy_metrics = evaluator.evaluate(corpus, queries, qrels, top_k=top_k)

        # Strategies 1-3: ProjectRetriever with doc_filter
        comparison = evaluator.compare_strategies(
            corpus, queries, qrels,
            strategies=["dense", "sparse", "hybrid"],
            top_k=top_k,
            retriever_cls=ProjectRetriever,
            retriever_kwargs={"doc_filter": beir_doc_filter},
        )
        comparison["dummy"] = dummy_metrics

        # Reorder: dummy, dense, sparse, hybrid
        ordered = {k: comparison[k] for k in ["dummy", "dense", "sparse", "hybrid"]}
        report = evaluator.format_comparison_report(ordered, output_path=output)
        print(report)
    else:
        # Single strategy
        retriever = ProjectRetriever(strategy="hybrid", doc_filter=beir_doc_filter)
        evaluator.retriever = retriever

        print(f"Running evaluation: hybrid, top_k={top_k}\n")
        metrics = evaluator.evaluate(corpus, queries, qrels, top_k=top_k)

        from src.evaluation.metrics import format_metrics_table
        print(format_metrics_table(metrics))

    return 0


def run_dummy_test():
    """Test the evaluation pipeline with a dummy retriever."""
    from src.evaluation.datasets import load_beir_dataset
    from src.evaluation.evaluator import RetrievalEvaluator
    from src.evaluation.retriever import DummyRetriever

    print("\n[Dummy mode] Testing evaluation pipeline...")

    # Try to load a small BEIR dataset
    try:
        corpus, queries, qrels = load_beir_dataset(name="scifact", split="test")
    except (FileNotFoundError, Exception):
        # Fallback: build a tiny synthetic dataset
        print("BEIR data not available — using synthetic dataset")
        corpus = {
            "d1": {"title": "Python async", "text": "asyncio is a library for writing concurrent code using async await syntax"},
            "d2": {"title": "Java threads", "text": "Java provides Thread class and ExecutorService for concurrent programming"},
            "d3": {"title": "Python GIL", "text": "The Global Interpreter Lock prevents multiple threads from executing Python bytecode at once"},
            "d4": {"title": "FastAPI", "text": "FastAPI is a modern web framework for building APIs with Python based on Starlette"},
            "d5": {"title": "Spring Boot", "text": "Spring Boot makes it easy to create stand-alone production-grade Spring applications"},
        }
        queries = {
            "q1": "Python concurrent programming",
            "q2": "Java web framework",
            "q3": "Python web API",
        }
        qrels = {
            "q1": {"d1": 2, "d3": 3},  # d3 is more relevant
            "q2": {"d5": 3},
            "q3": {"d4": 3, "d1": 1},
        }

    evaluator = RetrievalEvaluator(DummyRetriever())
    metrics = evaluator.evaluate(corpus, queries, qrels, top_k=5)

    from src.evaluation.metrics import format_metrics_table
    print(format_metrics_table(metrics))
    return 0


def run_custom_eval(dataset_path: str, top_k: int = 10):
    """Run evaluation on a custom annotated dataset."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from src.context import init_context
    from src.config import Configuration
    init_context(Configuration())

    from src.evaluation.datasets import load_custom_dataset
    from src.evaluation.evaluator import RetrievalEvaluator
    from src.evaluation.retriever import ProjectRetriever, DummyRetriever

    print(f"\nLoading custom dataset: {dataset_path}")
    corpus, queries, qrels = load_custom_dataset(dataset_path)

    if not queries:
        print("[ERROR] No queries found in dataset")
        return 1

    evaluator = RetrievalEvaluator()

    # Ablation: compare dense vs sparse vs graph vs hybrid
    comparison = evaluator.compare_strategies(
        corpus, queries, qrels,
        strategies=["dense", "sparse", "graph"],
        top_k=top_k,
    )
    report = evaluator.format_comparison_report(comparison)
    print(report)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RAG Retrieval Evaluation — BEIR benchmarks & custom datasets",
    )
    parser.add_argument("--dataset", type=str, default="",
                        help="BEIR dataset name (nfcorpus, scifact, fiqa, ...)")
    parser.add_argument("--dummy", action="store_true",
                        help="Test eval pipeline with a dummy retriever")
    parser.add_argument("--compare", action="store_true",
                        help="Run ablation: dense vs sparse vs hybrid")
    parser.add_argument("--custom", type=str, default="",
                        help="Path to custom JSON dataset")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results per query (default: 10)")
    parser.add_argument("--output", type=str, default="",
                        help="Save report JSON to file")
    parser.add_argument("--create-sample", action="store_true",
                        help="Generate sample custom dataset template")

    args = parser.parse_args()

    if args.create_sample:
        from src.evaluation.datasets import create_sample_dataset
        create_sample_dataset()
        return 0

    if args.dummy:
        return run_dummy_test()

    if args.custom:
        return run_custom_eval(args.custom, args.top_k)

    if args.dataset:
        return run_beir_benchmark(
            dataset=args.dataset,
            top_k=args.top_k,
            compare=args.compare,
            output=args.output,
        )

    parser.print_help()
    print("\nQuick start:")
    print("  python scripts/run_beir_eval.py --create-sample  # generate dataset template")
    print("  python scripts/run_beir_eval.py --dummy           # test pipeline")
    print("  python scripts/run_beir_eval.py --dataset nfcorpus # real benchmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
