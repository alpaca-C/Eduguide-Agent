"""
RAG Evaluation Module

Two-layer evaluation:
  1. Official benchmarks (BEIR) — proves the retrieval pipeline isn't a toy
  2. Custom annotated dataset — proves the system works on real教材questions

Usage:
  # Run BEIR benchmark
  python scripts/run_beir_eval.py --dataset nfcorpus --top-k 10

  # Run custom evaluation
  python scripts/run_beir_eval.py --custom data/eval/custom_questions.json
"""

from .metrics import compute_all_metrics
from .evaluator import RetrievalEvaluator
