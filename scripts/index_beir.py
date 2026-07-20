#!/usr/bin/env python
"""
Index a BEIR dataset into the project's ChromaDB + FTS5.

This makes the BEIR corpus searchable by the project's real retriever,
enabling meaningful ablation experiments.

Usage:
  # Index scifact (5183 short docs, ~30s)
  python scripts/index_beir.py --dataset scifact

  # Index nfcorpus (3633 short docs, ~20s)
  python scripts/index_beir.py --dataset nfcorpus

  # Skip KG extraction (default — LLM calls too expensive for benchmarks)
  python scripts/index_beir.py --dataset scifact --skip-kg

  # Force re-index (clear existing first)
  python scripts/index_beir.py --dataset scifact --clear
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index_beir")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def index_beir(
    dataset: str = "scifact",
    batch_size: int = 200,
    clear: bool = False,
    max_docs: int = 0,          # 0 = all docs
) -> dict:
    """
    Load a BEIR corpus, chunk it, and index into ChromaDB + FTS5.

    BEIR documents are already short (abstracts), so each doc = one chunk.
    chunk_id = BEIR doc_id — this mapping is critical for evaluation.
    """
    from src.evaluation.datasets import load_beir_dataset
    from src.context import init_context, get_context
    from src.config import Configuration
    from src.documents.chunker import TextChunk

    t0 = time.time()

    # ── 1. Initialize project context ──────────────────────────
    config = Configuration()
    config.chunk_size = 2000   # BEIR docs are short, keep as single chunks
    config.chunk_overlap = 0

    logger.info("Initializing app context...")
    ctx = init_context(config)
    vs = ctx.vector_store
    logger.info("Vector store ready: %d existing chunks", len(vs._all_texts))

    # ── 2. Load BEIR corpus ────────────────────────────────────
    logger.info("Loading BEIR dataset: %s", dataset)
    corpus, queries, qrels = load_beir_dataset(name=dataset, split="test")

    # ── 3. Detect dimension mismatch & clear if needed ─────────
    # If the existing collection uses a different embedding model (dim mismatch),
    # indexing will fail. Detect this early.
    try:
        existing_count = vs._collection.count()
        if existing_count > 0:
            # Check a sample embedding to verify dimensions
            test_vec = vs._ef(["test"])[0]
            logger.info("Existing collection: %d chunks, embedding dim=%d",
                         existing_count, len(test_vec))
    except Exception as e:
        if "dimension" in str(e).lower():
            logger.warning("Dimension mismatch detected! Use --clear to rebuild.")
            logger.warning("  python scripts/index_beir.py --dataset %s --clear", dataset)
            raise

    if clear:
        logger.info("Clearing existing index...")
        vs.clear()
        ctx.knowledge_graph.clear()

    # ── 4. Convert BEIR docs → TextChunks ──────────────────────
    # Each BEIR doc becomes one chunk. chunk_id = BEIR doc_id.
    # This is the mapping that lets evaluation metrics work.

    # Apply max_docs limit for quick testing
    doc_items = list(corpus.items())
    if max_docs and max_docs < len(doc_items):
        doc_items = doc_items[:max_docs]
        logger.info("Limited to %d/%d docs (--max-docs)", max_docs, len(corpus))

    chunks: list[TextChunk] = []
    skipped = 0
    for doc_id, doc in doc_items:
        title = doc.get("title", "")
        text = doc.get("text", "")
        combined = f"{title}\n\n{text}" if title else text

        if not combined.strip():
            skipped += 1
            continue

        # Trim to chunk_size to avoid unnecessary splits
        if len(combined) > config.chunk_size:
            combined = combined[:config.chunk_size]

        chunks.append(TextChunk(
            chunk_id=doc_id,            # ← critical: BEIR doc_id = chunk_id
            text=combined,
            doc_filename=f"beir:{dataset}",
            chapter_title="",
            chunk_index=len(chunks),
        ))

    logger.info("Converted %d docs → %d chunks (%d empty skipped)",
                 len(doc_items), len(chunks), skipped)

    # ── 5. Index into ChromaDB + FTS5 ─────────────────────────
    logger.info("Indexing %d chunks (batch_size=%d)...", len(chunks), batch_size)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_dicts = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "doc_filename": c.doc_filename,
                "chapter_title": c.chapter_title,
                "chunk_index": c.chunk_index,
            }
            for c in batch
        ]
        vs.index_chunks(batch_dicts)

        if (i // batch_size) % 10 == 0 or i + batch_size >= len(chunks):
            logger.info("  Indexed %d/%d chunks", min(i + batch_size, len(chunks)), len(chunks))

    elapsed = time.time() - t0
    logger.info("Indexing complete: %d chunks in %.1fs (%.1f docs/s)",
                 len(chunks), elapsed, len(chunks) / elapsed if elapsed > 0 else 0)

    # ── 6. Verify ─────────────────────────────────────────────
    # Test search to confirm indexing worked
    sample_query = list(queries.values())[0] if queries else "test"
    try:
        results = vs._search_dense(sample_query, top_k=3)
        logger.info("Verification search: '%s' → %d results", sample_query[:60], len(results))
    except Exception as e:
        logger.warning("Verification search failed: %s", e)

    return {
        "dataset": dataset,
        "docs_total": len(corpus),
        "chunks_indexed": len(chunks),
        "skipped_empty": skipped,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Index a BEIR dataset into ChromaDB + FTS5")
    parser.add_argument("--dataset", type=str, default="scifact",
                        help="BEIR dataset name (scifact, nfcorpus)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Chunks per batch (default: 200)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing index before importing")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="Only index first N docs (0=all, for quick tests)")

    args = parser.parse_args()

    if args.max_docs:
        print(f"\nIndexing BEIR/{args.dataset} — first {args.max_docs} docs (quick mode)\n")
    else:
        print(f"\nIndexing BEIR/{args.dataset} into ChromaDB + FTS5\n")

    try:
        result = index_beir(
            dataset=args.dataset,
            batch_size=args.batch_size,
            clear=args.clear,
            max_docs=args.max_docs,
        )
        print(f"\nDone: {result['chunks_indexed']} chunks indexed in {result['elapsed_s']}s")
        print(f"Ready to evaluate:")
        print(f"  python scripts/run_beir_eval.py --dataset {args.dataset} --top-k 10")
        print(f"  python scripts/run_beir_eval.py --dataset {args.dataset} --compare")
    except Exception as e:
        logger.error("Indexing failed: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
