"""
Re-extract Knowledge Graph from existing vector store chunks.
Uses the FIXED extractor (with source_fragment support).

This avoids re-parsing PDFs and re-chunking — only the KG step is re-run.

Usage:
  python scripts/re_extract_kg.py           # re-extract all docs
  python scripts/re_extract_kg.py --dry-run # check what would happen
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("re_extract_kg")


def main():
    parser = argparse.ArgumentParser(description="Re-extract KG from existing vector store chunks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    args = parser.parse_args()

    # ── Init context ──
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from src.context import init_context
    from src.config import Configuration
    init_context(Configuration())

    from src.context import get_context
    ctx = get_context()
    vs = ctx.vector_store
    kg = ctx.knowledge_graph

    # ── Load all chunks from vector store ──
    all_data = vs._collection.get(include=['documents', 'metadatas'])
    total_chunks = len(all_data['ids'])
    logger.info("Loaded %d chunks from ChromaDB", total_chunks)

    # Group chunks by document
    from collections import defaultdict
    doc_chunks: dict[str, list[dict]] = defaultdict(list)
    for cid, text, meta in zip(all_data['ids'], all_data['documents'], all_data['metadatas']):
        doc_name = meta.get('doc_filename', 'unknown')
        chapter_title = meta.get('chapter_title', '')
        # Parse chunk_index from chunk_id (format: "filename_N" or "filename_N_vM")
        chunk_index = meta.get('chunk_index', 0)
        doc_chunks[doc_name].append({
            'chunk_id': cid,
            'text': text,
            'doc_filename': doc_name,
            'chapter_title': chapter_title,
            'chunk_index': chunk_index,
        })

    logger.info("Documents found: %d", len(doc_chunks))
    for doc_name, chunks in sorted(doc_chunks.items()):
        logger.info("  %s: %d chunks", doc_name, len(chunks))

    # ── Check current KG state ──
    stats_before = kg.stats()
    logger.info("KG before re-extraction: %d concepts, %d relations", stats_before['concepts'], stats_before['relations'])

    # Check current source_chunk_id diversity
    from collections import Counter
    all_concepts_before = kg.get_all_concepts()
    sid_counter_before = Counter(c.source_chunk_id for c in all_concepts_before)
    logger.info("Current source_chunk_id distribution:")
    for sid, count in sid_counter_before.most_common():
        logger.info("  [%s]: %d concepts", sid, count)

    if args.dry_run:
        logger.info("DRY RUN — no changes made.")
        return 0

    # ── Re-extract KG per document ──
    from src.documents.chunker import TextChunk
    from src.agents.extractor import extract_full_document
    from src.config import Configuration as Cfg

    config = Cfg.from_env()

    for doc_name, chunk_dicts in sorted(doc_chunks.items()):
        # Skip BEIR datasets (indexed for evaluation, not KG extraction)
        if doc_name.startswith("beir:") or doc_name.startswith("beir"):
            logger.info("Skipping BEIR dataset: %s (%d chunks)", doc_name, len(chunk_dicts))
            continue
        logger.info("=" * 60)
        logger.info("Re-extracting KG for: %s (%d chunks)", doc_name, len(chunk_dicts))

        # Convert dicts back to TextChunk objects
        chunks = []
        for i, cd in enumerate(chunk_dicts):
            chunks.append(TextChunk(
                chunk_id=cd['chunk_id'],
                text=cd['text'],
                doc_filename=cd['doc_filename'],
                chunk_index=cd['chunk_index'],
                chapter_title=cd.get('chapter_title', ''),
            ))

        # Remove old KG data for this document
        removed = kg.remove_by_doc(doc_name)
        logger.info("  Removed %d old concepts for '%s'", removed, doc_name)

        # Re-extract
        t0 = time.time()
        result = extract_full_document(chunks, config, kg)
        elapsed = time.time() - t0
        logger.info("  Extracted: %d concepts, %d relations in %.1fs",
                     result['concepts_extracted'], result['relations_extracted'], elapsed)

    # ── Verify ──
    stats_after = kg.stats()
    logger.info("=" * 60)
    logger.info("KG after re-extraction: %d concepts, %d relations", stats_after['concepts'], stats_after['relations'])

    all_concepts_after = kg.get_all_concepts()
    sid_counter_after = Counter(c.source_chunk_id for c in all_concepts_after)
    logger.info("New source_chunk_id distribution (top 20):")
    for sid, count in sid_counter_after.most_common(20):
        logger.info("  [%s]: %d concepts", sid, count)

    unique_sids = len(sid_counter_after)
    empty_sids = sid_counter_after.get("", 0)
    logger.info("Unique source_chunk_ids: %d (was %d before)", unique_sids, len(sid_counter_before))
    logger.info("Empty source_chunk_ids: %d", empty_sids)

    if unique_sids > len(doc_chunks):
        logger.info("SUCCESS: source_chunk_ids are now diverse — KG should work in ablation!")
    else:
        logger.warning("source_chunk_ids still concentrated — may still have issues")

    return 0


if __name__ == "__main__":
    sys.exit(main())
