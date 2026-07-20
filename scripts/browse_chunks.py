#!/usr/bin/env python
"""
Browse indexed chunks to find chunk_ids for evaluation dataset.

Usage:
  # List all documents in the index
  python scripts/browse_chunks.py --list

  # Show all chunks for a document
  python scripts/browse_chunks.py --doc "电磁学.pdf"

  # Search for chunks containing keywords
  python scripts/browse_chunks.py --search "高斯定理"

  # Export all chunks for a doc to JSON (for building eval dataset)
  python scripts/browse_chunks.py --doc "电磁学.pdf" --export chunks_dump.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.context import init_context
from src.config import Configuration


def list_docs():
    """Show all indexed documents."""
    init_context(Configuration())
    ctx = __import__('src.context', fromlist=['get_context']).get_context()
    vs = ctx.vector_store

    doc_counts: dict[str, int] = {}
    for m in vs._all_metas:
        fn = m.get("doc_filename", "unknown")
        doc_counts[fn] = doc_counts.get(fn, 0) + 1

    print(f"\n{'Document':<40} {'Chunks':>8}")
    print("-" * 50)
    for fn, count in sorted(doc_counts.items()):
        print(f"{fn:<40} {count:>8}")
    print(f"\nTotal: {len(doc_counts)} documents, {sum(doc_counts.values())} chunks")


def show_doc_chunks(doc_name: str):
    """Show all chunks for a specific document."""
    init_context(Configuration())
    ctx = __import__('src.context', fromlist=['get_context']).get_context()
    vs = ctx.vector_store

    # Get real ChromaDB IDs by querying the collection
    try:
        all_data = vs._collection.get(include=["documents", "metadatas"])
        all_ids = all_data.get("ids", [])
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])
    except Exception:
        all_ids = [""] * len(vs._all_texts)
        all_docs = vs._all_texts
        all_metas = vs._all_metas

    chunks = []
    for i, (cid, text, meta) in enumerate(zip(all_ids, all_docs, all_metas)):
        if meta.get("doc_filename") == doc_name:
            chunks.append((cid, meta.get("chapter_title", ""), text))

    if not chunks:
        print(f"\nNo chunks found for '{doc_name}'")
        all_docs = sorted(set(m.get("doc_filename", "") for m in all_metas))
        print(f"Available documents: {all_docs}")
        return

    # Sort by chapter then chunk_id
    chunks.sort(key=lambda x: (x[1], x[0]))
    print(f"\nDocument: {doc_name}  ({len(chunks)} chunks)")
    print("=" * 70)

    for cid, chapter, text in chunks:
        preview = text[:120].replace("\n", " ")
        chapter_tag = f"[{chapter}]" if chapter else ""
        print(f"\n  chunk_id: {cid}")
        print(f"  chapter:  {chapter_tag}")
        print(f"  preview:  {preview}...")


def search_chunks(query: str):
    """Search chunks by keyword, show chunk_ids."""
    init_context(Configuration())
    ctx = __import__('src.context', fromlist=['get_context']).get_context()
    vs = ctx.vector_store

    results = vs._search_dense(query, top_k=15)
    if not results:
        print(f"\nNo results for '{query}'")
        return

    print(f"\nSearch: '{query}' → {len(results)} results")
    print("=" * 70)
    for i, r in enumerate(results):
        cid = r.get("chunk_id", "?")
        doc = r.get("doc_filename", "?")
        chapter = r.get("chapter_title", "")
        text = r.get("text", "")[:150].replace("\n", " ")
        chapter_tag = f" [{chapter}]" if chapter else ""
        print(f"\n  [{i + 1}] chunk_id: {cid}")
        print(f"       doc:      {doc}{chapter_tag}")
        print(f"       preview:  {text}...")


def export_doc(doc_name: str, output_path: str):
    """Export all chunks for a doc as JSON with REAL ChromaDB IDs."""
    init_context(Configuration())
    ctx = __import__('src.context', fromlist=['get_context']).get_context()
    vs = ctx.vector_store

    try:
        all_data = vs._collection.get(include=["documents", "metadatas"])
        all_ids = all_data.get("ids", [])
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])
    except Exception:
        all_ids = [f"chunk_{i}" for i in range(len(vs._all_texts))]
        all_docs = vs._all_texts
        all_metas = vs._all_metas

    chunks = []
    for cid, text, meta in zip(all_ids, all_docs, all_metas):
        if meta.get("doc_filename") == doc_name:
            chunks.append({
                "chunk_id": cid,
                "chapter": meta.get("chapter_title", ""),
                "text": text[:300],
            })

    chunks.sort(key=lambda x: (x["chapter"], x["chunk_id"]))
    output = Path(output_path)
    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nExported {len(chunks)} chunks to {output}")


def main():
    parser = argparse.ArgumentParser(description="Browse indexed chunks")
    parser.add_argument("--list", action="store_true", help="List all indexed documents")
    parser.add_argument("--doc", type=str, help="Show chunks for a document")
    parser.add_argument("--search", type=str, help="Search chunks by keyword")
    parser.add_argument("--export", type=str, help="Export chunks to JSON file (use with --doc)")
    args = parser.parse_args()

    if args.list:
        list_docs()
    elif args.doc and args.export:
        export_doc(args.doc, args.export)
    elif args.doc:
        show_doc_chunks(args.doc)
    elif args.search:
        search_chunks(args.search)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/browse_chunks.py --list")
        print("  python scripts/browse_chunks.py --doc '电磁学.pdf'")
        print("  python scripts/browse_chunks.py --search '高斯定理'")
        print("  python scripts/browse_chunks.py --doc '电磁学.pdf' --export chunks.json")


if __name__ == "__main__":
    main()
