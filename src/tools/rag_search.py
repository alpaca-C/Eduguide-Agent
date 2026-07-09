"""RAG search tool — 3-source hybrid retrieval with RRF fusion.

Sources:
  1. Dense:  ChromaDB + Qwen3-Embedding  (semantic, cross-lingual)
  2. Sparse: SQLite FTS5 + BM25          (keyword, exact match)
  3. Graph:  SQLite KnowledgeGraph       (concepts + 1-hop neighbors)

Fusion: RRF (Reciprocal Rank Fusion) with k=60.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from . import ToolResult, ToolErrorType, register_tool

logger = logging.getLogger(__name__)

# Lazily set by the app
_vector_store: Optional[object] = None
_knowledge_graph: Optional[object] = None

# RRF constant — larger k means more weight to low-ranked items
RRF_K = 60


def init_rag_tool(vs, kg):
    """Initialize with the app's DocumentVectorStore and KnowledgeGraph instances."""
    global _vector_store, _knowledge_graph
    _vector_store = vs
    _knowledge_graph = kg


def get_doc_names() -> list[str]:
    names = set()
    if _vector_store is not None:
        try:
            names.update(_vector_store.get_doc_names())
        except Exception as e:
            logger.debug("Failed to get doc names from vector store: %s", e)
    if _knowledge_graph is not None:
        try:
            names.update(_knowledge_graph.get_doc_names())
        except Exception as e:
            logger.debug("Failed to get doc names from knowledge graph: %s", e)
    return sorted(names)


# ── RRF Fusion ──────────────────────────────────────────────────────

def _text_key(item: dict) -> str:
    """Stable dedup key for a chunk, independent of source."""
    text = item.get("text", "")[:200]
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _rrf_fuse(results_by_source: list[tuple[str, list[dict]]], top_k: int) -> list[dict]:
    """Fuse ranked lists from multiple sources via RRF.

    Args:
        results_by_source: [("dense", [...ranked...]), ("sparse", [...]), ("graph", [...])]
        top_k: Number of results to return.

    Returns deduplicated, RRF-scored list sorted by score descending.
    """
    scores: dict[str, dict] = {}  # key → {item, score}

    for source, ranked_list in results_by_source:
        for rank, item in enumerate(ranked_list):
            key = _text_key(item)
            rrf_score = 1.0 / (RRF_K + rank + 1)
            if key not in scores:
                scores[key] = {**item, "rrf_score": rrf_score, "sources": [source]}
            else:
                scores[key]["rrf_score"] += rrf_score
                if source not in scores[key]["sources"]:
                    scores[key]["sources"].append(source)

    # Sort by RRF score descending
    sorted_items = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_items[:top_k]


# ── Main search ─────────────────────────────────────────────────────

async def rag_search(query: str, top_k: int = 5, filter_docs: Optional[set] = None) -> ToolResult:
    """Hybrid search with RRF fusion across dense + sparse + graph sources."""
    results_by_source: list[tuple[str, list[dict]]] = []

    # Guard
    if _vector_store is None and _knowledge_graph is None:
        return ToolResult(
            tool_name="rag_search", query=query,
            content="（本地搜索未初始化，请先上传并处理文档）",
            error=ToolErrorType.NOT_CONFIGURED,
        )

    # ── Source 1+2: Dense + Sparse from vector store ─────────────
    if _vector_store is not None:
        try:
            # Use hybrid search for both dense and sparse in one call
            hybrid = _vector_store.search_hybrid(query, top_k=top_k * 2, filter_docs=filter_docs)
            if hybrid.get("dense"):
                results_by_source.append(("dense", hybrid["dense"]))
            if hybrid.get("sparse"):
                results_by_source.append(("sparse", hybrid["sparse"]))
        except AttributeError:
            # Fallback for old vector store without search_hybrid
            try:
                dense_results = _vector_store.search(query, top_k=top_k * 2, filter_docs=filter_docs)
                if dense_results:
                    results_by_source.append(("dense", dense_results))
            except Exception as e:
                logger.warning("Dense search failed: %s", e)

    # ── Source 3: Knowledge Graph ───────────────────────────────
    kg_concepts = []
    if _knowledge_graph is not None:
        try:
            if filter_docs:
                kg_concepts = _knowledge_graph.search_concepts_by_docs(query, filter_docs, limit=top_k)
            else:
                kg_concepts = _knowledge_graph.search_concepts(query, limit=top_k)
            for c in kg_concepts:
                neighbors = _knowledge_graph.get_neighbors(c.id)
                if neighbors:
                    c.metadata["neighbors"] = [
                        {"name": n["concept_name"], "relation": n["relation_type"]}
                        for n in neighbors[:5]
                    ]
            # Format KG results as chunk-like dicts for RRF fusion
            graph_items = []
            for c in kg_concepts:
                graph_items.append({
                    "text": f"{c.name}: {c.description}",
                    "doc_filename": c.doc_filename or "",
                    "chapter_title": "",
                    "source": "graph",
                    "concept": c,
                })
            results_by_source.append(("graph", graph_items))
        except Exception as e:
            logger.warning("KG search failed: %s", e)

    # ── RRF Fusion ──────────────────────────────────────────────
    if not results_by_source:
        return ToolResult(
            tool_name="rag_search", query=query,
            content="（本地资料中未找到相关内容）",
            error=ToolErrorType.EMPTY_RESULT,
        )

    fused = _rrf_fuse(results_by_source, top_k)

    # ── Format output ───────────────────────────────────────────
    parts = []
    chunk_items = [f for f in fused if f.get("source") != "graph"]
    graph_items = [f for f in fused if f.get("source") == "graph"]

    if chunk_items:
        parts.append("=== 本地文档检索结果 ===")
        for i, item in enumerate(chunk_items):
            fn = item.get("doc_filename", "")
            ch = item.get("chapter_title", "")
            src_label = f" [{', '.join(item.get('sources', []))}]"
            source_parts = []
            if fn:
                source_parts.append(fn)
            if ch:
                source_parts.append(ch)
            source = f"  [来源: {' / '.join(source_parts)}]" if source_parts else ""
            parts.append(f"[片段 {i + 1}]{src_label} {item['text']}")
            if source:
                parts.append(source)

    if graph_items:
        parts.append("\n=== 知识图谱关联概念 ===")
        for item in graph_items:
            c = item.get("concept")
            if c:
                source_info = f" [来源: {c.doc_filename}]" if c.doc_filename else ""
                parts.append(f"- **{c.name}** ({c.category}){source_info}: {c.description[:150]}")
                neighbors = c.metadata.get("neighbors", [])
                if neighbors:
                    neighbor_str = ", ".join(f"{n['name']}({n['relation']})" for n in neighbors)
                    parts.append(f"  关联: {neighbor_str}")

    return ToolResult(
        tool_name="rag_search",
        query=query,
        content="\n".join(parts),
        metadata={
            "total_fused": len(fused),
            "chunks_found": len(chunk_items),
            "concepts_found": len(graph_items),
        },
    )


# Register the tool
register_tool(
    name="rag_search",
    description="搜索本地已上传的文档资料和知识图谱。当用户问题涉及已上传的教材、论文、笔记等本地资料时使用。",
    func=rag_search,
)
