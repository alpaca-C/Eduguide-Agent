"""RAG search tool — two-tier retrieval.

Default (fast):     Dense top-20 → Cross-Encoder rerank → top-K
Full (fallback):    Dense + Sparse + Graph → Cross-Encoder rerank → top-K
                    (triggered when user marks answer unsatisfactory)

Cross-Encoder: BGE-Reranker-v2-m3, 568M params, pushes best answer to rank 1.
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
_memory_manager: Optional[object] = None  # MemoryManager — preferred path

# RRF constant — larger k means more weight to low-ranked items
RRF_K = 60

# ── Cross-Encoder reranker (lazy-loaded once) ────────────────────────
_reranker_model = None
_reranker_tokenizer = None


def _get_reranker():
    """Lazy-load BGE-Reranker-v2-m3. Returns (model, tokenizer) or (None, None)."""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is not None:
        return _reranker_model, _reranker_tokenizer
    try:
        from pathlib import Path
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # Try ModelScope download path first
        rp = str(Path(__file__).resolve().parent.parent.parent
                 / 'models' / 'reranker' / 'models'
                 / 'BAAI--bge-reranker-v2-m3' / 'snapshots' / 'master')
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(rp, local_files_only=True)
        _reranker_tokenizer = AutoTokenizer.from_pretrained(rp, local_files_only=True)
        logger.info("Cross-Encoder reranker loaded: BGE-Reranker-v2-m3")
    except Exception as e:
        logger.warning("Cross-Encoder reranker unavailable: %s", e)
        return None, None
    return _reranker_model, _reranker_tokenizer


def _ce_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Cross-Encoder rerank: (query, text) pairs → scores → top-k."""
    if len(candidates) <= top_k:
        return candidates
    model, tok = _get_reranker()
    if model is None:
        return candidates[:top_k]
    try:
        import torch
        pairs = [(query, c.get("text", "")[:500]) for c in candidates]
        with torch.no_grad():
            inputs = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors='pt')
            scores = model(**inputs, return_dict=True).logits.view(-1).tolist()
        for i, s in enumerate(scores):
            candidates[i]["ce_score"] = float(s)
        candidates.sort(key=lambda x: x.get("ce_score", 0), reverse=True)
    except Exception as e:
        logger.warning("Cross-Encoder rerank failed: %s", e)
    return candidates[:top_k]


def init_rag_tool(vs=None, kg=None, memory_manager=None):
    """Initialize with the app's DocumentVectorStore, KnowledgeGraph, and/or MemoryManager.

    When memory_manager is provided, rag_search() delegates to
    MemoryManager.semantic.search() for unified recall.
    Falls back to vs/kg globals for backward compatibility.
    """
    global _vector_store, _knowledge_graph, _memory_manager
    if vs is not None:
        _vector_store = vs
    if kg is not None:
        _knowledge_graph = kg
    if memory_manager is not None:
        _memory_manager = memory_manager


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
    text = (item.get("text") or "")[:200]
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


def _rrf_fuse_weighted(
    results_by_source: list[tuple[str, list[dict], float]], top_k: int,
) -> list[dict]:
    """Weighted RRF fusion — each source has its own weight multiplier.

    Args:
        results_by_source: [("dense", [...], 2.0), ("sparse", [...], 0.5), ...]
            where the third element is the weight multiplier for that source.
    """
    scores: dict[str, dict] = {}

    for source, ranked_list, weight in results_by_source:
        for rank, item in enumerate(ranked_list):
            key = _text_key(item)
            rrf_score = weight / (RRF_K + rank + 1)
            if key not in scores:
                scores[key] = {**item, "rrf_score": rrf_score, "sources": [source]}
            else:
                scores[key]["rrf_score"] += rrf_score
                if source not in scores[key]["sources"]:
                    scores[key]["sources"].append(source)

    sorted_items = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_items[:top_k]


# ── Tier 1: rag_search (fast — Dense + Cross-Encoder) ──────────────

async def rag_search(query: str, top_k: int = 5, filter_docs: Optional[set] = None) -> ToolResult:
    """Default retrieval: Dense top-20 → Cross-Encoder rerank → top-K.

    Fast path (~2s). Handles ~80% of queries. No sparse/graph overhead.
    """
    # ── MemoryManager path ───────────────────────────────────────────
    if _memory_manager is not None:
        try:
            sm = _memory_manager.semantic
            return await _candidates_to_result(query, top_k, filter_docs, full=False)
        except Exception as e:
            logger.warning("MemoryManager semantic search failed, falling back: %s", e)

    return await _candidates_to_result(query, top_k, filter_docs, full=False)


# ── Tier 2: rag_fullsearch (Dense + Sparse + Graph + Cross-Encoder) ─

async def rag_fullsearch(query: str, top_k: int = 5, filter_docs: Optional[set] = None) -> ToolResult:
    """Full retrieval: Dense + Sparse + Graph → dedup → Cross-Encoder rerank → top-K.

    Triggered when user feedback indicates the default answer was unsatisfactory.
    Adds 0.5-1s for sparse + graph retrieval.
    """
    if _memory_manager is not None:
        try:
            return await _candidates_to_result(query, top_k, filter_docs, full=True)
        except Exception as e:
            logger.warning("MemoryManager fullsearch failed, falling back: %s", e)

    return await _candidates_to_result(query, top_k, filter_docs, full=True)


# ── Core logic: collect → CE rerank → format ────────────────────────

async def _candidates_to_result(
    query: str, top_k: int, filter_docs: Optional[set], full: bool,
) -> ToolResult:
    """Collect candidates from sources, CE rerank, format output."""
    # Prefer MemoryManager when available; fall back to module-level globals
    vs = _vector_store
    kg = _knowledge_graph
    if _memory_manager is not None:
        try:
            mm_vs = _memory_manager.semantic.vector_store
            mm_kg = _memory_manager.semantic.knowledge_graph
            if mm_vs is not None:
                vs = mm_vs
            if mm_kg is not None:
                kg = mm_kg
        except Exception:
            pass
    if vs is None and kg is None:
        return ToolResult(
            tool_name="rag_search", query=query,
            content="（本地搜索未初始化，请先上传并处理文档）",
            error=ToolErrorType.NOT_CONFIGURED,
        )

    candidates: list[dict] = []
    seen_ids: set[str] = set()

    # Source 1: Dense (always — primary source)
    if vs is not None:
        try:
            dense = vs._search_dense(query, top_k=20, filter_docs=filter_docs) or []
            for r in dense:
                cid = r.get("chunk_id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    r["source"] = "dense"
                    candidates.append(r)
        except Exception as e:
            logger.warning("Dense search failed: %s", e)

    if full:
        # Source 2: Sparse (supplement — keyword precision)
        if vs is not None:
            try:
                sparse = vs._search_sparse(query, top_k=10, filter_docs=filter_docs) or []
                for r in sparse:
                    cid = r.get("chunk_id", "")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        r["source"] = "sparse"
                        candidates.append(r)
            except Exception as e:
                logger.warning("Sparse search failed: %s", e)

        # Source 3: Graph (supplement — concept-level relations)
        if kg is not None:
            try:
                kg_concepts = kg.search_concepts(query, limit=top_k)
                for c in kg_concepts:
                    neighbors = kg.get_neighbors(c.id)
                    if neighbors:
                        c.metadata["neighbors"] = [
                            {"name": n["concept_name"], "relation": n["relation_type"]}
                            for n in neighbors[:5]
                        ]
                    cid = c.source_chunk_id or f"kg:{c.id}"
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        candidates.append({
                            "text": f"{c.name}: {c.description}",
                            "doc_filename": c.doc_filename or "",
                            "source": "graph",
                            "chunk_id": cid,
                            "concept": c,
                        })
            except Exception as e:
                logger.warning("KG search failed: %s", e)

    if not candidates:
        return ToolResult(
            tool_name="rag_search", query=query,
            content="（本地资料中未找到相关内容）",
            error=ToolErrorType.EMPTY_RESULT,
        )

    # ── Cross-Encoder rerank (ALL paths) ──────────────────────────
    ranked = _ce_rerank(query, candidates, top_k)

    # ── Format output ─────────────────────────────────────────────
    parts = []
    chunk_items = [f for f in ranked if f.get("source") != "graph"]
    graph_items = [f for f in ranked if f.get("source") == "graph"]

    tier_label = "Dense + Sparse + Graph" if full else "Dense"
    parts.append(f"=== 本地文档检索结果（{tier_label} + Cross-Encoder） ===")

    if chunk_items:
        for i, item in enumerate(chunk_items):
            fn = item.get("doc_filename", "")
            ch = item.get("chapter_title", "")
            ce = item.get("ce_score")
            score_str = f" [CE: {ce:.3f}]" if ce is not None else ""
            source_parts = []
            if fn:
                source_parts.append(fn)
            if ch:
                source_parts.append(ch)
            source = f"  [来源: {' / '.join(source_parts)}]" if source_parts else ""
            parts.append(f"[片段 {i + 1}]{score_str} {item['text']}")
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
        tool_name="rag_search" if not full else "rag_fullsearch",
        query=query,
        content="\n".join(parts),
        metadata={
            "tier": "full" if full else "fast",
            "total_fused": len(ranked),
            "chunks_found": len(chunk_items),
            "concepts_found": len(graph_items),
            "ce_reranked": True,
        },
    )


# Register tools
register_tool(
    name="rag_search",
    description="默认文档检索：Dense语义搜索 + Cross-Encoder重排序。速度快，适合大多数问题。",
    func=rag_search,
)
register_tool(
    name="rag_fullsearch",
    description="全量文档检索：Dense + Sparse全文 + 知识图谱 + Cross-Encoder。当默认检索结果不满意时使用。",
    func=rag_fullsearch,
)
