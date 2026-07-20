# SemanticMemory — document knowledge (vector store + knowledge graph)
#
# Wraps DocumentVectorStore (ChromaDB + FTS5) and KnowledgeGraph (SQLite concepts)
# under a unified "语义记忆" abstraction with RRF (Reciprocal Rank Fusion).
#
# Usage:
#   sm = SemanticMemory(vector_store, knowledge_graph)
#   result = await sm.search("量子力学是什么", filter_docs={"physics.pdf"})
#   sm.index_chunks(chunks, doc_id="physics.pdf")
#   concepts = sm.search_concepts("量子")

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from ..tools import ToolResult, ToolErrorType

logger = logging.getLogger(__name__)

# RRF constant — larger k means more weight to low-ranked items
RRF_K = 60


# ── RRF helpers ──────────────────────────────────────────────────────────

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

    sorted_items = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_items[:top_k]


# ── SemanticMemory ────────────────────────────────────────────────────────

class SemanticMemory:
    """语义记忆：向量存储 + 知识图谱。

    Provides a unified search() that fuses results from all three sources:
      1. Dense:  ChromaDB + Qwen3-Embedding (semantic)
      2. Sparse: SQLite FTS5 + BM25 (keyword)
      3. Graph:  SQLite KnowledgeGraph (concepts + 1-hop neighbors)

    Combined via RRF (Reciprocal Rank Fusion) with k=60.
    """

    def __init__(self, vector_store, knowledge_graph):
        """Wrap existing DocumentVectorStore and KnowledgeGraph instances.

        Args:
            vector_store: DocumentVectorStore from src.agents.qa.vector_store.
            knowledge_graph: KnowledgeGraph from src.knowledge.graph.
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph

    # ── Unified search ──────────────────────────────────────────────────

    async def search(
        self, query: str, top_k: int = 5,
        filter_docs: set[str] | None = None,
    ) -> ToolResult:
        """Hybrid search with RRF fusion across dense + sparse + graph sources.

        This is the semantic search entry point — mirrors rag_search() in
        src/tools/rag_search.py, but as a method on SemanticMemory.
        """
        vs = self.vector_store
        kg = self.knowledge_graph
        results_by_source: list[tuple[str, list[dict]]] = []

        # Guard
        if vs is None and kg is None:
            return ToolResult(
                tool_name="semantic_search", query=query,
                content="（本地搜索未初始化，请先上传并处理文档）",
                error=ToolErrorType.NOT_CONFIGURED,
            )

        # ── Source 1+2: Dense + Sparse from vector store ─────────────
        if vs is not None:
            try:
                hybrid = vs.search_hybrid(query, top_k=top_k * 2, filter_docs=filter_docs)
                if hybrid.get("dense"):
                    results_by_source.append(("dense", hybrid["dense"]))
                if hybrid.get("sparse"):
                    results_by_source.append(("sparse", hybrid["sparse"]))
            except AttributeError:
                # Fallback for old vector store without search_hybrid
                try:
                    dense_results = vs.search(query, top_k=top_k * 2, filter_docs=filter_docs)
                    if dense_results:
                        results_by_source.append(("dense", dense_results))
                except Exception as e:
                    logger.warning("Dense search failed: %s", e)

        # ── Source 3: Knowledge Graph ───────────────────────────────
        kg_concepts = []
        if kg is not None:
            try:
                if filter_docs:
                    kg_concepts = kg.search_concepts_by_docs(query, filter_docs, limit=top_k)
                else:
                    kg_concepts = kg.search_concepts(query, limit=top_k)
                for c in kg_concepts:
                    neighbors = kg.get_neighbors(c.id)
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
                tool_name="semantic_search", query=query,
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
                        neighbor_str = ", ".join(
                            f"{n['name']}({n['relation']})" for n in neighbors
                        )
                        parts.append(f"  关联: {neighbor_str}")

        return ToolResult(
            tool_name="semantic_search",
            query=query,
            content="\n".join(parts),
            metadata={
                "total_fused": len(fused),
                "chunks_found": len(chunk_items),
                "concepts_found": len(graph_items),
            },
        )

    # ── Standalone search methods ────────────────────────────────────────

    def search_dense(self, query: str, top_k: int = 10,
                     filter_docs: set[str] | None = None) -> list[dict]:
        """Dense (ChromaDB) search only."""
        if self.vector_store is None:
            return []
        return self.vector_store.search(query, top_k, filter_docs)

    def search_hybrid_raw(self, query: str, top_k: int = 10,
                          filter_docs: set[str] | None = None) -> dict:
        """Return raw dense + sparse results dict (for RRF fusion by callers)."""
        if self.vector_store is None:
            return {"dense": [], "sparse": []}
        return self.vector_store.search_hybrid(query, top_k * 2, filter_docs)

    def search_concepts(self, query: str, limit: int = 10,
                        filter_docs: set[str] | None = None) -> list:
        """Search knowledge graph concepts."""
        if self.knowledge_graph is None:
            return []
        if filter_docs:
            return self.knowledge_graph.search_concepts_by_docs(query, filter_docs, limit)
        return self.knowledge_graph.search_concepts(query, limit)

    def get_neighbors(self, concept_id: str) -> list[dict]:
        """Get neighboring concepts in the knowledge graph."""
        if self.knowledge_graph is None:
            return []
        return self.knowledge_graph.get_neighbors(concept_id)

    # ── Document management ──────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict], doc_id: str = ""):
        """Index chunks into the vector store."""
        if self.vector_store is not None:
            self.vector_store.index_chunks(chunks, doc_id=doc_id)

    def add_concepts(self, concepts: list):
        """Add concepts to the knowledge graph."""
        if self.knowledge_graph is not None:
            self.knowledge_graph.add_concepts_batch(concepts)

    def add_relations(self, relations: list):
        """Add relations to the knowledge graph."""
        if self.knowledge_graph is not None:
            self.knowledge_graph.add_relations_batch(relations)

    def remove_document(self, doc_filename: str):
        """Remove all data for a document from both stores."""
        if self.vector_store is not None:
            self.vector_store.remove_document(doc_filename)
        if self.knowledge_graph is not None:
            self.knowledge_graph.remove_by_doc(doc_filename)

    def remove_by_chapter(self, doc_filename: str, chapter_title: str) -> int:
        """Remove data for a specific chapter. Returns chunks removed."""
        removed = 0
        if self.vector_store is not None:
            removed = self.vector_store.remove_by_chapter(doc_filename, chapter_title)
        if self.knowledge_graph is not None:
            self.knowledge_graph.remove_by_chapter(doc_filename, chapter_title)
        return removed

    def clear(self):
        """Clear all semantic data."""
        if self.vector_store is not None:
            self.vector_store.clear()
        if self.knowledge_graph is not None:
            self.knowledge_graph.clear()

    def get_doc_names(self) -> list[str]:
        """Get unique document filenames from both stores."""
        names: set[str] = set()
        if self.vector_store is not None:
            try:
                names.update(self.vector_store.get_doc_names())
            except Exception as e:
                logger.debug("Failed to get doc names from vector store: %s", e)
        if self.knowledge_graph is not None:
            try:
                names.update(self.knowledge_graph.get_doc_names())
            except Exception as e:
                logger.debug("Failed to get doc names from knowledge graph: %s", e)
        return sorted(names)

    def get_stats(self) -> dict:
        """Get combined stats from both stores."""
        stats = {"vector_chunks": 0, "kg_concepts": 0, "kg_relations": 0}
        if self.vector_store is not None:
            stats["vector_chunks"] = len(self.vector_store._all_texts)
        if self.knowledge_graph is not None:
            kg_stats = self.knowledge_graph.stats()
            stats["kg_concepts"] = kg_stats.get("concepts", 0)
            stats["kg_relations"] = kg_stats.get("relations", 0)
            stats["kg_categories"] = kg_stats.get("categories", {})
        return stats
