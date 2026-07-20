# Integration tests for RAG search — RRF fusion, dedup, empty results
#
# Tests rag_search with mocked vector store and knowledge graph.

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── Helpers to build mock hybrid results ─────────────────────────

def _mock_vs_dense(query: str, top_k: int, filter_docs=None) -> list[dict]:
    return [
        {"chunk_id": "chunk_1", "text": "库仑定律：两点电荷间的作用力与电荷量的乘积成正比，与距离的平方成反比。",
         "doc_filename": "电磁学.pdf", "chapter_title": "第一章 静电场", "source": "dense"},
        {"chunk_id": "chunk_2", "text": "电场强度定义为单位正电荷所受的力，是矢量。",
         "doc_filename": "电磁学.pdf", "chapter_title": "第一章 静电场", "source": "dense"},
    ]


def _mock_vs_sparse(query: str, top_k: int, filter_docs=None) -> list[dict]:
    return [
        {"chunk_id": "chunk_3", "text": "库仑定律的数学表达式为 F = k·q₁q₂/r²。",
         "doc_filename": "电磁学.pdf", "chapter_title": "第一章 静电场", "source": "sparse"},
        {"chunk_id": "chunk_4", "text": "库仑定律适用于点电荷，对于连续分布电荷需要积分处理。",
         "doc_filename": "电磁学.pdf", "chapter_title": "第二章 电场计算", "source": "sparse"},
    ]


def _mock_kg_concepts(query: str, limit: int) -> list:
    """Return mock ConceptNode list."""
    from src.knowledge.graph import ConceptNode
    return [
        ConceptNode(id="c1", name="库仑定律", description="描述两点电荷间作用力的定律",
                     category="definition", doc_filename="电磁学.pdf"),
    ]


def _mock_kg_neighbors(concept_id: str) -> list[dict]:
    return [
        {"concept_name": "电场强度", "relation_type": "related_to",
         "concept_id": "c2", "description": "单位正电荷受力", "category": "definition",
         "relation_desc": ""},
    ]


class TestRAGSearchIntegration:
    """Integration tests for rag_search — the 3-source hybrid retrieval pipeline."""

    def _setup_mock_backends(self):
        """Wire mock functions into the rag_search module's globals."""
        import src.tools.rag_search as rs

        mock_vs = MagicMock()
        # search_hybrid + search (legacy compat)
        mock_vs.search_hybrid.side_effect = (
            lambda query, top_k=5, filter_docs=None: {
                "dense": _mock_vs_dense(query, top_k, filter_docs),
                "sparse": _mock_vs_sparse(query, top_k, filter_docs),
            }
        )
        mock_vs.search.side_effect = (
            lambda query, top_k=5, filter_docs=None:
                _mock_vs_dense(query, top_k, filter_docs)
        )
        # _search_dense + _search_sparse (new code calls these directly)
        mock_vs._search_dense.side_effect = (
            lambda query, top_k=20, filter_docs=None:
                _mock_vs_dense(query, top_k, filter_docs)
        )
        mock_vs._search_sparse.side_effect = (
            lambda query, top_k=10, filter_docs=None:
                _mock_vs_sparse(query, top_k, filter_docs)
        )
        mock_vs.get_doc_names.return_value = ["电磁学.pdf", "量子力学.pdf"]

        mock_kg = MagicMock()
        mock_kg.search_concepts.side_effect = _mock_kg_concepts
        mock_kg.search_concepts_by_docs.side_effect = (
            lambda query, docs, limit: _mock_kg_concepts(query, limit)
        )
        mock_kg.get_neighbors.side_effect = _mock_kg_neighbors
        mock_kg.get_doc_names.return_value = ["电磁学.pdf"]

        # Store originals for cleanup
        orig_vs = rs._vector_store
        orig_kg = rs._knowledge_graph
        orig_mm = rs._memory_manager
        rs._vector_store = mock_vs
        rs._knowledge_graph = mock_kg
        rs._memory_manager = None  # Bypass manager path, use globals directly
        return orig_vs, orig_kg, orig_mm

    def _teardown_mock_backends(self, state):
        """Restore original backends."""
        orig_vs, orig_kg, orig_mm = state
        import src.tools.rag_search as rs
        rs._vector_store = orig_vs
        rs._knowledge_graph = orig_kg
        rs._memory_manager = orig_mm

    @pytest.mark.asyncio
    async def test_default_search_uses_dense_with_ce(self):
        """Default rag_search: Dense + Cross-Encoder only (no sparse/graph)."""
        state = self._setup_mock_backends()  # (orig_vs, orig_kg, orig_mm)
        try:
            from src.tools.rag_search import rag_search

            result = await rag_search("库仑定律", top_k=3)

            assert result.error is None
            assert "本地文档检索结果" in result.content
            assert result.metadata["total_fused"] >= 1
            assert result.metadata["chunks_found"] > 0
            assert result.metadata["ce_reranked"] is True
        finally:
            self._teardown_mock_backends(state)

    @pytest.mark.asyncio
    async def test_fullsearch_includes_graph_and_sparse(self):
        """rag_fullsearch: Dense + Sparse + Graph + Cross-Encoder."""
        state = self._setup_mock_backends()  # (orig_vs, orig_kg, orig_mm)
        try:
            from src.tools.rag_search import rag_fullsearch

            result = await rag_fullsearch("库仑定律", top_k=3)

            assert result.error is None
            assert "本地文档检索结果" in result.content
            # Full search queries 3 sources; at minimum gets dense chunks
            assert result.metadata["total_fused"] >= 2
            assert result.metadata["chunks_found"] > 0
            assert result.metadata["ce_reranked"] is True
        finally:
            self._teardown_mock_backends(state)

    @pytest.mark.asyncio
    async def test_rrf_dedup_removes_duplicates(self):
        """RRF fusion should merge same-content chunks from different sources."""
        # Test the RRF function directly
        from src.tools.rag_search import _rrf_fuse

        # Same text appears in both dense and sparse
        same_text = "库仑定律公式 F=kq₁q₂/r²"
        dense = [{"text": same_text, "doc_filename": "a.pdf",
                   "chapter_title": "Ch1", "source": "dense"}]
        sparse = [{"text": same_text, "doc_filename": "a.pdf",
                    "chapter_title": "Ch1", "source": "sparse"}]

        fused = _rrf_fuse([("dense", dense), ("sparse", sparse)], top_k=5)

        assert len(fused) == 1  # deduplicated
        assert fused[0]["rrf_score"] > 0
        assert "dense" in fused[0]["sources"]
        assert "sparse" in fused[0]["sources"]

    @pytest.mark.asyncio
    async def test_empty_result_when_no_backends(self):
        """When no backends are configured, result should be an empty/error result."""
        import src.tools.rag_search as rs

        orig_vs = rs._vector_store
        orig_kg = rs._knowledge_graph
        rs._vector_store = None
        rs._knowledge_graph = None
        try:
            result = await rs.rag_search("测一下")
            assert result.error is not None
        finally:
            rs._vector_store = orig_vs
            rs._knowledge_graph = orig_kg

    @pytest.mark.asyncio
    async def test_rrf_fusion_sorts_by_score(self):
        """Higher-ranked results should get higher RRF scores."""
        from src.tools.rag_search import _rrf_fuse

        # Create distinct texts so no dedup
        dense = [
            {"text": f"text_{i}", "doc_filename": "a.pdf",
             "chapter_title": "Ch1", "source": "dense"}
            for i in range(5)
        ]
        sparse = [
            {"text": f"sparse_{i}", "doc_filename": "a.pdf",
             "chapter_title": "Ch1", "source": "sparse"}
            for i in range(5)
        ]

        fused = _rrf_fuse([("dense", dense), ("sparse", sparse)], top_k=10)
        scores = [f["rrf_score"] for f in fused]

        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True)
        assert len(fused) == 10
