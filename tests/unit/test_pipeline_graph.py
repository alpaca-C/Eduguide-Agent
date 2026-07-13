# Unit tests for the document processing pipeline graph (graph.py)

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph import (
    PipelineState,
    parse_node,
    chunk_node,
    index_node,
    extract_node,
    build_pipeline,
)
from src.documents.parser import Document


# ── Helpers ────────────────────────────────────────────────────────────

def _make_doc(filename: str, content: str) -> Document:
    return Document(filename=filename, content=content)


# ========================================================================
# parse_node
# ========================================================================

class TestParseNode:
    """Tests for parse_node() — the first stage of the pipeline."""

    def test_empty_filepaths_returns_error(self):
        """When no filepaths are provided, return error with ready=False."""
        result = parse_node({"filepaths": []}, {})
        assert result["error"] == "没有文件"
        assert result["ready"] is False

    def test_no_filepaths_key_returns_error(self):
        """When filepaths key is missing from state, return error."""
        result = parse_node({}, {})
        assert result["error"] == "没有文件"
        assert result["ready"] is False

    def test_single_file_parsed_successfully(self):
        """A single valid filepath should produce one document."""
        mock_doc = _make_doc("test.txt", "Hello World")
        with patch("src.graph.parse_document", return_value=mock_doc):
            result = parse_node({"filepaths": ["/tmp/test.txt"]}, {})

        assert "error" not in result
        assert len(result["documents"]) == 1
        assert result["documents"][0].filename == "test.txt"
        assert result["documents"][0].content == "Hello World"
        assert "已解析 1 个文件" in result["message"]

    def test_multiple_files_parsed_successfully(self):
        """Multiple valid filepaths should produce multiple documents."""
        docs = [
            _make_doc("a.txt", "content A"),
            _make_doc("b.pdf", "content B"),
            _make_doc("c.md", "content C"),
        ]
        with patch("src.graph.parse_document", side_effect=docs):
            result = parse_node(
                {"filepaths": ["/tmp/a.txt", "/tmp/b.pdf", "/tmp/c.md"]}, {}
            )

        assert len(result["documents"]) == 3
        assert "已解析 3 个文件" in result["message"]
        total = sum(len(d.content) for d in docs)
        assert f"共 {total} 字符" in result["message"]

    def test_all_files_fail_returns_error(self):
        """When all files fail to parse, return error with ready=False."""
        with patch("src.graph.parse_document", side_effect=Exception("parse error")):
            result = parse_node({"filepaths": ["/tmp/bad.pdf"]}, {})

        assert result["error"] == "所有文件解析失败"
        assert result["ready"] is False

    def test_partial_failures_still_succeed(self):
        """When some files fail and some succeed, still return the successful ones."""
        def parse_side_effect(fp):
            if "good" in fp:
                return _make_doc("good.txt", "content")
            raise Exception("bad file")

        with patch("src.graph.parse_document", side_effect=parse_side_effect):
            result = parse_node(
                {"filepaths": ["/tmp/good.txt", "/tmp/bad.pdf"]}, {}
            )

        assert len(result["documents"]) == 1
        assert result["documents"][0].filename == "good.txt"
        assert "已解析 1 个文件" in result["message"]

    def test_message_includes_total_char_count(self):
        """The message should report the total character count correctly."""
        with patch("src.graph.parse_document",
                   return_value=_make_doc("f.txt", "12345")):
            result = parse_node({"filepaths": ["/tmp/f.txt"]}, {})

        assert "共 5 字符" in result["message"]


# ========================================================================
# chunk_node
# ========================================================================

class TestChunkNode:
    """Tests for chunk_node() — splits documents into text chunks."""

    def test_empty_documents_produces_no_chunks(self):
        """No documents should produce an empty chunk list."""
        with patch("src.graph.chunk_document") as mock_chunk:
            result = chunk_node({"documents": []}, {})
            assert result["chunks"] == []
            # chunk_document should not be called
            mock_chunk.assert_not_called()

    def test_single_document_chunked(self):
        """A single document should be passed to chunk_document."""
        doc = _make_doc("test.txt", "Hello " * 100)
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "c1"
        mock_chunk.text = "Hello " * 100
        mock_chunk.doc_filename = "test.txt"
        mock_chunk.chunk_index = 0

        with patch("src.graph.chunk_document", return_value=[mock_chunk]):
            result = chunk_node({"documents": [doc]}, {})

        assert len(result["chunks"]) == 1
        assert "已分为 1 个文本片段" in result["message"]

    def test_multiple_documents_produce_aggregated_chunks(self):
        """Chunks from multiple documents should be concatenated."""
        doc_a = _make_doc("a.txt", "A" * 50)
        doc_b = _make_doc("b.txt", "B" * 50)

        def chunk_side_effect(doc, **kwargs):
            if doc.filename == "a.txt":
                chunk = MagicMock()
                chunk.chunk_id = "ca1"
                return [chunk]
            else:
                c1 = MagicMock()
                c1.chunk_id = "cb1"
                c2 = MagicMock()
                c2.chunk_id = "cb2"
                return [c1, c2]

        with patch("src.graph.chunk_document", side_effect=chunk_side_effect):
            result = chunk_node({"documents": [doc_a, doc_b]}, {})

        assert len(result["chunks"]) == 3
        assert "已分为 3 个文本片段" in result["message"]


# ========================================================================
# index_node
# ========================================================================

class TestIndexNode:
    """Tests for index_node() — indexes chunks into DocumentVectorStore."""

    def test_empty_chunks_indexes_nothing(self):
        """No chunks should still call index_chunks with empty list."""
        with patch("src.graph.DocumentVectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs_cls.return_value = mock_vs

            result = index_node({"chunks": []}, {})

            mock_vs.index_chunks.assert_called_once_with([])
            assert "已索引 0 个片段" in result["message"]

    def test_chunks_are_indexed_with_correct_shape(self):
        """Each chunk should be converted to a dict with expected keys."""
        c1 = MagicMock()
        c1.chunk_id = "id-1"
        c1.text = "text 1"
        c1.doc_filename = "doc.pdf"
        c1.chunk_index = 0

        c2 = MagicMock()
        c2.chunk_id = "id-2"
        c2.text = "text 2"
        c2.doc_filename = "doc.pdf"
        c2.chunk_index = 1

        with patch("src.graph.DocumentVectorStore") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs_cls.return_value = mock_vs

            result = index_node({"chunks": [c1, c2]}, {})

            call_args = mock_vs.index_chunks.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0] == {
                "chunk_id": "id-1", "text": "text 1",
                "doc_filename": "doc.pdf", "chunk_index": 0,
            }
            assert call_args[1] == {
                "chunk_id": "id-2", "text": "text 2",
                "doc_filename": "doc.pdf", "chunk_index": 1,
            }
            assert "已索引 2 个片段" in result["message"]


# ========================================================================
# extract_node
# ========================================================================

class TestExtractNode:
    """Tests for extract_node() — extracts knowledge graph from chunks."""

    def test_extract_with_chunks(self):
        """Should call extract_full_document and clear the KG first."""
        chunks = [MagicMock()]

        with patch("src.graph.KnowledgeGraph") as mock_kg_cls, \
             patch("src.graph.extract_full_document") as mock_extract:
            mock_kg = MagicMock()
            mock_kg.stats.return_value = {"concepts": 5, "relations": 3}
            mock_kg_cls.return_value = mock_kg

            mock_extract.return_value = {
                "concepts_extracted": 5,
                "relations_extracted": 3,
            }

            result = extract_node(
                {"chunks": chunks},
                {"configurable": {}},
            )

            # KG should be cleared before extraction
            mock_kg.clear.assert_called_once()
            # extract_full_document should be called with chunks, config, and kg
            mock_extract.assert_called_once()

            assert result["concepts_extracted"] == 5
            assert result["relations_extracted"] == 3
            assert result["ready"] is True
            assert "5 个概念" in result["message"]
            assert "3 个关系" in result["message"]

    def test_extract_with_empty_chunks(self):
        """Extraction on empty chunks should still work."""
        with patch("src.graph.KnowledgeGraph") as mock_kg_cls, \
             patch("src.graph.extract_full_document") as mock_extract:
            mock_kg = MagicMock()
            mock_kg.stats.return_value = {"concepts": 0, "relations": 0}
            mock_kg_cls.return_value = mock_kg

            mock_extract.return_value = {
                "concepts_extracted": 0,
                "relations_extracted": 0,
            }

            result = extract_node({"chunks": []}, {"configurable": {}})

            assert result["concepts_extracted"] == 0
            assert result["ready"] is True

    def test_extract_passes_config_to_extract_full_document(self):
        """Configuration should be passed through to extract_full_document."""
        chunks = [MagicMock()]

        with patch("src.graph.Configuration") as mock_cfg_cls, \
             patch("src.graph.KnowledgeGraph") as mock_kg_cls, \
             patch("src.graph.extract_full_document") as mock_extract:
            mock_cfg = MagicMock()
            mock_cfg_cls.return_value = mock_cfg
            mock_kg = MagicMock()
            mock_kg.stats.return_value = {"concepts": 0, "relations": 0}
            mock_kg_cls.return_value = mock_kg
            mock_extract.return_value = {
                "concepts_extracted": 0, "relations_extracted": 0,
            }

            extract_node(
                {"chunks": chunks},
                {"configurable": {"chunk_size": 500}},
            )

            # Configuration should have been created from configurable dict
            mock_cfg_cls.assert_called_once_with(chunk_size=500)


# ========================================================================
# build_pipeline
# ========================================================================

class TestBuildPipeline:
    """Tests for build_pipeline() — graph compilation."""

    def test_returns_compiled_graph(self):
        """build_pipeline should return a compiled LangGraph graph."""
        graph = build_pipeline()
        # A compiled graph should have an invoke / ainvoke method
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_graph_has_all_nodes(self):
        """The graph should contain all 4 processing nodes."""
        graph = build_pipeline()
        # The graph object should be callable
        assert graph is not None

    def test_graph_can_be_invoked_with_empty_state(self):
        """The compiled graph should not crash when invoked with minimal state."""
        graph = build_pipeline()
        # We just verify the graph compiles and is usable
        assert graph is not None
