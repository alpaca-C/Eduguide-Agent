# Unit tests for knowledge extractor pure utility functions

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.agents.extractor import (
    _build_chapter_batches,
    _parse_json_response,
    _build_batch_input,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _make_chunk(chunk_index=0, doc_filename="test.pdf",
                chapter_title=None, text="sample text"):
    """Create a mock TextChunk."""
    chunk = MagicMock()
    chunk.chunk_index = chunk_index
    chunk.doc_filename = doc_filename
    chunk.chapter_title = chapter_title
    chunk.text = text
    return chunk


# ========================================================================
# _build_chapter_batches
# ========================================================================

class TestBuildChapterBatches:
    def test_empty_chunks_returns_empty(self):
        batches = _build_chapter_batches([], max_batch_size=3)
        assert batches == []

    def test_single_chunk_single_batch(self):
        chunks = [_make_chunk(0)]
        batches = _build_chapter_batches(chunks, max_batch_size=3)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_batch_size_limit_splits(self):
        """When batch reaches max size, next chunk starts a new batch."""
        chunks = [_make_chunk(i) for i in range(5)]
        batches = _build_chapter_batches(chunks, max_batch_size=2)
        # 5 chunks, max 2 per batch → 3 batches (2+2+1)
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_chapter_boundary_splits(self):
        """Chunks from different chapters should not be mixed."""
        chunks = [
            _make_chunk(0, chapter_title="第一章"),
            _make_chunk(1, chapter_title="第一章"),
            _make_chunk(2, chapter_title="第二章"),
            _make_chunk(3, chapter_title="第二章"),
        ]
        batches = _build_chapter_batches(chunks, max_batch_size=5)
        assert len(batches) == 2
        # All chapter 1 chunks in batch 0
        assert all(c.chapter_title == "第一章" for c in batches[0])
        # All chapter 2 chunks in batch 1
        assert all(c.chapter_title == "第二章" for c in batches[1])

    def test_document_boundary_splits(self):
        """Chunks from different documents should not be mixed."""
        chunks = [
            _make_chunk(0, doc_filename="a.pdf"),
            _make_chunk(1, doc_filename="b.pdf"),
        ]
        batches = _build_chapter_batches(chunks, max_batch_size=5)
        assert len(batches) == 2

    def test_no_chapter_title_uses_doc_filename(self):
        """When chapter_title is None, doc_filename is used as grouping key."""
        chunks = [
            _make_chunk(0, doc_filename="a.pdf", chapter_title=None),
            _make_chunk(1, doc_filename="a.pdf", chapter_title=None),
            _make_chunk(2, doc_filename="b.pdf", chapter_title=None),
        ]
        batches = _build_chapter_batches(chunks, max_batch_size=5)
        assert len(batches) == 2
        assert len(batches[0]) == 2  # a.pdf chunks together
        assert len(batches[1]) == 1  # b.pdf chunk alone


# ========================================================================
# _parse_json_response
# ========================================================================

class TestParseJsonResponse:
    def test_parse_valid_json(self):
        data = _parse_json_response('{"key": "value"}')
        assert data == {"key": "value"}

    def test_parse_json_in_text_with_prefix(self):
        """Should extract JSON even when prefixed with text."""
        data = _parse_json_response('Some text\n{"concepts": [], "relations": []}\nMore text')
        assert data == {"concepts": [], "relations": []}

    def test_parse_json_with_markdown_code_block(self):
        """Should handle JSON inside ```json code blocks."""
        content = '```json\n{"concepts": [{"name": "AI"}]}\n```'
        data = _parse_json_response(content)
        assert data is not None
        assert data["concepts"][0]["name"] == "AI"

    def test_parse_nested_json(self):
        """Should handle nested braces correctly."""
        content = '{"a": {"b": {"c": "deep"}}}'
        data = _parse_json_response(content)
        assert data == {"a": {"b": {"c": "deep"}}}

    def test_parse_malformed_json_returns_none(self):
        assert _parse_json_response("not json at all") is None

    def test_parse_unbalanced_braces_returns_none(self):
        assert _parse_json_response('{"a": "b"') is None

    def test_parse_empty_string_returns_none(self):
        assert _parse_json_response("") is None

    def test_parse_no_braces_returns_none(self):
        assert _parse_json_response("no curly braces here") is None

    def test_parse_multiple_json_objects_takes_first(self):
        """Should parse the first complete JSON object."""
        content = '{"first": 1} extra {"second": 2}'
        data = _parse_json_response(content)
        assert data == {"first": 1}


# ========================================================================
# _build_batch_input
# ========================================================================

class TestBuildBatchInput:
    def test_basic_input_with_chapter(self):
        chunks = [
            _make_chunk(0, doc_filename="book.pdf", chapter_title="第一章",
                       text="这是第一章的内容。"),
        ]
        combined, source_file, source_chapter = _build_batch_input(
            chunks, chunk_max_chars=1000,
        )
        assert source_file == "book.pdf"
        assert source_chapter == "第一章"
        assert "这是第一章的内容" in combined
        assert "[片段 0 | book.pdf | 第一章]" in combined

    def test_basic_input_without_chapter(self):
        chunks = [
            _make_chunk(0, doc_filename="notes.txt", chapter_title=None,
                       text="纯文本内容。"),
        ]
        combined, source_file, source_chapter = _build_batch_input(
            chunks, chunk_max_chars=1000,
        )
        assert source_file == "notes.txt"
        assert source_chapter is None
        assert "文件: notes.txt" in combined

    def test_truncates_long_text(self):
        """Text longer than chunk_max_chars should be truncated."""
        long_text = "A" * 500
        chunks = [_make_chunk(0, text=long_text)]
        combined, _, _ = _build_batch_input(chunks, chunk_max_chars=100)
        assert len(combined) < 500

    def test_skips_empty_chunks(self):
        chunks = [
            _make_chunk(0, text="   "),  # whitespace only
            _make_chunk(1, text="valid text"),
        ]
        combined, _, _ = _build_batch_input(chunks, chunk_max_chars=1000)
        assert "valid text" in combined

    def test_multiple_chunks_separated(self):
        chunks = [
            _make_chunk(0, text="chunk1"),
            _make_chunk(1, text="chunk2"),
        ]
        combined, _, _ = _build_batch_input(chunks, chunk_max_chars=1000)
        assert "---" in combined  # Separator between chunks
        assert "chunk1" in combined
        assert "chunk2" in combined

    def test_all_empty_chunks_produces_empty_string(self):
        chunks = [
            _make_chunk(0, text="   "),
            _make_chunk(1, text=""),
        ]
        combined, _, _ = _build_batch_input(chunks, chunk_max_chars=1000)
        assert combined == ""
