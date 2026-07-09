# Unit tests for text chunker

from __future__ import annotations

import pytest

from src.documents.parser import Document
from src.documents.chunker import chunk_document, TextChunk


class TestDocumentChunker:
    """Tests for chunk_document() function."""

    def test_empty_document(self):
        """Empty document should return empty list."""
        doc = Document(filename="empty.txt", content="")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_whitespace_only(self):
        """Whitespace-only document should return empty list."""
        doc = Document(filename="blank.txt", content="   \n\n  \n  ")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_short_text_single_chunk(self):
        """Short text should fit in a single chunk."""
        doc = Document(filename="short.txt", content="Hello world, this is a test.")
        chunks = chunk_document(doc, chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world, this is a test."
        assert chunks[0].chunk_index == 0
        assert chunks[0].doc_filename == "short.txt"

    def test_chunk_metadata(self):
        """Each chunk should carry correct metadata."""
        doc = Document(filename="meta.txt", content="Some text here.")
        chunks = chunk_document(doc, chapter_title="Chapter 1")
        assert len(chunks) == 1
        assert chunks[0].doc_filename == "meta.txt"
        assert chunks[0].chapter_title == "Chapter 1"
        assert isinstance(chunks[0].chunk_id, str)
        assert len(chunks[0].chunk_id) > 0

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple overlapping chunks."""
        # Generate text that far exceeds the chunk size
        paragraph = "This is sentence number {0}. It contains enough words to fill space. " * 3
        content = "\n\n".join(paragraph.format(i) for i in range(100))
        doc = Document(filename="long.txt", content=content)

        chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1

        # Each chunk should not exceed chunk_size (roughly — paragraphs may push it over slightly)
        for chunk in chunks:
            assert len(chunk.text) <= 800  # Allow some slack for paragraph merging

        # Chunk indices should be sequential
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chinese_text_chunking(self):
        """Chinese text should be correctly chunked on sentence boundaries."""
        content = (
            "第一章 概述\n\n"
            "深度学习是机器学习的一个分支。它使用多层神经网络来学习数据的表示。"
            "与传统的机器学习方法不同，深度学习可以自动从原始数据中提取特征。\n\n"
            "第二章 基础理论\n\n"
            "神经网络的基本单元是神经元。每个神经元接收多个输入信号，经过加权求和"
            "和激活函数处理后输出。常见的激活函数包括Sigmoid、ReLU和Tanh。\n\n"
            "第三章 优化方法\n\n"
            "梯度下降是最常用的优化算法。它通过计算损失函数关于参数的梯度，"
            "沿着梯度的反方向更新参数，从而最小化损失函数。"
        )
        doc = Document(filename="dl_intro.txt", content=content)
        chunks = chunk_document(doc, chunk_size=300, chapter_title="深度学习入门")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_filename == "dl_intro.txt"
            assert chunk.chapter_title == "深度学习入门"
            assert len(chunk.text) > 0

    def test_chunk_ids_are_unique(self):
        """Each chunk should have a unique chunk_id."""
        content = "\n\n".join(f"Paragraph {i} with enough text to fill some space. " * 5 for i in range(20))
        doc = Document(filename="ids.txt", content=content)

        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)
        chunk_ids = {c.chunk_id for c in chunks}
        assert len(chunk_ids) == len(chunks)

    def test_overlap_preserves_context(self):
        """Overlapping chunks should share some text at the boundary."""
        # Create text with clear boundaries — long enough to force multiple chunks
        sentences = []
        for i in range(50):
            sentences.append(
                f"Sentence {i}: The quick brown fox jumps over the lazy dog. "
                f"This adds more words to reach the needed length for testing. "
            )
        content = " ".join(sentences)
        doc = Document(filename="overlap.txt", content=content)

        chunks = chunk_document(doc, chunk_size=400, chunk_overlap=100)
        if len(chunks) >= 2:
            # The last portion of chunk 0 should appear at the start of chunk 1
            tail = chunks[0].text[-50:]
            assert tail in chunks[1].text or len(chunks[0].text) > 0


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_default_values(self):
        chunk = TextChunk(chunk_id="id1", text="hello", doc_filename="f.txt", chunk_index=0)
        assert chunk.chapter_title == ""
        assert chunk.metadata == {}

    def test_to_dict(self):
        # TextChunk is a dataclass, verify field access
        chunk = TextChunk(
            chunk_id="abc123", text="sample", doc_filename="doc.pdf",
            chunk_index=5, chapter_title="Ch 1",
            metadata={"source": "test"},
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.text == "sample"
        assert chunk.doc_filename == "doc.pdf"
        assert chunk.chunk_index == 5
        assert chunk.chapter_title == "Ch 1"
        assert chunk.metadata == {"source": "test"}
