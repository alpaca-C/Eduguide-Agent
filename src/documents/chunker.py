# Text Chunker — split documents into overlapping chunks for embedding

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import Document


@dataclass
class TextChunk:
    """A chunk of text with positional metadata."""
    chunk_id: str
    text: str
    doc_filename: str
    chunk_index: int
    chapter_title: str = ""    # 所属章节标题
    metadata: dict = field(default_factory=dict)


def chunk_document(
    doc: Document,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    chapter_title: str = "",
) -> list[TextChunk]:
    """Split a document into overlapping text chunks.
    
    Tries to split on natural boundaries (paragraphs, sentences)
    before falling back to fixed-size splits.
    """
    text = doc.content
    if not text.strip():
        return []
    
    # Strategy 1: Split by double newlines (paragraphs), then merge
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""
    chunk_idx = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        else:
            if current_chunk:
                chunks.append(_make_chunk(current_chunk, doc.filename, chunk_idx, chapter_title))
                chunk_idx += 1
            current_chunk = para
    
    if current_chunk:
        chunks.append(_make_chunk(current_chunk, doc.filename, chunk_idx, chapter_title))
        chunk_idx += 1
    
    # If chunks are too large, split further by sentences
    final_chunks = []
    for ch in chunks:
        if len(ch.text) <= chunk_size * 1.2:
            final_chunks.append(ch)
        else:
            sub = _split_by_sentences(ch, chunk_size, chunk_overlap)
            final_chunks.extend(sub)
    
    # Reindex
    for i, ch in enumerate(final_chunks):
        ch.chunk_index = i
        ch.chunk_id = f"{doc.filename}_{i}"
    
    return final_chunks


def _make_chunk(text: str, filename: str, idx: int, chapter_title: str = "") -> TextChunk:
    return TextChunk(
        chunk_id=f"{filename}_{idx}",
        text=text,
        doc_filename=filename,
        chapter_title=chapter_title,
        chunk_index=idx,
    )


def _split_by_sentences(chunk: TextChunk, max_size: int, overlap: int) -> list[TextChunk]:
    """Split a large chunk by sentence boundaries."""
    sentences = re.split(r"(?<=[。！？.!?])\s*", chunk.text)
    result = []
    current = ""
    idx = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= max_size:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                result.append(TextChunk(
                    chunk_id=f"{chunk.doc_filename}_{chunk.chunk_index}_{idx}",
                    text=current,
                    doc_filename=chunk.doc_filename,
                    chapter_title=chunk.chapter_title,
                    chunk_index=idx,
                ))
                idx += 1
                # Overlap: keep last portion
                overlap_text = current[-overlap:] if len(current) > overlap else ""
                current = overlap_text + " " + sent if overlap_text else sent
    
    if current:
        result.append(TextChunk(
            chunk_id=f"{chunk.doc_filename}_{chunk.chunk_index}_{idx}",
            text=current,
            doc_filename=chunk.doc_filename,
            chapter_title=chunk.chapter_title,
            chunk_index=idx,
        ))
    
    return result
