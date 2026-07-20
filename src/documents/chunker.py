# Text Chunker — split documents into overlapping chunks for embedding

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import Document


# Chinese numeral → digit mapping
_CN_NUM = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}


def _parse_cn_number(s: str) -> int:
    """Parse Chinese numeral string to int. '十二' -> 12, '一百二十三' -> 123."""
    s = s.strip()
    if s.isdigit():
        return int(s)
    # Simple case: just digits 0-10 or "十二" = 12
    result = 0
    if '百' in s:
        parts = s.split('百')
        result += _CN_NUM.get(parts[0], 0) * 100
        s = parts[1] if len(parts) > 1 else ''
    if '十' in s:
        parts = s.split('十')
        if parts[0]:
            result += _CN_NUM.get(parts[0], 0) * 10
        else:
            result += 10
        s = parts[1] if len(parts) > 1 else ''
    if s:
        result += _CN_NUM.get(s, 0)
    return result


def _chapter_slug(chapter_title: str) -> str:
    """Extract chapter number from title to make chunk_ids unique per chapter.

    '第1章 绪论' -> 'Ch1'
    '第6章 关系数据理论' -> 'Ch6'
    '第十二章 数学物理方程' -> 'Ch12'
    '' -> ''
    """
    if not chapter_title:
        return ''
    m = re.match(r'第\s*([零〇一二三四五六七八九十百千\d]+)\s*[章童篇]', chapter_title)
    if not m:
        m = re.match(r'Chapter\s+(\d+)', chapter_title, re.IGNORECASE)
    if m:
        try:
            num = _parse_cn_number(m.group(1))
            return f'_Ch{num}'
        except (ValueError, KeyError):
            pass
    return ''


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
    slug = _chapter_slug(chapter_title)
    for i, ch in enumerate(final_chunks):
        ch.chunk_index = i
        ch.chunk_id = f"{doc.filename}{slug}_{i}"

    return final_chunks


def _make_chunk(text: str, filename: str, idx: int, chapter_title: str = "") -> TextChunk:
    slug = _chapter_slug(chapter_title)
    return TextChunk(
        chunk_id=f"{filename}{slug}_{idx}",
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
    slug = _chapter_slug(chunk.chapter_title)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= max_size:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                result.append(TextChunk(
                    chunk_id=f"{chunk.doc_filename}{slug}_{chunk.chunk_index}_{idx}",
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
            chunk_id=f"{chunk.doc_filename}{slug}_{chunk.chunk_index}_{idx}",
            text=current,
            doc_filename=chunk.doc_filename,
            chapter_title=chunk.chapter_title,
            chunk_index=idx,
        ))

    return result
