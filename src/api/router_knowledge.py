"""Knowledge processing — SSE streaming: parse → chunk → index → extract KG."""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.config import Configuration
from src.documents.parser import parse_document, Document
from src.documents.chunker import chunk_document as chunk_doc_func
from src.agents.extractor import extract_full_document
from src.agents.chapterizer import _split_by_meta

from .schemas import ProcessRequest
from .deps import kg, vs, chapters_cache, UPLOAD_DIR, uploaded_files, _load_chapters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/process")
async def process_knowledge(req: ProcessRequest):
    """Process selected chapters with SSE progress streaming."""
    if not req.filepaths:
        raise HTTPException(400, "No files provided")

    async def event_stream():
        _total_t0 = _time.time()
        session_id = str(uuid.uuid4())[:12]

        def _sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield _sse({"type": "status", "text": "正在解析文档...", "session_id": session_id})
        await asyncio.sleep(0.1)

        # ── Step 0: Group chapters by source file ──
        # Label format: "[filename] chapter_title" — parse filename from label.
        import re as _re
        file_chapters: dict[str, list[dict]] = {}
        for label in (req.selected_chapters or []):
            # Parse filename from label: "[电磁学 梁灿彬.pdf] 第一章 xxx"
            _m = _re.match(r'\[(.+?)\]\s+(.+)', label)
            _fname = _m.group(1) if _m else ""
            _ctitle = _m.group(2) if _m else label

            # Resolve full path: cache → uploaded_files → upload dir
            info = chapters_cache.get(label, {})
            fp = info.get("full_path", "")
            if not fp and _fname:
                fp = uploaded_files.get(_fname, "")
            if not fp and _fname:
                _candidate = UPLOAD_DIR / _fname
                if _candidate.exists():
                    fp = str(_candidate)
            if not fp:
                logger.error("Cannot resolve file for label=%s fname=%s", label, _fname)
                yield _sse({"type": "error", "msg": f"找不到文件: {_fname}"})
                return

            if fp not in file_chapters:
                file_chapters[fp] = []
            # Prefer cache title, fall back to parsed title from label
            title = info.get("title", "") or _ctitle
            marker = info.get("start_marker", "") or title
            file_chapters[fp].append({
                "label": label,
                "title": title,
                "start_marker": marker,
            })

        # ── Step 1: Full parse + split ──
        documents: list[Document] = []
        chapter_map: dict[str, str] = {}
        processed_files: set[str] = set()
        processed_chapters: dict[str, list[str]] = {}  # fname → [chapter_titles]
        total_files = len(file_chapters)

        for idx, (full_path, chap_infos) in enumerate(file_chapters.items()):
            fname = Path(full_path).name
            processed_files.add(fname)
            processed_chapters[fname] = [ci.get("title", "") for ci in chap_infos]
            yield _sse({
                "type": "progress",
                "stage": f"解析: {fname}",
                "pct": int((idx / max(total_files, 1)) * 25),
            })
            await asyncio.sleep(0.1)

            # Resolve page range (required for image PDF targeted OCR).
            # Apply toc_offset to correct for cover/copyright/preface/TOC pages.
            _page_range: tuple[int, int] | None = None
            _toc_offset = 0
            if chap_infos:
                _ci = chap_infos[0]
                _info = chapters_cache.get(_ci["label"], {})
                _sp = _info.get("start_page", 0)
                _ep = _info.get("end_page", 0)
                _toc_offset = _info.get("toc_offset", 0)
                if _sp <= 0 or _ep <= 0:
                    _saved = _load_chapters(fname)
                    for _sc in _saved:
                        if _sc.get("title") == _ci.get("title"):
                            _sp = _sc.get("start_page", 0)
                            _ep = _sc.get("end_page", 0)
                            _toc_offset = _toc_offset or _sc.get("toc_offset", 0)
                            break
                if _sp > 0 and _ep > 0 and _ep >= _sp:
                    if _toc_offset:
                        _sp += _toc_offset
                        _ep += _toc_offset
                        logger.info(
                            "[API KNOWLEDGE] %s | toc_offset=%d applied -> "
                            "page_range=(%d, %d)", fname, _toc_offset, _sp, _ep,
                        )
                    _page_range = (_sp, _ep)
                else:
                    yield _sse({
                        "type": "error",
                        "msg": f"缺少章节页码信息，请重新检测章节后再处理: {_ci['title']}",
                    })
                    return

            full_doc = await asyncio.to_thread(
                parse_document, full_path, page_range=_page_range,
            )

            if _page_range:
                # Image PDF with targeted OCR: the entire output IS the
                # chapter's text. No _split_by_meta needed — we already
                # OCR'd exactly the right pages.
                if full_doc.content.strip():
                    for ci in chap_infos:
                        _header = f"《{fname}》{ci['title']}\n\n"
                        documents.append(Document(
                            filename=fname, content=_header + full_doc.content,
                        ))
                        chapter_map[ci["label"]] = ci["title"]
                continue

            # Digital PDF: split by markers as usual
            markers = [
                {"title": ci["title"], "start_marker": ci["start_marker"]}
                for ci in chap_infos
            ]
            full_chapters = _split_by_meta(full_doc.content, markers) if markers else []

            if not full_chapters:
                for ci in chap_infos:
                    info = chapters_cache.get(ci["label"], {})
                    text = info.get("text", "")
                    if text.strip():
                        documents.append(Document(filename=fname, content=text))
                        chapter_map[ci["label"]] = ci["title"]
                continue

            for fc in full_chapters:
                matched_ci = next(
                    (ci for ci in chap_infos if ci["title"] == fc.title),
                    chap_infos[0] if chap_infos else None,
                )
                if matched_ci and fc.text.strip():
                    documents.append(Document(filename=fname, content=fc.text))
                    chapter_map[matched_ci["label"]] = fc.title

        if not documents:
            yield _sse({
                "type": "error",
                "msg": "No chapters selected or all failed to parse",
            })
            return

        yield _sse({
            "type": "progress",
            "stage": f"已解析 {len(documents)} 个章节, 开始分块...",
            "pct": 30,
        })
        await asyncio.sleep(0.1)

        # ── Step 2: Chunk ──
        config = Configuration.from_env()
        all_chunks = []
        for idx, doc in enumerate(documents):
            chapter_title = ""
            for label, title in chapter_map.items():
                if title and title in doc.filename + doc.content[:50]:
                    chapter_title = title
                    break
            if not chapter_title:
                for label, title in chapter_map.items():
                    cache_info = chapters_cache.get(label, {})
                    cache_text = cache_info.get("text", "")
                    if cache_text and cache_text[:50] == doc.content[:50]:
                        chapter_title = title
                        break

            chunks = await asyncio.to_thread(
                chunk_doc_func, doc,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                chapter_title=chapter_title,
            )
            all_chunks.extend(chunks)
            if idx % 3 == 0 or idx == len(documents) - 1:
                yield _sse({
                    "type": "progress",
                    "stage": f"分块: {idx + 1}/{len(documents)}",
                    "pct": 35 + int((idx / max(len(documents), 1)) * 15),
                })
                await asyncio.sleep(0.1)

        yield _sse({
            "type": "progress",
            "stage": f"共 {len(all_chunks)} 个文本片段, 开始索引...",
            "pct": 55,
        })
        await asyncio.sleep(0.1)

        # ── Step 3: Index ──
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id, "text": c.text,
                "doc_filename": c.doc_filename, "chapter_title": c.chapter_title,
                "chunk_index": c.chunk_index,
            }
            for c in all_chunks
        ]
        await asyncio.to_thread(vs.index_chunks, chunk_dicts)

        yield _sse({
            "type": "progress",
            "stage": "索引完成, 开始提取知识图谱...",
            "pct": 65,
        })
        await asyncio.sleep(0.1)

        # ── Step 4: Extract Knowledge Graph (incremental) ──
        # Only extract from NEW chunks (not previously indexed).
        # Old concepts are preserved — no deletion needed.
        new_chunks = []
        vs_content_hashes = getattr(vs, '_content_hashes', set())
        for ch in all_chunks:
            h = vs._text_hash(ch.text[:2000]) if hasattr(vs, '_text_hash') else hash(ch.text[:500])
            if h not in vs_content_hashes:
                new_chunks.append(ch)
        skipped = len(all_chunks) - len(new_chunks)
        if skipped > 0:
            yield _sse({"type": "progress", "stage": f"跳过{skipped}个已有chunk, 提取{len(new_chunks)}个新chunk...", "pct": 66})
            await asyncio.sleep(0.1)

        if new_chunks:
            result = await asyncio.to_thread(extract_full_document, new_chunks, config, kg)
            # Cross-chapter linking: connect new concepts to existing
            try:
                from src.config import Configuration as _Cfg
                _cfg = _Cfg.from_env()
                from langchain_openai import ChatOpenAI
                _llm = ChatOpenAI(model=_cfg.llm_model_id, api_key=_cfg.llm_api_key,
                                  base_url=_cfg.llm_base_url, temperature=0.0, max_tokens=500)
                kg.link_cross_chapter(embedding_fn=vs._ef, llm=_llm)
            except Exception as e:
                logger.warning("Cross-chapter linking skipped: %s", e)
        else:
            result = {"concepts_extracted": 0, "relations_extracted": 0}
        stats = kg.stats()

        yield _sse({
            "type": "progress",
            "stage": f"提取完成: {stats['concepts']} 概念, {stats['relations']} 关系",
            "pct": 95,
        })
        await asyncio.sleep(0.1)

        all_concepts = kg.get_all_concepts()
        cats_display = {
            "definition": "定义", "theorem": "定理", "method": "方法",
            "example": "示例", "concept": "概念",
        }
        categories_summary = {}
        for cat, cat_cn in cats_display.items():
            cat_concepts = [c for c in all_concepts if c.category == cat]
            if cat_concepts:
                categories_summary[cat_cn] = [
                    {"name": c.name, "description": c.description[:80]}
                    for c in cat_concepts[:10]
                ]

        elapsed = _time.time() - _total_t0
        yield _sse({
            "type": "complete",
            "session_id": session_id,
            "documents_processed": len(documents),
            "chunks_total": len(all_chunks),
            "concepts_extracted": result["concepts_extracted"],
            "relations_extracted": result["relations_extracted"],
            "stats": {
                "concepts": stats["concepts"],
                "relations": stats["relations"],
                "categories": stats.get("categories", {}),
            },
            "categories_summary": categories_summary,
            "elapsed": round(elapsed, 1),
        })

        # ── Flush processing stats ──
        try:
            from src.monitoring.usage_store import (
                get_recorder_store, ProcessingRecord,
            )
            store = get_recorder_store()
            for fname, chap_titles in processed_chapters.items():
                # Count chunks for this doc
                doc_chunks = sum(
                    1 for c in all_chunks
                    if getattr(c, 'doc_filename', '') == fname
                )
                store.insert_processing(ProcessingRecord(
                    doc_filename=fname,
                    chapter_title=", ".join(chap_titles[:3]),
                    operation="full",
                    pages=0,  # not tracked here (page range handled by parser)
                    chunks=doc_chunks,
                    llm_calls=0,   # captured by token_usage.db via harness hooks
                    total_tokens=0,
                    latency_ms=round(elapsed * 1000),
                ))
            logger.info(
                "[STATS] processing flushed: %d files, %d chapters, %d chunks, %.1fs",
                len(processed_chapters), len(documents),
                len(all_chunks), elapsed,
            )
        except Exception as e:
            logger.warning("Failed to flush processing stats: %s", e)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/chapters/{label:path}")
async def delete_chapter(label: str):
    """Remove a specific chapter's chunks from vector store and KG.

    Label format: "[filename] chapter_title" (as produced by chapter detection).
    This allows granular deletion without affecting other chapters from the same file.
    """
    import re
    m = re.match(r'\[(.+?)\]\s+(.+)', label)
    if not m:
        raise HTTPException(400, f"Invalid label format: {label}")

    doc_filename = m.group(1)
    chapter_title = m.group(2)

    # Remove from vector store (ChromaDB + FTS5)
    vs_removed = vs.remove_by_chapter(doc_filename, chapter_title) if vs else 0

    # Remove from KG (concepts for this specific chapter)
    kg_removed = kg.remove_by_chapter(doc_filename, chapter_title) if kg else 0

    # Clean up cached state
    from .deps import chapters_cache as cc
    if label in cc:
        del cc[label]

    logger.info(
        "[API KNOWLEDGE] deleted chapter '%s': %d chunks, %d concepts",
        label, vs_removed, kg_removed,
    )
    return {
        "status": "deleted",
        "label": label,
        "chunks_removed": vs_removed,
        "concepts_removed": kg_removed,
    }


@router.delete("/documents/{filename}")
async def delete_document_knowledge(filename: str):
    """Remove all chunks and KG data for a specific document.

    Used when re-detecting chapters — clears previous import state
    so the user starts fresh for this document only.
    """
    from urllib.parse import unquote
    filename = unquote(filename)

    # Remove from vector store (ChromaDB + FTS5)
    vs_removed = 0
    if vs:
        vs.remove_document(filename)
        # Count removed from FTS
        try:
            cursor = vs._fts_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunk_fts WHERE doc_filename = ?", (filename,))
            remaining = cursor.fetchone()[0]
            vs_removed = "cleared"
        except Exception:
            vs_removed = "cleared"

    # Remove from KG
    kg_removed = kg.remove_by_doc(filename) if kg else 0

    # Clean up cached chapters
    from .deps import chapters_cache as cc
    labels_to_remove = [l for l in cc if cc[l].get("filename") == filename]
    for l in labels_to_remove:
        del cc[l]

    logger.info(
        "[API KNOWLEDGE] deleted document '%s': vs=%s, kg=%d concepts",
        filename, vs_removed, kg_removed,
    )
    return {
        "status": "deleted",
        "filename": filename,
        "vector_store": vs_removed,
        "concepts_removed": kg_removed,
    }


@router.delete("/clear")
async def clear_knowledge():
    """Clear all knowledge graph and vector store data."""
    kg.clear()
    vs.clear()
    from .deps import chapters_cache as cc
    cc.clear()
    return {"status": "cleared"}


@router.get("/stats")
async def get_stats():
    """Get knowledge graph statistics."""
    stats = kg.stats()
    doc_names = kg.get_doc_names()
    return {
        "concepts": stats["concepts"],
        "relations": stats["relations"],
        "categories": stats.get("categories", {}),
        "documents": doc_names,
    }


@router.get("/documents")
async def list_documents():
    """List indexed document names from the knowledge graph."""
    return {"documents": kg.get_doc_names()}
