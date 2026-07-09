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
        total_files = len(file_chapters)

        for idx, (full_path, chap_infos) in enumerate(file_chapters.items()):
            fname = Path(full_path).name
            yield _sse({
                "type": "progress",
                "stage": f"解析: {fname}",
                "pct": int((idx / max(total_files, 1)) * 25),
            })
            await asyncio.sleep(0.1)

            # Resolve page range (required for image PDF targeted OCR).
            _page_range: tuple[int, int] | None = None
            if chap_infos:
                _ci = chap_infos[0]
                _info = chapters_cache.get(_ci["label"], {})
                _sp = _info.get("start_page", 0)
                _ep = _info.get("end_page", 0)
                if _sp <= 0 or _ep <= 0:
                    _saved = _load_chapters(fname)
                    for _sc in _saved:
                        if _sc.get("title") == _ci.get("title"):
                            _sp = _sc.get("start_page", 0)
                            _ep = _sc.get("end_page", 0)
                            break
                if _sp > 0 and _ep > 0 and _ep >= _sp:
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
                        # Prepend chapter/section header so chunks carry citation info
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

        # ── Step 4: Extract Knowledge Graph ──
        kg.clear()
        result = await asyncio.to_thread(extract_full_document, all_chunks, config, kg)
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

    # Remove from KG (concepts for this document)
    kg_removed = kg.remove_by_doc(doc_filename) if kg else 0

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
