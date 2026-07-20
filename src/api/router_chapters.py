"""Chapter detection & management — SSE streaming detection, save, load.

Multi-file chapter detection uses asyncio.gather + Semaphore for true
concurrency — N files are processed in parallel (default 3-way), each
pushing progress events to an asyncio.Queue consumed by the SSE stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.config import Configuration
from src.documents.parser import parse_document
from src.agents.chapterizer import _split_by_meta

from .schemas import ChapterDetectRequest, SaveChaptersRequest
from .deps import (
    chapter_agent, chapters_cache, uploaded_files,
    _save_chapters, _load_chapters, kg, vs, UPLOAD_DIR,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chapters", tags=["chapters"])


async def _detect_one_file(
    fp: str,
    full_path: str,
    file_idx: int,
    file_total: int,
    semaphore: asyncio.Semaphore,
    queue: asyncio.Queue,
) -> int:
    """Process one file: parse → detect → cache. Pushes SSE events to queue.

    Returns the number of chapters found (0 on failure).
    Runs under a semaphore to limit concurrent LLM calls.
    Always pushes a None sentinel when done (success or failure).
    """
    try:
        return await _detect_one_file_impl(fp, full_path, file_idx, file_total, semaphore, queue)
    except Exception as e:
        logger.error(
            "[API CHAPTERIZE] worker crashed for %s: %s",
            Path(full_path).name, e,
        )
        await queue.put({
            "type": "error", "file": fp,
            "msg": f"worker error: {e}",
        })
        await queue.put(None)  # sentinel
        return 0


async def _detect_one_file_impl(
    fp: str,
    full_path: str,
    file_idx: int,
    file_total: int,
    semaphore: asyncio.Semaphore,
    queue: asyncio.Queue,
) -> int:
    async with semaphore:
        _t0 = _time.time()
        fname = Path(full_path).name

        # Step 1: Parse first 20 pages
        await queue.put({
            "type": "progress", "file": fname,
            "stage": "preview parse...",
            "file_idx": file_idx, "file_total": file_total,
        })
        preview_doc = await asyncio.to_thread(parse_document, full_path, max_pages=20)
        _t1 = _time.time()
        logger.info(
            "[API CHAPTERIZE] %s | preview parse (20p): %.1fs | chars=%d",
            fname, _t1 - _t0, len(preview_doc.content),
        )

        # Step 2: LLM-based chapter detection
        await queue.put({
            "type": "progress", "file": fname,
            "stage": "detecting...",
            "file_idx": file_idx, "file_total": file_total,
        })
        try:
            preview_chapters = await chapter_agent.detect_all(preview_doc, None)
        except RuntimeError:
            logger.error(
                "[API CHAPTERIZE] %s | detection FAILED after max reflection rounds",
                fname,
            )
            await queue.put({
                "type": "error", "file": fp,
                "msg": "detection failed",
            })
            return 0

        _t2 = _time.time()
        chapters_found = len(preview_chapters)
        logger.info(
            "[API CHAPTERIZE] %s | detect_all: %.1fs | chapters=%d",
            fname, _t2 - _t1, chapters_found,
        )

        # Step 3: Cache chapters (with page ranges from VLM if available)
        vlm_ranges: dict[str, dict] = {}
        for _r in (preview_doc.metadata.get("vlm_chapter_ranges") or []):
            vlm_ranges[_r["title"]] = _r

        # Compute TOC offset: the gap between printed page numbers (from TOC)
        # and actual PDF pages (after cover/copyright/preface/TOC).
        # VLM reports body_start = which PDF page (1-indexed) Ch1 content begins.
        # toc_offset = body_start - first_chapter_printed_page
        _vlm_body_start = preview_doc.metadata.get("vlm_body_start") or 0
        toc_offset = 0
        if _vlm_body_start > 0 and vlm_ranges:
            _first_range = next(iter(vlm_ranges.values()))
            _first_printed = _first_range.get("start_page", 0)
            if _first_printed > 0:
                toc_offset = _vlm_body_start - _first_printed
                logger.info(
                    "[API CHAPTERIZE] %s | toc_offset=%d (body_start=%d - first_printed=%d)",
                    fname, toc_offset, _vlm_body_start, _first_printed,
                )

        # Check which chapters already have chunks in vector store
        _imported_chapters = vs.get_imported_chapter_titles(fname) if vs else set()
        chapter_list = []
        for ch in preview_chapters:
            label = f"[{fname}] {ch.title}"
            _ch_title_norm = ch.title.replace("：", " ").replace(":", " ").strip()
            _range = vlm_ranges.get(ch.title) or vlm_ranges.get(_ch_title_norm)
            chapter_list.append({
                "label": label,
                "title": ch.title,
                "filename": fname,
                "level": ch.level,
                "text_preview": ch.text[:200] if ch.text else "",
                "text_length": len(ch.text) if ch.text else 0,
                "start_page": _range.get("start_page", 0) if _range else 0,
                "end_page": _range.get("end_page", 0) if _range else 0,
                "imported": ch.title.replace("：", "").replace(":", "").replace(" ", "") in {
                    t.replace("：", "").replace(":", "").replace(" ", "")
                    for t in _imported_chapters
                },
            })
            chapters_cache[label] = {
                "filename": fname,
                "title": ch.title,
                "text": ch.text,
                "level": ch.level,
                "start_marker": ch.start_marker,
                "full_path": full_path,
                "start_page": _range.get("start_page", 0) if _range else 0,
                "end_page": _range.get("end_page", 0) if _range else 0,
                "toc_offset": toc_offset,
            }

        _t3 = _time.time()
        logger.info(
            "[API CHAPTERIZE] %s | TOTAL: %.1fs (parse=%.1fs detect=%.1fs cache=%.1fs)",
            fname, _t3 - _t0, _t1 - _t0, _t2 - _t1, _t3 - _t2,
        )

        # Push file_done with chapters list embedded
        await queue.put({
            "type": "file_done",
            "file": fname,
            "chapters_found": chapters_found,
            "file_idx": file_idx,
            "file_total": file_total,
            "chapters": chapter_list,
        })
        # Sentinel: signal consumer that this worker is done
        await queue.put(None)
        return chapters_found


@router.post("/detect")
async def detect_chapters_api(req: ChapterDetectRequest):
    """Detect chapters for multiple files concurrently via SSE progress streaming.

    Files are processed in parallel (limited by CHAPTER_DETECT_CONCURRENCY),
    each pushing progress events to an asyncio.Queue. A consumer coroutine
    streams events to the client in real time as workers complete.
    """
    chapters_cache.clear()
    if not req.filepaths:
        return {"chapters": [], "message": "No files uploaded"}

    # Resolve file paths
    resolved: list[tuple[str, str]] = []  # (fp, full_path)
    for fp in req.filepaths:
        full_path = uploaded_files.get(fp, "")
        if not full_path:
            full_path = str(Path(__file__).resolve().parent.parent.parent / "uploads" / fp)
        if Path(full_path).exists():
            resolved.append((fp, full_path))
        else:
            logger.warning("[API CHAPTERIZE] file not found, skipping: %s", fp)

    if not resolved:
        return {"chapters": [], "message": "No valid files found"}

    total = len(resolved)

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        config = Configuration.from_env()
        concurrency = getattr(config, "chapter_detect_concurrency", 3) or 3
        semaphore = asyncio.Semaphore(concurrency)

        logger.info(
            "[API CHAPTERIZE] starting %d files with concurrency=%d",
            total, concurrency,
        )

        # Launch concurrent workers
        tasks = [
            _detect_one_file(fp, full_path, idx + 1, total, semaphore, queue)
            for idx, (fp, full_path) in enumerate(resolved)
        ]

        # Consumer: read events from queue and yield SSE, counting sentinels
        pending = len(tasks)
        gather_task = asyncio.ensure_future(
            asyncio.gather(*tasks, return_exceptions=True)
        )

        while pending > 0:
            # Wait for next event (with timeout to check if all tasks crashed)
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # No event for 30s — check if gather finished
                if gather_task.done():
                    break
                continue

            if event is None:  # Sentinel from a finished worker
                pending -= 1
                continue

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # Collect results from gather
        results = await gather_task
        all_chapters_found = sum(
            r for r in results if isinstance(r, int) and r > 0
        )

        # Gather all cached chapters for the complete event
        all_chapters = []
        for fp, full_path in resolved:
            fname = Path(full_path).name
            for label, info in chapters_cache.items():
                if info.get("filename") == fname:
                    all_chapters.append({
                        "label": label,
                        "title": info.get("title", ""),
                        "filename": fname,
                        "level": info.get("level", 1),
                        "text_preview": info.get("text", "")[:200],
                        "text_length": len(info.get("text", "")),
                    })

        # Fallback: if no chapters detected, add "all" entries
        if not all_chapters:
            for fp in req.filepaths:
                name = Path(fp).name
                label = f"[{name}] all"
                all_chapters.append({
                    "label": label, "title": "all", "filename": name,
                    "level": 1, "text_preview": "", "text_length": 0,
                })
                chapters_cache[label] = {
                    "filename": name, "title": "all", "text": "", "level": 1,
                }

        yield f"data: {json.dumps({'type': 'complete', 'chapters': all_chapters, 'total': len(all_chapters), 'files_processed': total, 'total_chapters_found': all_chapters_found})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/save")
async def save_chapters(req: SaveChaptersRequest):
    """Save detected chapters to persistent storage."""
    if not req.filename:
        raise HTTPException(400, "filename is required")

    # Enrich with page ranges from in-memory cache (frontend may drop them)
    enriched = []
    for ch in req.chapters:
        label = ch.get("label", "")
        cached = chapters_cache.get(label, {})
        enriched.append({
            **ch,
            "start_page": ch.get("start_page") or cached.get("start_page", 0),
            "end_page": ch.get("end_page") or cached.get("end_page", 0),
            "start_marker": ch.get("start_marker") or cached.get("start_marker", ch.get("title", "")),
            "toc_offset": ch.get("toc_offset") or cached.get("toc_offset", 0),
        })

    _save_chapters(req.filename, enriched)
    return {"status": "ok"}


@router.get("/{filename}")
async def get_chapters(filename: str):
    """Get cached chapters for a file. Also restores page ranges to memory
    so processing works after server restart without re-detection."""
    chapters = _load_chapters(filename)
    _imported_chapters = vs.get_imported_chapter_titles(filename) if vs else set()
    for ch in chapters:
        _t = ch.get("title", "").replace("：", "").replace(":", "").replace(" ", "")
        ch["imported"] = _t in {
            t.replace("：", "").replace(":", "").replace(" ", "")
            for t in _imported_chapters
        }
        # Restore to memory cache so knowledge processing has page ranges
        label = ch.get("label", f"[{filename}] {ch.get('title', '')}")
        if label not in chapters_cache:
            chapters_cache[label] = {
                "filename": filename,
                "title": ch.get("title", ""),
                "text": ch.get("text_preview", ""),
                "level": ch.get("level", 1),
                "start_marker": ch.get("start_marker", ch.get("title", "")),
                "full_path": str(UPLOAD_DIR / filename),
                "start_page": ch.get("start_page", 0),
                "end_page": ch.get("end_page", 0),
                "toc_offset": ch.get("toc_offset", 0),
            }
    return {"chapters": chapters}
