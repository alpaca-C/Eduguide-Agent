"""MinerU OCR tool — call MinerU cloud API to parse scanned PDFs.

Async flow: upload file → poll task → download result → extract text.
Requires MINERU_API_TOKEN in .env or config.

Registered as agent tool: mineru_ocr
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import zipfile
from pathlib import Path

import requests

from . import ToolResult, ToolErrorType, register_tool

logger = logging.getLogger(__name__)

# ---- Configurable ----
MINERU_BASE_URL = os.environ.get("MINERU_API_BASE_URL", "https://mineru.net").rstrip("/")
MINERU_TOKEN = os.environ.get("MINERU_API_TOKEN", "")
MINERU_POLL_INTERVAL = float(os.environ.get("MINERU_POLL_INTERVAL", "3"))  # seconds
MINERU_POLL_TIMEOUT = float(os.environ.get("MINERU_POLL_TIMEOUT", "300"))  # 5 min max wait


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {MINERU_TOKEN}",
        "Content-Type": "application/json",
    }


# ===========================================================================
# Public API
# ===========================================================================


def parse_pdf(
    filepath: str | Path,
    max_pages: int | None = None,
    page_range: tuple[int, int] | None = None,
    language: str = "ch",
) -> str:
    """Parse a PDF using MinerU cloud API. Returns extracted text (markdown).

    Args:
        filepath: Path to the PDF file.
        max_pages: Only parse the first N pages (None = all).
        page_range: Only parse pages [start, end] (1-indexed, both inclusive).
                    Uses server-side page_ranges parameter.
        language: Document language code (ch / en / auto).

    Returns:
        Extracted markdown text, or empty string on failure.
    """
    if not MINERU_TOKEN or "placeholder" in MINERU_TOKEN:
        logger.warning("MinerU token not configured, cannot OCR")
        return ""

    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return ""

    # ---- Step 0: Build clean PDF from specified pages ----
    # MinerU rejects many real-world PDFs. Render pages to PNG images
    # and embed in a fresh standards-compliant PDF → 100% compatible.
    upload_path = filepath
    temp_file = None
    if max_pages is not None or page_range is not None:
        temp_file = _make_clean_pdf(filepath, max_pages=max_pages, page_range=page_range)
        if temp_file:
            upload_path = temp_file
        else:
            logger.info("[MinerU] local truncation unavailable, will upload full file")

    try:
        logger.info(
            "[MinerU] uploading %s (%s bytes)",
            upload_path.name, upload_path.stat().st_size,
        )

        # Server-side page_ranges: only needed if uploading original PDF.
        # When using a clean temp PDF, pages are already extracted.
        page_ranges = None
        if not temp_file:
            if page_range:
                page_ranges = f"{page_range[0]}-{page_range[1]}"
            elif max_pages:
                page_ranges = f"1-{max_pages}"
        task_data = _submit_upload(upload_path, language, page_ranges=page_ranges)
        if not task_data:
            return ""

        task_id = task_data.get("task_id", "")
        batch_id = task_data.get("batch_id", "")

        result_url = _poll_task(task_id, batch_id)
        if result_url:
            return _download_and_extract(result_url)

        return ""

    finally:
        if temp_file is not None and temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e:
                logger.debug("Failed to remove temp OCR file %s: %s", temp_file, e)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _make_clean_pdf(filepath: Path, max_pages: int | None = None,
                   page_range: tuple[int, int] | None = None) -> Path | None:
    """Render specified PDF pages to PNG images → new clean standards-compliant PDF.

    MinerU rejects many real-world PDFs due to JBIG2/JPEG2000/corrupted xref
    tables. This function bypasses all format issues by rendering pages to
    PNG images (universally supported) and embedding them in a fresh PDF.

    Args:
        filepath: Original PDF.
        max_pages: First N pages (None = all).
        page_range: Pages [start, end] (1-indexed, both inclusive).

    Returns path to temp PDF, or None if no extraction needed.
    """
    import tempfile
    import os as _os

    try:
        import fitz
    except ImportError:
        logger.warning("[MinerU] pymupdf not available, cannot extract pages locally")
        return None

    try:
        doc = fitz.open(str(filepath))
        total = len(doc)

        # Determine page indices (0-indexed)
        if page_range:
            indices = list(range(max(0, page_range[0] - 1), min(page_range[1], total)))
        elif max_pages:
            limit = min(max_pages, total)
            if limit >= total:
                doc.close()
                return None
            indices = list(range(limit))
        else:
            doc.close()
            return None

        if not indices:
            doc.close()
            return None

        # Render each page as PNG and embed into a clean PDF.
        new_doc = fitz.open()
        for i in indices:
            page = doc[i]
            # Render at 200 DPI for good OCR quality
            pix = page.get_pixmap(dpi=200)
            png_bytes = pix.tobytes("png")
            img_page = new_doc.new_page(width=pix.width, height=pix.height)
            img_page.insert_image(img_page.rect, stream=png_bytes)

        tmp_fd, tmp_path_str = tempfile.mkstemp(
            suffix=".pdf", prefix="mineru_pages_",
        )
        _os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)
        new_doc.save(str(tmp_path), deflate=True, garbage=4)
        new_doc.close()
        doc.close()

        logger.info(
            "[MinerU] rendered %d pages as PNG-image PDF: %.1f MB → %.1f MB",
            len(indices),
            filepath.stat().st_size / (1024 * 1024),
            tmp_path.stat().st_size / (1024 * 1024),
        )
        return tmp_path

    except Exception as e:
        logger.warning("[MinerU] page extraction error: %s", e)
        return None


def _submit_upload(filepath: Path, language: str, page_ranges: str | None = None) -> dict | None:
    """Request a pre-signed upload URL, then PUT the file. Returns {task_id, batch_id}.

    Args:
        filepath: PDF to upload.
        language: Document language code.
        page_ranges: Optional page range for server-side truncation (e.g. "1-20").
                     Used as fallback when local page extraction produces a file
                     that MinerU cannot process.
    """
    try:
        # Request upload URL
        payload: dict = {
            "files": [{"name": filepath.name}],
            "is_ocr": True,
            "enable_formula": False,
            "enable_table": False,
            "language": language,
        }
        if page_ranges:
            payload["page_ranges"] = page_ranges

        resp = requests.post(
            f"{MINERU_BASE_URL}/api/v4/file-urls/batch",
            headers=_headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "[MinerU] batch response: %s",
            json.dumps(data, ensure_ascii=False)[:500],
        )

        batch_id = data["data"]["batch_id"]
        file_urls = data["data"]["file_urls"]
        if not file_urls:
            logger.error("[MinerU] no upload URLs returned")
            return None

        upload_url = file_urls[0]
        logger.info("[MinerU] got upload URL, batch=%s", batch_id)

        # PUT the file
        with open(filepath, "rb") as f:
            put_resp = requests.put(upload_url, data=f, timeout=120)
            put_resp.raise_for_status()

        logger.info("[MinerU] file uploaded, creating extraction task...")

        # Try explicit task creation (Strategy 1), fall back to polling.
        # Pass both the pre-signed upload URL and the batch response data
        # in case task creation needs different URL formats.
        return _wait_for_task_id(batch_id, upload_url, filepath.name, language, data)

    except requests.RequestException as e:
        logger.error("[MinerU] upload failed: %s", e)
        return None


def _wait_for_task_id(batch_id: str, upload_url: str = "", file_name: str = "", language: str = "ch", batch_data: dict | None = None) -> dict | None:
    """After file upload, get the MinerU parse task ID.

    Strategy (in order):
    1. Explicitly create an extraction task via POST (most reliable)
    2. Poll batch status endpoint (legacy — waits for auto-creation)
    3. List recent tasks as last resort
    """
    # ── Strategy 1: Explicit task creation ──
    task_id = _create_task_explicit(batch_id, upload_url, file_name, language, batch_data)
    if task_id:
        return {"task_id": task_id, "batch_id": batch_id}

    # ── Strategy 2: Poll batch endpoint ──
    logger.info("[MinerU] explicit creation skipped/unavailable, polling batch status...")
    for attempt in range(15):
        time.sleep(3)
        try:
            resp = requests.get(
                f"{MINERU_BASE_URL}/api/v4/extract/task/batch/{batch_id}",
                headers=_headers(),
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            tasks = data.get("data", {}).get("tasks", [])
            if tasks:
                task_id = tasks[0].get("task_id", "")
                logger.info("[MinerU] task found via batch poll: %s", task_id)
                return {"task_id": task_id, "batch_id": batch_id}
        except requests.RequestException:
            continue

    # ── Strategy 3: List recent tasks ──
    logger.warning("[MinerU] batch polling timed out, trying task list fallback")
    try:
        resp = requests.get(
            f"{MINERU_BASE_URL}/api/v4/extract/task?limit=10",
            headers=_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            tasks = resp.json().get("data", {}).get("tasks", [])
            if tasks:
                task_id = tasks[0].get("task_id", "")
                logger.info("[MinerU] task found via recent list: %s", task_id)
                return {"task_id": task_id, "batch_id": batch_id}
    except Exception as e:
        logger.debug("[MinerU] task list fallback failed: %s", e)

    logger.error("[MinerU] could not find or create task after upload (batch=%s)", batch_id)
    return None


def _create_task_explicit(batch_id: str, upload_url: str, file_name: str, language: str, batch_data: dict | None = None) -> str | None:
    """Explicitly create an extraction task via POST /api/v4/extract/task.

    This is the recommended flow for MinerU v4: upload → create task → poll → download.
    Returns task_id on success, None if the endpoint is unavailable or fails.

    Tries multiple payloads because MinerU API varies between deployments:
    - Some need `url` pointing to the uploaded file
    - Some locate files by `batch_id` alone
    - The `url` value may need to be the pre-signed URL or a relative path
    """
    if not batch_id:
        return None

    # Extract any additional URLs from batch response
    batch_urls = []
    if batch_data:
        batch_info = batch_data.get("data", {})
        # Some deployments return a download_url / file_path
        for key in ("download_url", "file_url", "result_url"):
            val = batch_info.get(key, "")
            if val:
                batch_urls.append(val)
        # Also check per-file info
        for f in batch_info.get("files", []):
            for key in ("url", "download_url"):
                val = f.get(key, "")
                if val and val != upload_url:
                    batch_urls.append(val)

    # Build candidate URL values
    url_candidates = [upload_url] + batch_urls if upload_url else batch_urls
    if not url_candidates:
        url_candidates = [""]

    payloads: list[dict] = []
    for url_val in url_candidates[:2]:  # limit to 2 URL variants
        base = {"batch_id": batch_id}
        if url_val:
            base["url"] = url_val
        # Variant A: full config without model_version (standard pipeline, most compatible)
        payloads.append({
            **base,
            "is_ocr": True,
            "enable_formula": False,
            "enable_table": False,
            "language": language,
        })
        # Variant B: minimal
        payloads.append(dict(base))

    for i, payload in enumerate(payloads):
        try:
            logger.info(
                "[MinerU] trying explicit task creation (attempt %d/%d)...",
                i + 1, len(payloads),
            )
            resp = requests.post(
                f"{MINERU_BASE_URL}/api/v4/extract/task",
                headers=_headers(),
                json=payload,
                timeout=30,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                # Response shapes: {"data": {"task_id": "..."}} or {"task_id": "..."}
                task_id = (
                    data.get("data", {}).get("task_id", "")
                    or data.get("task_id", "")
                )
                if task_id:
                    logger.info("[MinerU] task created explicitly: %s", task_id)
                    return task_id
                logger.warning(
                    "[MinerU] task creation returned 200 but no task_id in response: %s",
                    json.dumps(data, ensure_ascii=False)[:300],
                )
            elif resp.status_code == 404:
                # Endpoint doesn't exist on this deployment — skip remaining attempts
                logger.info("[MinerU] POST /api/v4/extract/task not available (404)")
                return None
            else:
                logger.warning(
                    "[MinerU] explicit task creation returned %d — payload=%s — response=%s",
                    resp.status_code,
                    json.dumps(payload, ensure_ascii=False)[:300],
                    resp.text[:300],
                )
        except requests.RequestException as e:
            logger.warning("[MinerU] explicit task creation request failed: %s", e)
            # If first payload fails on network error, try next
            continue

    return None


def _poll_task(task_id: str, batch_id: str) -> str | None:
    """Poll task status until done. Returns the result download URL, or None on failure."""
    if not task_id:
        return None

    url = f"{MINERU_BASE_URL}/api/v4/extract/task/{task_id}"
    deadline = time.time() + MINERU_POLL_TIMEOUT

    while time.time() < deadline:
        time.sleep(MINERU_POLL_INTERVAL)
        try:
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            task = data.get("data", {})
            state = task.get("state", "unknown")
            logger.info("[MinerU] task %s: %s", task_id, state)

            if state == "done":
                result_url = task.get("full_zip_url", "")
                if result_url:
                    return result_url
                # Try markdown URL fallback
                return task.get("md_url", "")
            elif state == "failed":
                logger.error(
                    "[MinerU] task %s failed — error_msg=%s | full_response=%s",
                    task_id,
                    task.get("error_msg", ""),
                    json.dumps(task, ensure_ascii=False)[:500],
                )
                return None
        except requests.RequestException as e:
            logger.warning("[MinerU] poll error: %s", e)

    logger.error("[MinerU] task %s timed out after %.0fs", task_id, MINERU_POLL_TIMEOUT)
    return None


def _download_and_extract(result_url: str) -> str:
    """Download the result zip and extract markdown text."""
    if not result_url:
        return ""
    try:
        logger.info("[MinerU] downloading result...")
        resp = requests.get(result_url, timeout=120)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find the markdown file
            md_files = [n for n in zf.namelist() if n.endswith(".md")]
            if not md_files:
                logger.warning("[MinerU] no markdown in result zip: %s", zf.namelist()[:10])
                return ""
            # Usually one main .md file
            text = zf.read(md_files[0]).decode("utf-8", errors="replace")
            logger.info("[MinerU] extracted %d chars from %s", len(text), md_files[0])
            return text

    except Exception as e:
        logger.error("[MinerU] download/extract failed: %s", e)
        return ""


# ===========================================================================
# Agent tool registration
# ===========================================================================

# Cache for OCR results to avoid re-processing same file
_ocr_result_cache: dict[str, str] = {}


async def mineru_ocr(query: str, max_pages: int = 20) -> ToolResult:
    """Agent-callable tool: OCR a scanned PDF using MinerU API.

    `query` should be the filename as shown in the uploaded files list.
    Only processes the first `max_pages` (default 20, enough for chapter detection).
    """
    from pathlib import Path

    # Resolve filename to full path
    full_path = ""
    upload_dir = Path(__file__).resolve().parent.parent.parent / "uploads"

    # Try direct path first, then upload dir
    if Path(query).exists():
        full_path = query
    elif (upload_dir / query).exists():
        full_path = str(upload_dir / query)
    else:
        # Search uploads for partial match
        for f in upload_dir.iterdir():
            if f.is_file() and query.lower() in f.name.lower():
                full_path = str(f)
                break

    if not full_path:
        return ToolResult(
            tool_name="mineru_ocr",
            query=query,
            content=f"文件未找到: {query}",
            error=ToolErrorType.NOT_CONFIGURED,
            error_detail=f"Could not resolve filename to a file path: {query}",
            metadata={"error": "file_not_found"},
        )

    # Check cache
    cache_key = f"{full_path}:{max_pages}"
    if cache_key in _ocr_result_cache:
        logger.info("[MinerU tool] using cached result for %s", Path(full_path).name)
        return ToolResult(
            tool_name="mineru_ocr",
            query=query,
            content=_ocr_result_cache[cache_key],
            metadata={"cached": True},
        )

    # Run OCR in thread (blocking HTTP calls)
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, parse_pdf, full_path, max_pages)

    if not text.strip():
        return ToolResult(
            tool_name="mineru_ocr",
            query=query,
            content="OCR 识别失败或文档无文字内容。请确认 MINERU_API_TOKEN 已配置且文件为扫描版 PDF。",
            error=ToolErrorType.INTERNAL,
            error_detail="MinerU OCR returned empty text. Check MINERU_API_TOKEN and file type.",
            metadata={"error": "ocr_failed"},
        )

    # Cache result
    _ocr_result_cache[cache_key] = text

    preview = text[:300]
    return ToolResult(
        tool_name="mineru_ocr",
        query=query,
        content=f"OCR 识别完成（{len(text)} 字符）。\n\n预览：\n{preview}...",
        metadata={"chars": len(text), "full_text": text},
    )


# Register as agent tool
register_tool(
    name="mineru_ocr",
    description="对扫描版PDF进行OCR文字识别。当文档上传后检测到无文字内容时使用此工具。输入为已上传的文件名。",
    func=mineru_ocr,
)
