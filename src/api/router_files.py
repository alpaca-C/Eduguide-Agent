"""File management endpoints — upload, list, delete documents."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from .deps import uploaded_files, UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more document files."""
    saved = []
    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in (".pdf", ".txt", ".md", ".docx"):
            raise HTTPException(400, f"Unsupported format: {suffix}")
        dest = UPLOAD_DIR / f.filename
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        uploaded_files[f.filename] = str(dest)
        saved.append(f.filename)
    return {"uploaded": saved, "total": len(saved)}


@router.get("/list")
async def list_files():
    """List currently uploaded files (synced with disk)."""
    # Remove entries for files that no longer exist on disk
    to_remove = [fn for fn, fp in uploaded_files.items() if not Path(fp).exists()]
    for fn in to_remove:
        del uploaded_files[fn]
    # Scan uploads dir for orphaned files
    if UPLOAD_DIR.exists():
        for p in UPLOAD_DIR.iterdir():
            if p.is_file() and p.name not in uploaded_files:
                uploaded_files[p.name] = str(p)
    return {"files": list(uploaded_files.keys())}


@router.delete("/{filename}")
async def delete_file(filename: str):
    """Delete an uploaded file from disk and tracking dict."""
    if filename not in uploaded_files:
        raise HTTPException(404, "File not found")
    filepath = uploaded_files.pop(filename)
    try:
        Path(filepath).unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to delete file on disk: %s", e)
    # Also check uploads dir for duplicates
    alt_path = UPLOAD_DIR / filename
    try:
        alt_path.unlink(missing_ok=True)
    except Exception as e:
        logger.debug("Failed to delete alt file %s: %s", alt_path, e)
    return {"deleted": filename, "remaining": list(uploaded_files.keys())}
