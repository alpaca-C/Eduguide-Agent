"""Shared dependencies and global state for all API routers.

All singletons are managed through src.context.AppContext — a single
centralized container that replaces the previously scattered module-level
globals. Routers can continue to import individual names (config, store,
kg, vs, etc.) without changes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

# ── Boot AppContext at import time ─────────────────────────────────────

from src.context import init_context, get_context, AppContext

_ctx: AppContext = init_context()

# ── Re-export individual names for backward compat ─────────────────────

config         = _ctx.config
store          = _ctx.memory_store
kg             = _ctx.knowledge_graph
chapter_agent  = _ctx.chapter_agent
vs             = _ctx.vector_store
memory_manager = _ctx.memory_manager
exact_cache    = _ctx.exact_cache
gssc_pipeline  = _ctx.gssc_pipeline
rag_skill      = _ctx.rag_skill
supervisor     = _ctx.supervisor  # may be None if init failed

chapters_cache: dict = _ctx.chapters_cache
uploaded_files: dict[str, str] = _ctx.uploaded_files

# ── Paths ──────────────────────────────────────────────────────────────

UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CHAPTERS_DB = PROJECT_ROOT / "storage" / "memory.db"


# ── FastAPI dependency (for gradual migration) ─────────────────────────

def get_app_context() -> AppContext:
    """FastAPI Depends callable — returns the shared AppContext."""
    return get_context()


# ── Chapter DB helpers ─────────────────────────────────────────────────

def _get_chapters_conn() -> sqlite3.Connection:
    CHAPTERS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHAPTERS_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS doc_chapters "
        "(filename TEXT PRIMARY KEY, chapters_json TEXT, updated_at REAL)"
    )
    conn.commit()
    return conn


def _load_chapters(filename: str = ""):
    conn = _get_chapters_conn()
    if filename:
        row = conn.execute(
            "SELECT chapters_json FROM doc_chapters WHERE filename = ?",
            (filename,),
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else []
    rows = conn.execute(
        "SELECT filename, chapters_json FROM doc_chapters"
    ).fetchall()
    conn.close()
    return {r[0]: json.loads(r[1]) for r in rows}


def _save_chapters(filename: str, chapters: list):
    import time
    conn = _get_chapters_conn()
    conn.execute(
        "INSERT OR REPLACE INTO doc_chapters "
        "(filename, chapters_json, updated_at) VALUES (?, ?, ?)",
        (filename, json.dumps(chapters, ensure_ascii=False), time.time()),
    )
    conn.commit()
    conn.close()
