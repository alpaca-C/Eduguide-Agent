# Usage store — per-request QA stats + per-chapter processing stats
#
# Two tables:
#   request_stats   — one row per chat request (aggregated from harness hooks)
#   processing_stats — one row per chapter processing operation
#
# Usage:
#   from src.monitoring.usage_store import get_recorder_store
#   store = get_recorder_store()
#   store.insert_request(...)
#   stats = store.request_stats(since_ts=...)

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RequestRecord:
    """Aggregated stats for one chat request."""
    request_id: str = ""
    session_id: str = ""
    question: str = ""
    route: str = ""                  # trivial / moderate / complex / tutor
    rounds: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    total_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessingRecord:
    """Stats for one chapter / document processing operation."""
    doc_filename: str = ""
    chapter_title: str = ""
    operation: str = ""              # chunk / index / extract / full
    pages: int = 0
    chunks: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ── Store ─────────────────────────────────────────────────────────────────────

class RecorderStore:
    """Thread-safe stats store backed by SQLite."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = str(
                Path(__file__).resolve().parent.parent.parent
                / "data" / "usage_stats.db"
            )
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def db_path(self) -> str:
        return str(self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL DEFAULT '',
                route TEXT NOT NULL DEFAULT '',
                rounds INTEGER NOT NULL DEFAULT 0,
                llm_calls INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                total_latency_ms REAL NOT NULL DEFAULT 0.0,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_req_ts ON request_stats(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_req_route ON request_stats(route)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_filename TEXT NOT NULL DEFAULT '',
                chapter_title TEXT NOT NULL DEFAULT '',
                operation TEXT NOT NULL DEFAULT '',
                pages INTEGER NOT NULL DEFAULT 0,
                chunks INTEGER NOT NULL DEFAULT 0,
                llm_calls INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_proc_ts ON processing_stats(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_proc_doc ON processing_stats(doc_filename)
        """)
        conn.commit()
        conn.close()

    # ── Insert ────────────────────────────────────────────────────────────

    def insert_request(self, record: RequestRecord) -> int:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                """INSERT INTO request_stats
                   (request_id, session_id, question, route, rounds,
                    llm_calls, prompt_tokens, completion_tokens, tool_calls,
                    total_latency_ms, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.request_id, record.session_id,
                    record.question[:200], record.route, record.rounds,
                    record.llm_calls, record.prompt_tokens,
                    record.completion_tokens, record.tool_calls,
                    record.total_latency_ms, record.timestamp,
                ),
            )
            row_id = cur.lastrowid
            conn.commit()
            conn.close()
            return row_id

    def insert_processing(self, record: ProcessingRecord) -> int:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                """INSERT INTO processing_stats
                   (doc_filename, chapter_title, operation, pages, chunks,
                    llm_calls, total_tokens, latency_ms, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.doc_filename, record.chapter_title, record.operation,
                    record.pages, record.chunks, record.llm_calls,
                    record.total_tokens, record.latency_ms, record.timestamp,
                ),
            )
            row_id = cur.lastrowid
            conn.commit()
            conn.close()
            return row_id

    # ── Query ─────────────────────────────────────────────────────────────

    def request_stats(self, since_ts: float = 0.0, limit: int = 50) -> list[dict]:
        """Get recent request stats, optionally filtered by time."""
        conn = self._get_conn()
        if since_ts:
            rows = conn.execute(
                "SELECT * FROM request_stats WHERE timestamp >= ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (since_ts, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM request_stats ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [_request_row(r) for r in rows]

    def request_aggregate(self, since_ts: float = 0.0) -> dict:
        """Aggregate stats: total requests, tokens, avg latency by route."""
        conn = self._get_conn()
        params = (since_ts,) if since_ts else ()
        where = "WHERE timestamp >= ?" if since_ts else ""

        total = conn.execute(
            f"SELECT COUNT(*), SUM(llm_calls), SUM(prompt_tokens), "
            f"SUM(completion_tokens), SUM(tool_calls), AVG(total_latency_ms) "
            f"FROM request_stats {where}", params,
        ).fetchone()

        by_route = conn.execute(
            f"SELECT route, COUNT(*), AVG(prompt_tokens + completion_tokens), "
            f"AVG(total_latency_ms) FROM request_stats {where} "
            f"GROUP BY route ORDER BY COUNT(*) DESC", params,
        ).fetchall()

        conn.close()
        return {
            "total_requests": total[0] or 0,
            "total_llm_calls": total[1] or 0,
            "total_prompt_tokens": total[2] or 0,
            "total_completion_tokens": total[3] or 0,
            "total_tool_calls": total[4] or 0,
            "avg_latency_ms": round(total[5] or 0, 1),
            "by_route": [
                {"route": r[0], "count": r[1],
                 "avg_tokens": round(r[2] or 0, 0),
                 "avg_latency_ms": round(r[3] or 0, 1)}
                for r in by_route
            ],
        }

    def processing_stats(self, doc: str = "", limit: int = 20) -> list[dict]:
        """Get processing stats, optionally filtered by document."""
        conn = self._get_conn()
        if doc:
            rows = conn.execute(
                "SELECT * FROM processing_stats WHERE doc_filename = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (doc, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM processing_stats ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [_proc_row(r) for r in rows]

    def processing_aggregate(self, since_ts: float = 0.0) -> dict:
        """Aggregate processing stats: total docs, chapters, tokens."""
        conn = self._get_conn()
        params = (since_ts,) if since_ts else ()
        where = "WHERE timestamp >= ?" if since_ts else ""

        total = conn.execute(
            f"SELECT COUNT(*), SUM(chunks), SUM(llm_calls), SUM(total_tokens), "
            f"AVG(latency_ms) FROM processing_stats {where}", params,
        ).fetchone()

        by_doc = conn.execute(
            f"SELECT doc_filename, COUNT(*), SUM(chunks), SUM(total_tokens) "
            f"FROM processing_stats {where} GROUP BY doc_filename "
            f"ORDER BY SUM(total_tokens) DESC", params,
        ).fetchall()

        conn.close()
        return {
            "total_operations": total[0] or 0,
            "total_chunks": total[1] or 0,
            "total_llm_calls": total[2] or 0,
            "total_tokens": total[3] or 0,
            "avg_latency_ms": round(total[4] or 0, 1),
            "by_document": [
                {"doc": r[0], "ops": r[1], "chunks": r[2],
                 "total_tokens": r[3]}
                for r in by_doc
            ],
        }


# ── Row helpers ──────────────────────────────────────────────────────────────

def _request_row(r: tuple) -> dict:
    return {
        "id": r[0], "request_id": r[1], "session_id": r[2],
        "question": r[3], "route": r[4], "rounds": r[5],
        "llm_calls": r[6], "prompt_tokens": r[7],
        "completion_tokens": r[8], "tool_calls": r[9],
        "total_latency_ms": r[10], "timestamp": r[11],
    }


def _proc_row(r: tuple) -> dict:
    return {
        "id": r[0], "doc_filename": r[1], "chapter_title": r[2],
        "operation": r[3], "pages": r[4], "chunks": r[5],
        "llm_calls": r[6], "total_tokens": r[7],
        "latency_ms": r[8], "timestamp": r[9],
    }


# ── Singleton ────────────────────────────────────────────────────────────────

_store: Optional[RecorderStore] = None


def get_recorder_store() -> RecorderStore:
    global _store
    if _store is None:
        _store = RecorderStore()
    return _store
