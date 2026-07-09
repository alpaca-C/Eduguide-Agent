# Token usage storage -- SQLite + JSON fallback

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TokenRecord:
    """A single token usage record from one LLM call."""
    id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_type: str = ""  # e.g. "qa_answerer", "qa_reviewer", "extractor"
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class TokenStore:
    """Thread-safe token usage store backed by SQLite."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = str(Path(__file__).resolve().parent.parent.parent / "storage" / "token_usage.db")
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
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                call_type TEXT NOT NULL DEFAULT '',
                duration_ms REAL NOT NULL DEFAULT 0.0,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_ts ON token_usage(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_token_model ON token_usage(model)
        """)
        conn.commit()
        conn.close()

    def insert(self, record: TokenRecord) -> int:
        """Insert a token usage record. Returns the row id."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                """INSERT INTO token_usage
                   (timestamp, model, prompt_tokens, completion_tokens, total_tokens,
                    call_type, duration_ms, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.timestamp,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.call_type,
                    record.duration_ms,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
            row_id = cur.lastrowid
            conn.commit()
            conn.close()
            return row_id

    def stats(self, since_ts: float = 0.0) -> dict:
        """Aggregate token usage stats since given timestamp."""
        conn = self._get_conn()
        params = (since_ts,) if since_ts else ()
        where = "WHERE timestamp >= ?" if since_ts else ""

        # Overall
        row = conn.execute(
            f"SELECT COUNT(*), SUM(total_tokens), SUM(prompt_tokens), SUM(completion_tokens) FROM token_usage {where}",
            params,
        ).fetchone()
        overall = {
            "calls": row[0] or 0,
            "total_tokens": row[1] or 0,
            "prompt_tokens": row[2] or 0,
            "completion_tokens": row[3] or 0,
        }

        # By model
        by_model_rows = conn.execute(
            f"SELECT model, COUNT(*), SUM(total_tokens) FROM token_usage {where} GROUP BY model ORDER BY SUM(total_tokens) DESC",
            params,
        ).fetchall()
        by_model = [{"model": r[0], "calls": r[1], "total_tokens": r[2]} for r in by_model_rows]

        # By call_type
        by_type_rows = conn.execute(
            f"SELECT call_type, COUNT(*), SUM(total_tokens) FROM token_usage {where} GROUP BY call_type ORDER BY SUM(total_tokens) DESC",
            params,
        ).fetchall()
        by_type = [{"call_type": r[0] or "unknown", "calls": r[1], "total_tokens": r[2]} for r in by_type_rows]

        conn.close()
        return {"overall": overall, "by_model": by_model, "by_call_type": by_type}

    def recent(self, limit: int = 50) -> list[dict]:
        """Get most recent token usage records."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM token_usage ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "model": r[2],
                "prompt_tokens": r[3],
                "completion_tokens": r[4],
                "total_tokens": r[5],
                "call_type": r[6],
                "duration_ms": r[7],
                "metadata": json.loads(r[8]) if r[8] else {},
            }
            for r in rows
        ]


# Global singleton
_store: Optional[TokenStore] = None


def get_token_store() -> TokenStore:
    """Get or create the global token store."""
    global _store
    if _store is None:
        _store = TokenStore()
    return _store
