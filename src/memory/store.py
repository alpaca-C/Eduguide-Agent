# Memory Store — SQLite + Qdrant 双存储，提供搜索缓存和会话记忆

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports
_qdrant_client = None
_embedding_model = None

def _get_qdrant_client(url: str, api_key: str):
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=url, api_key=api_key, timeout=20)
    return _qdrant_client

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model

def _embed(text: str) -> list[float]:
    """将文本转为向量嵌入（384 维）。"""
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()

def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


class MemoryStore:
    """SQLite + Qdrant 混合记忆存储。

    SQLite: 精确匹配缓存（查询哈希 → 搜索结果 / 规划）
    Qdrant: 语义相似度缓存（跨会话复用）
    """

    SEARCH_COLLECTION = "search_cache"
    PLAN_COLLECTION = "plan_cache"
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2

    def __init__(self, db_path: str = "", qdrant_url: str = "", qdrant_api_key: str = ""):
        if db_path:
            p = Path(db_path)
            if p.is_dir() or p.suffix == "":
                p = p / "memory.db"
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(p)
        else:
            default_dir = Path(__file__).resolve().parent.parent.parent / "data"
            default_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(default_dir / "memory.db")
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._use_qdrant = bool(qdrant_url and qdrant_api_key)

        self._init_sqlite()
        if self._use_qdrant:
            self._init_qdrant()

    # ==================================================================
    # SQLite
    # ==================================================================
    def _init_sqlite(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_days INTEGER NOT NULL DEFAULT 7
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_cache (
                    topic_hash TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    background_json TEXT,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    plan_json TEXT,
                    report TEXT,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_search_created ON search_cache(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plan_created ON plan_cache(created_at)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_session ON session_messages(session_id)")
            # Migration: add updated_at column for older DBs
            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN updated_at REAL")
            except sqlite3.OperationalError:
                pass
            conn.commit()

    # ==================================================================
    # Qdrant
    # ==================================================================
    def _init_qdrant(self):
        try:
            logger.info("Connecting to Qdrant Cloud...")
            client = _get_qdrant_client(self._qdrant_url, self._qdrant_api_key)
            from qdrant_client.models import Distance, VectorParams
            for name in [self.SEARCH_COLLECTION, self.PLAN_COLLECTION]:
                try:
                    client.get_collection(name)
                    logger.info("Qdrant collection exists: %s", name)
                except Exception:
                    client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
                    )
                    logger.info("Created Qdrant collection: %s", name)
            logger.info("Qdrant connected successfully")
        except Exception as e:
            logger.warning("Qdrant init failed, falling back to SQLite-only: %s", e)
            self._use_qdrant = False

    # ==================================================================
    # Search Cache
    # ==================================================================
    def cache_search_result(self, query: str, results: list, answer: str = ""):
        """缓存搜索结果到 SQLite + Qdrant。"""
        query_hash = _hash_query(query)
        payload = {
            "results": [{"title": r.title, "url": r.url, "content": r.content, "score": r.score} for r in results],
            "answer": answer,
        }
        now = time.time()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO search_cache(query_hash, query_text, results_json, created_at) VALUES(?, ?, ?, ?)",
                (query_hash, query, json.dumps(payload, ensure_ascii=False), now),
            )
            conn.commit()

        if self._use_qdrant:
            try:
                client = _get_qdrant_client(self._qdrant_url, self._qdrant_api_key)
                from qdrant_client.models import PointStruct
                vector = _embed(query)
                client.upsert(
                    collection_name=self.SEARCH_COLLECTION,
                    points=[PointStruct(id=query_hash, vector=vector, payload={"query": query, "ts": now})],
                )
            except Exception as e:
                logger.warning("Qdrant search cache upsert failed: %s", e)

    def get_cached_search(self, query: str) -> Optional[tuple[list, str]]:
        """查询搜索缓存，先精确匹配 SQLite，再语义搜索 Qdrant。"""
        query_hash = _hash_query(query)
        now = time.time()

        # 1) SQLite 精确匹配
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT results_json, created_at, ttl_days FROM search_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
            if row:
                results_json, created_at, ttl_days = row
                if now - created_at < ttl_days * 86400:
                    payload = json.loads(results_json)
                    results = []
                    for r in payload.get("results", []):
                        from ..memory.schemas import SearchResult
                        results.append(SearchResult(**r))
                    logger.info("search cache HIT (exact): %s", query[:50])
                    return results, payload.get("answer", "")

        # 2) Qdrant 语义搜索（已禁用同步读取，避免跨洲延迟阻塞）
        # 如需启用，取消下方注释。建议改用国内的向量数据库（如 Zilliz Cloud）

        return None

    # ==================================================================
    # Plan Cache
    # ==================================================================
    def cache_plan(self, topic: str, plan: list[dict], background: str = ""):
        """缓存规划结果。"""
        topic_hash = _hash_query(topic)
        now = time.time()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO plan_cache(topic_hash, topic, plan_json, background_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (topic_hash, topic, json.dumps(plan, ensure_ascii=False), json.dumps({"bg": background}, ensure_ascii=False), now),
            )
            conn.commit()

        if self._use_qdrant:
            try:
                client = _get_qdrant_client(self._qdrant_url, self._qdrant_api_key)
                from qdrant_client.models import PointStruct
                vector = _embed(topic)
                client.upsert(
                    collection_name=self.PLAN_COLLECTION,
                    points=[PointStruct(id=topic_hash, vector=vector, payload={"topic": topic, "ts": now, "n_blocks": len(plan)})],
                )
            except Exception as e:
                logger.warning("Qdrant plan cache upsert failed: %s", e)

    def get_cached_plan(self, topic: str) -> Optional[tuple[list[dict], str]]:
        """查询缓存的规划结果。返回 (plan_items, background) 或 None。"""
        topic_hash = _hash_query(topic)

        # 1) SQLite
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT plan_json, background_json FROM plan_cache WHERE topic_hash = ?",
                (topic_hash,),
            ).fetchone()
            if row:
                plan = json.loads(row[0])
                bg = json.loads(row[1]).get("bg", "")
                logger.info("plan cache HIT (exact): %s", topic[:50])
                return plan, bg

        # 2) Qdrant 语义搜索（已禁用同步读取）

        return None

    # ==================================================================
    # Session
    # ==================================================================
    def save_session(self, session_id: str, topic: str, plan: list[dict] | None = None, report: str = ""):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions(session_id, topic, plan_json, report, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
                (session_id, topic, json.dumps(plan, ensure_ascii=False) if plan else None, report, time.time(), time.time()),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT topic, plan_json, report FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return {
                    "topic": row[0],
                    "plan": json.loads(row[1]) if row[1] else None,
                    "report": row[2],
                }
        return None
    def delete_session(self, session_id: str):
        """Delete a session and its chat messages."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def list_sessions(self) -> list[dict]:
        """List all saved sessions ordered by updated_at descending."""
        with sqlite3.connect(self._db_path) as conn:
            # Ensure updated_at column exists (migration for old DBs)
            try:
                rows = conn.execute(
                    "SELECT session_id, topic, created_at, updated_at, report FROM sessions ORDER BY COALESCE(updated_at, created_at) DESC"
                ).fetchall()
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE sessions ADD COLUMN updated_at REAL")
                conn.commit()
                rows = conn.execute(
                    "SELECT session_id, topic, created_at, updated_at, report FROM sessions ORDER BY COALESCE(updated_at, created_at) DESC"
                ).fetchall()
        return [
            {
                "session_id": row[0],
                "topic": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "report_preview": (row[4] or "")[:200],
            }
            for row in rows
        ]

    def add_chat_message(self, session_id: str, role: str, content: str):
        """Add a chat message to a session and bump updated_at."""
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO session_messages(session_id, role, content, created_at) VALUES(?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()

    def get_chat_history(self, session_id: str) -> list[dict]:
        """Get chat history for a session ordered by time."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM session_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [{"role": row[0], "content": row[1], "created_at": row[2]} for row in rows]

