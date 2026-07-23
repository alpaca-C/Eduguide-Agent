# EpisodicMemory — cross-session experience recording and recall
#
# Stores structured episodes (tasks, actions, observations, outcomes, reflections)
# for future recall. NOT a cache — episodes persist and accumulate knowledge.
#
# Storage: SQLite (structured data) + ChromaDB (vector search for semantic recall)
#
# Usage:
#   em = EpisodicMemory()
#   ep_id = em.record({
#       "task": {"goal": "优化RAG", "type": "system_optimization"},
#       "context": {"system": "RAG Agent", "vector_store": "Chroma"},
#       "actions": [{"type": "experiment", "action": "add sparse"}],
#       "observations": ["sparse improves recall"],
#       "outcome": {"decision": "dense+sparse hybrid", "success": True},
#       "reflection": {"lesson": "supplement > replace", "avoid": "heavy reranker"},
#   })
#   episodes = em.recall("RAG检索优化", top_k=5)

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """A single recorded experience in episodic memory.

    Core fields (always populated):
        id, task_goal, task_type, context, actions, observations, outcome,
        reflection, session_id, user_id, created_at

    Diagnosis fields (populated when Reflector provides feedback):
        failure_stage: "router" | "rewriter" | "planner.plan" | "executor"
                      | "planner.solve" | "none"
        insufficiency_type: "plan" | "knowledge" | "reasoning" | ""
        missing_aspects: what the answer was missing
        suggested_queries: Reflector's suggested search terms

    Lesson fields (human-readable insights):
        lesson: one-line takeaway
        corrected_behavior: what to do differently next time
        good_pattern: what went well (positive reinforcement)

    Matching fields (for semantic recall quality):
        question_pattern: abstracted question pattern ("物理公式推导")
        keywords: matching keywords extracted from the question
    """
    id: str
    task_goal: str
    task_type: str = ""
    context: dict = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    reflection: dict = field(default_factory=dict)
    session_id: str = ""
    user_id: str = ""
    created_at: float = 0.0

    # ── Diagnosis (from Reflector verdict) ──
    failure_stage: str = ""
    insufficiency_type: str = ""       # "plan" | "knowledge" | "reasoning"
    missing_aspects: list[str] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)

    # ── Lessons (structured insights) ──
    lesson: str = ""
    corrected_behavior: str = ""
    good_pattern: str = ""

    # ── Matching ──
    question_pattern: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_goal": self.task_goal,
            "task_type": self.task_type,
            "context": self.context,
            "actions": self.actions,
            "observations": self.observations,
            "outcome": self.outcome,
            "reflection": self.reflection,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "failure_stage": self.failure_stage,
            "insufficiency_type": self.insufficiency_type,
            "missing_aspects": self.missing_aspects,
            "suggested_queries": self.suggested_queries,
            "lesson": self.lesson,
            "corrected_behavior": self.corrected_behavior,
            "good_pattern": self.good_pattern,
            "question_pattern": self.question_pattern,
            "keywords": self.keywords,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "Episode":
        """Reconstruct from SQLite row. Handles both old (11-col) and new (20-col) rows."""
        n = len(row)
        return cls(
            id=row[0],
            task_goal=row[1],
            task_type=row[2] if n > 2 else "",
            context=json.loads(row[3]) if n > 3 and row[3] else {},
            actions=json.loads(row[4]) if n > 4 and row[4] else [],
            observations=json.loads(row[5]) if n > 5 and row[5] else [],
            outcome=json.loads(row[6]) if n > 6 and row[6] else {},
            reflection=json.loads(row[7]) if n > 7 and row[7] else {},
            session_id=row[8] if n > 8 else "",
            user_id=row[9] if n > 9 else "",
            created_at=row[10] if n > 10 else 0.0,
            # New fields (columns 11-19, nullable)
            failure_stage=row[11] if n > 11 else "",
            insufficiency_type=row[12] if n > 12 else "",
            missing_aspects=json.loads(row[13]) if n > 13 and row[13] else [],
            suggested_queries=json.loads(row[14]) if n > 14 and row[14] else [],
            lesson=row[15] if n > 15 else "",
            corrected_behavior=row[16] if n > 16 else "",
            good_pattern=row[17] if n > 17 else "",
            question_pattern=row[18] if n > 18 else "",
            keywords=json.loads(row[19]) if n > 19 and row[19] else [],
        )

    def _search_text(self) -> str:
        """Build the searchable text for vector embedding."""
        parts = [self.task_goal]
        parts.extend(self.observations)
        # Prefer structured lesson field (new), fall back to reflection dict (legacy)
        lesson_text = self.lesson or self.reflection.get("lesson", "")
        if lesson_text:
            parts.append(lesson_text)
        if self.corrected_behavior:
            parts.append(self.corrected_behavior)
        if self.question_pattern:
            parts.append(self.question_pattern)
        if self.keywords:
            parts.extend(self.keywords)
        return " ".join(parts)


# ── EpisodicMemory ──────────────────────────────────────────────────────────

class EpisodicMemory:
    """跨 session 的情景记忆 — 记录经验、语义召回。

    SQLite: 完整结构化数据 (episodes 表)
    ChromaDB: 向量索引 (task_goal + observations + reflection.lesson)
    """

    COLLECTION_NAME = "episodes"

    def __init__(self, storage_dir: str = ""):
        if storage_dir:
            persist_dir = Path(storage_dir)
        else:
            persist_dir = Path(__file__).resolve().parent.parent.parent / "data"
        persist_dir.mkdir(parents=True, exist_ok=True)

        # ── SQLite ──
        self._db_path = str(persist_dir / "episodes.db")
        self._init_sqlite()

        # ── ChromaDB ──
        self._chroma_dir = str(persist_dir / "chroma")
        self._ef = None   # lazy init via _ensure_chroma
        self._collection = None
        self._client = None

    # ── SQLite ──────────────────────────────────────────────────────────

    def _init_sqlite(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    task_goal TEXT NOT NULL,
                    task_type TEXT DEFAULT '',
                    context_json TEXT DEFAULT '{}',
                    actions_json TEXT DEFAULT '[]',
                    observations_json TEXT DEFAULT '[]',
                    outcome_json TEXT DEFAULT '{}',
                    reflection_json TEXT DEFAULT '{}',
                    session_id TEXT DEFAULT '',
                    user_id TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)
            # Migrations: add columns that may not exist in older DBs
            migrations = [
                "ALTER TABLE episodes ADD COLUMN user_id TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN failure_stage TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN insufficiency_type TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN missing_aspects_json TEXT DEFAULT '[]'",
                "ALTER TABLE episodes ADD COLUMN suggested_queries_json TEXT DEFAULT '[]'",
                "ALTER TABLE episodes ADD COLUMN lesson TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN corrected_behavior TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN good_pattern TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN question_pattern TEXT DEFAULT ''",
                "ALTER TABLE episodes ADD COLUMN keywords_json TEXT DEFAULT '[]'",
            ]
            for m in migrations:
                try:
                    conn.execute(m)
                except sqlite3.OperationalError:
                    pass  # column already exists
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_task_type ON episodes(task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_created ON episodes(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_user ON episodes(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_failure_stage ON episodes(failure_stage)")
            conn.commit()

    # ── ChromaDB (lazy init, best-effort) ────────────────────────────────

    def _ensure_chroma(self):
        """Initialize ChromaDB collection. Fails gracefully if model unavailable.

        Sets self._collection = False on failure — SQLite still works.
        """
        if self._collection is not None:
            return
        if self._collection is False:  # previously failed, don't retry
            return

        # Quick pre-check: if the embedding model source is a HF model ID
        # (not a local path) and no HF_ENDPOINT mirror is set, skip ChromaDB
        # to avoid hanging on model download.
        model_src = (
            os.environ.get("EMBEDDING_MODEL_PATH")
            or os.environ.get("BGE_M3_MODEL_PATH")
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        if "/" in model_src and not os.path.isdir(model_src):
            # Looks like a HF model ID — need network
            if not os.environ.get("HF_ENDPOINT"):
                logger.info(
                    "EpisodicMemory: no local embedding model found and no HF_ENDPOINT set. "
                    "Skipping ChromaDB (SQLite-only mode). Set BGE_M3_MODEL_PATH to a local "
                    "model directory to enable semantic episode search."
                )
                self._collection = False
                return

        try:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=self._chroma_dir, settings=Settings(anonymized_telemetry=False),
            )
            try:
                self._collection = self._client.get_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self._get_ef(),
                )
                logger.info("EpisodicMemory: ChromaDB collection '%s' loaded (%d episodes)",
                            self.COLLECTION_NAME, self._collection.count())
            except Exception:
                self._collection = self._client.create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self._get_ef(),
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("EpisodicMemory: created ChromaDB collection '%s'", self.COLLECTION_NAME)
        except Exception as e:
            logger.warning("EpisodicMemory: ChromaDB unavailable (%s). SQLite-only mode.", e)
            self._collection = False  # Sentinel: don't retry

    def _get_ef(self):
        if self._ef is not None:
            return self._ef
        # Reuse the same embedding function pattern as DocumentVectorStore
        from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

        class _EpisodeEF(EmbeddingFunction):
            def __init__(self2):
                self2._model_source = (
                    os.environ.get("EMBEDDING_MODEL_PATH")
                    or os.environ.get("BGE_M3_MODEL_PATH")
                    or "Qwen/Qwen3-Embedding-0.6B"
                )
                self2._model = None
                self2._lock = None

            def _ensure(self2):
                if self2._model is not None:
                    return
                import threading
                if self2._lock is None:
                    self2._lock = threading.Lock()
                with self2._lock:
                    if self2._model is not None:
                        return
                    from sentence_transformers import SentenceTransformer
                    logger.info("EpisodicMemory: loading embedding model '%s'...", self2._model_source)
                    self2._model = SentenceTransformer(self2._model_source)
                    logger.info("EpisodicMemory: embedding model ready, dim=%d",
                                self2._model.get_sentence_embedding_dimension())

            def __call__(self2, input: Documents) -> Embeddings:
                self2._ensure()
                return self2._model.encode(
                    input, normalize_embeddings=True, show_progress_bar=False,
                ).tolist()

        self._ef = _EpisodeEF()
        return self._ef

    # ── Public API ──────────────────────────────────────────────────────

    def record(self, episode_dict: dict, session_id: str = "",
               user_id: str = "") -> str:
        """Record a new episode. Returns the episode ID.

        Args:
            episode_dict: {
                "task": {"goal": "...", "type": "..."},
                "context": {...},
                "actions": [{...}],
                "observations": ["...", "..."],
                "outcome": {"decision": "...", "success": true},
                "reflection": {"lesson": "...", "avoid": "..."},

                # ── New diagnosis fields (from Reflector) ──
                "failure_stage": "planner.plan",
                "insufficiency_type": "plan",
                "missing_aspects": ["未解释负号物理意义"],
                "suggested_queries": ["楞次定律 负号"],

                # ── New lesson fields ──
                "lesson": "遇到推导题要保留公式子问题",
                "corrected_behavior": "分解时显式检查公式/推导维度",
                "good_pattern": "Dense 单路就命中了",

                # ── New matching fields ──
                "question_pattern": "物理公式推导",
                "keywords": ["库仑定律", "高斯定理"],
            }
            session_id: Optional session this episode belongs to.
            user_id: Optional user identifier for cross-session recall.

        Returns:
            The generated episode ID (e.g. "ep_001").
        """
        task = episode_dict.get("task", {})
        ep = Episode(
            id=str(uuid.uuid4())[:12],
            task_goal=task.get("goal", ""),
            task_type=task.get("type", ""),
            context=episode_dict.get("context", {}),
            actions=episode_dict.get("actions", []),
            observations=episode_dict.get("observations", []),
            outcome=episode_dict.get("outcome", {}),
            reflection=episode_dict.get("reflection", {}),
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            # New fields
            failure_stage=episode_dict.get("failure_stage", ""),
            insufficiency_type=episode_dict.get("insufficiency_type", ""),
            missing_aspects=episode_dict.get("missing_aspects", []),
            suggested_queries=episode_dict.get("suggested_queries", []),
            lesson=episode_dict.get("lesson", ""),
            corrected_behavior=episode_dict.get("corrected_behavior", ""),
            good_pattern=episode_dict.get("good_pattern", ""),
            question_pattern=episode_dict.get("question_pattern", ""),
            keywords=episode_dict.get("keywords", []),
        )

        # 1) SQLite
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (id, task_goal, task_type, context_json, actions_json,
                    observations_json, outcome_json, reflection_json,
                    session_id, user_id, created_at,
                    failure_stage, insufficiency_type,
                    missing_aspects_json, suggested_queries_json,
                    lesson, corrected_behavior, good_pattern,
                    question_pattern, keywords_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ep.id, ep.task_goal, ep.task_type,
                    json.dumps(ep.context, ensure_ascii=False),
                    json.dumps(ep.actions, ensure_ascii=False),
                    json.dumps(ep.observations, ensure_ascii=False),
                    json.dumps(ep.outcome, ensure_ascii=False),
                    json.dumps(ep.reflection, ensure_ascii=False),
                    ep.session_id, ep.user_id, ep.created_at,
                    # New fields
                    ep.failure_stage, ep.insufficiency_type,
                    json.dumps(ep.missing_aspects, ensure_ascii=False),
                    json.dumps(ep.suggested_queries, ensure_ascii=False),
                    ep.lesson, ep.corrected_behavior, ep.good_pattern,
                    ep.question_pattern,
                    json.dumps(ep.keywords, ensure_ascii=False),
                ),
            )
            conn.commit()

        # 2) ChromaDB (best-effort)
        try:
            self._ensure_chroma()
            if self._collection and self._collection is not False:
                search_text = ep._search_text()
                self._collection.add(
                    ids=[ep.id],
                    documents=[search_text],
                    metadatas=[{
                        "task_goal": ep.task_goal[:200],
                        "task_type": ep.task_type,
                        "session_id": session_id,
                        "user_id": user_id,
                        "failure_stage": ep.failure_stage,
                        "insufficiency_type": ep.insufficiency_type,
                    }],
                )
        except Exception as e:
            logger.warning("EpisodicMemory: ChromaDB upsert failed (SQLite OK): %s", e)

        logger.info("EpisodicMemory: recorded episode '%s' — '%s'", ep.id, ep.task_goal[:60])
        return ep.id

    def recall(self, query: str, top_k: int = 5,
               user_id: str = "") -> list[Episode]:
        """Semantic recall of relevant past episodes.

        Searches ChromaDB for episodes similar to the query,
        then loads full data from SQLite.

        Args:
            query: Natural language query (e.g. "RAG检索优化").
            top_k: Max episodes to return.
            user_id: Optional user filter — only returns this user's episodes.
                     Empty string = no filter (backward compatible).

        Returns:
            List of Episode objects, most relevant first.
        """
        try:
            self._ensure_chroma()
            if not self._collection or self._collection is False:
                return []  # ChromaDB unavailable, no semantic recall

            where_filter = None
            if user_id:
                where_filter = {"user_id": user_id}

            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
                include=["metadatas", "distances"],
            )
            ids = results.get("ids", [[]])[0]
            if not ids:
                logger.debug("EpisodicMemory: no episodes found for '%s'", query[:60])
                return []

            episodes = []
            with sqlite3.connect(self._db_path) as conn:
                for ep_id in ids:
                    row = conn.execute(
                        "SELECT * FROM episodes WHERE id = ?", (ep_id,),
                    ).fetchone()
                    if row:
                        episodes.append(Episode.from_row(row))

            logger.info("EpisodicMemory: recalled %d episodes for '%s'", len(episodes), query[:60])
            return episodes
        except Exception as e:
            logger.warning("EpisodicMemory: recall failed: %s", e)
            return []

    def get(self, episode_id: str) -> Episode | None:
        """Retrieve a single episode by ID."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,),
            ).fetchone()
            if row:
                return Episode.from_row(row)
        return None

    def list(self, task_type: str = "", user_id: str = "",
             limit: int = 20) -> list[Episode]:
        """List recent episodes, optionally filtered by task_type and/or user_id."""
        with sqlite3.connect(self._db_path) as conn:
            conditions = []
            params: list = []
            if task_type:
                conditions.append("task_type = ?")
                params.append(task_type)
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM episodes {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [Episode.from_row(r) for r in rows]

    def delete(self, episode_id: str):
        """Delete an episode from both stores."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            conn.commit()
        try:
            self._ensure_chroma()
            if self._collection and self._collection is not False:
                self._collection.delete(ids=[episode_id])
        except Exception as e:
            logger.warning("EpisodicMemory: ChromaDB delete failed for '%s': %s", episode_id, e)
        logger.info("EpisodicMemory: deleted episode '%s'", episode_id)

    def stats(self) -> dict:
        """Get episode count and type distribution."""
        with sqlite3.connect(self._db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            types = conn.execute(
                "SELECT task_type, COUNT(*) FROM episodes GROUP BY task_type"
            ).fetchall()
        return {
            "total_episodes": total,
            "by_type": {t[0] or "(none)": t[1] for t in types},
        }
