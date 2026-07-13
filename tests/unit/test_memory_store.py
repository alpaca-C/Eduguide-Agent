# Unit tests for MemoryStore (SQLite + Qdrant hybrid memory)
#
# Tests focus on the SQLite layer. Qdrant paths require `_use_qdrant=True`,
# which is only set when both qdrant_url and qdrant_api_key are provided.
# Passing neither defaults to SQLite-only mode.

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.memory.store import MemoryStore, _hash_query


# ── Mock search result for cache_search_result ────────────────────────

@dataclass
class MockSearchResult:
    """Minimal dataclass matching the shape expected by cache_search_result."""
    title: str
    url: str
    content: str
    score: float


class TestHashQuery:
    """Tests for _hash_query helper."""

    def test_same_query_produces_same_hash(self):
        """Identical queries should produce identical hashes."""
        h1 = _hash_query("什么是机器学习")
        h2 = _hash_query("什么是机器学习")
        assert h1 == h2

    def test_different_queries_produce_different_hashes(self):
        """Different queries should produce different hashes."""
        h1 = _hash_query("机器学习")
        h2 = _hash_query("深度学习")
        assert h1 != h2

    def test_case_insensitive(self):
        """Hashing should be case-insensitive."""
        h1 = _hash_query("Machine Learning")
        h2 = _hash_query("machine learning")
        assert h1 == h2

    def test_whitespace_insensitive(self):
        """Leading/trailing whitespace should not affect hash."""
        h1 = _hash_query("  机器学习  ")
        h2 = _hash_query("机器学习")
        assert h1 == h2

    def test_hash_is_hex_string(self):
        """Hash should be a 16-character hex string."""
        h = _hash_query("test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


class TestMemoryStoreSession:
    """Tests for session CRUD operations."""

    @pytest.fixture
    def store(self):
        """Create a MemoryStore backed by a temporary SQLite database (no Qdrant)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            yield MemoryStore(db_path=db_path)

    # ── save / get / delete session ────────────────────────────────────

    def test_save_and_get_session_basic(self, store):
        """Save a session then retrieve it."""
        store.save_session("s1", "电磁学学习", plan=[{"step": 1}], report="学习报告内容")

        session = store.get_session("s1")
        assert session is not None
        assert session["topic"] == "电磁学学习"
        assert session["plan"] == [{"step": 1}]
        assert session["report"] == "学习报告内容"

    def test_get_nonexistent_session_returns_none(self, store):
        """Getting a non-existent session should return None."""
        assert store.get_session("no-such-session") is None

    def test_save_session_overwrites_existing(self, store):
        """Saving the same session_id twice should overwrite."""
        store.save_session("s1", "第一次", plan=[{"a": 1}], report="旧报告")
        store.save_session("s1", "第二次", plan=[{"b": 2}], report="新报告")

        session = store.get_session("s1")
        assert session["topic"] == "第二次"
        assert session["plan"] == [{"b": 2}]
        assert session["report"] == "新报告"

    def test_save_session_with_none_plan(self, store):
        """Saving with plan=None should not crash."""
        store.save_session("s2", "无计划", plan=None, report="")
        session = store.get_session("s2")
        assert session is not None
        assert session["topic"] == "无计划"
        assert session["plan"] is None
        assert session["report"] == ""

    def test_delete_session_removes_record(self, store):
        """delete_session should remove the session entirely."""
        store.save_session("s1", "测试", plan=[], report="内容")
        store.delete_session("s1")

        assert store.get_session("s1") is None

    def test_delete_session_also_removes_messages(self, store):
        """Deleting a session should also delete its chat messages."""
        store.save_session("s1", "聊天")
        store.add_chat_message("s1", "user", "你好")
        store.add_chat_message("s1", "assistant", "你好！")

        store.delete_session("s1")

        # Re-create the same session
        store.save_session("s1", "新会话")
        history = store.get_chat_history("s1")
        assert history == []

    def test_delete_nonexistent_session_does_not_crash(self, store):
        """Deleting a non-existent session should not raise."""
        store.delete_session("no-such-session")

    # ── list sessions ──────────────────────────────────────────────────

    def test_list_sessions_empty(self, store):
        """list_sessions on empty store returns empty list."""
        assert store.list_sessions() == []

    def test_list_sessions_returns_all(self, store):
        """list_sessions should return all saved sessions."""
        store.save_session("s1", "Topic 1")
        store.save_session("s2", "Topic 2")
        store.save_session("s3", "Topic 3")

        sessions = store.list_sessions()
        assert len(sessions) == 3
        topics = {s["topic"] for s in sessions}
        assert topics == {"Topic 1", "Topic 2", "Topic 3"}

    def test_list_sessions_has_expected_fields(self, store):
        """Each session in the list should have expected fields."""
        store.save_session("abc", "话题")

        sessions = store.list_sessions()
        s = sessions[0]
        assert s["session_id"] == "abc"
        assert s["topic"] == "话题"
        assert "created_at" in s
        assert "updated_at" in s
        assert "report_preview" in s

    def test_list_sessions_report_preview_truncated(self, store):
        """report_preview should be truncated to 200 characters."""
        long_report = "R" * 500
        store.save_session("s1", "长报告", report=long_report)

        sessions = store.list_sessions()
        assert len(sessions[0]["report_preview"]) <= 200

    # ── chat messages ──────────────────────────────────────────────────

    def test_add_and_get_chat_history(self, store):
        """Chat messages should be retrievable in chronological order."""
        store.add_chat_message("s1", "user", "第一问")
        store.add_chat_message("s1", "assistant", "第一答")
        store.add_chat_message("s1", "user", "第二问")
        store.add_chat_message("s1", "assistant", "第二答")

        history = store.get_chat_history("s1")
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "第一问"
        assert history[1]["role"] == "assistant"
        assert history[2]["content"] == "第二问"
        assert history[3]["content"] == "第二答"

    def test_get_chat_history_empty_session(self, store):
        """Chat history for a session with no messages should be empty."""
        store.save_session("s1", "空会话")
        history = store.get_chat_history("s1")
        assert history == []

    def test_chat_history_session_isolation(self, store):
        """Messages from different sessions should not leak."""
        store.add_chat_message("s1", "user", "s1 的消息")
        store.add_chat_message("s2", "user", "s2 的消息")

        h1 = store.get_chat_history("s1")
        h2 = store.get_chat_history("s2")

        assert len(h1) == 1
        assert h1[0]["content"] == "s1 的消息"
        assert len(h2) == 1
        assert h2[0]["content"] == "s2 的消息"

    def test_chat_history_includes_created_at(self, store):
        """Each history entry should include a created_at timestamp."""
        store.add_chat_message("s1", "user", "消息")
        history = store.get_chat_history("s1")
        assert "created_at" in history[0]
        assert isinstance(history[0]["created_at"], float)

    def test_add_chat_message_bumps_updated_at(self, store):
        """Adding a message should update the session's updated_at."""
        store.save_session("s1", "测试")
        before = store.list_sessions()[0]["updated_at"]

        time.sleep(0.01)  # Ensure timestamp difference
        store.add_chat_message("s1", "user", "新消息")

        after = store.list_sessions()[0]["updated_at"]
        assert after > before


class TestMemoryStoreCache:
    """Tests for search cache and plan cache operations."""

    @pytest.fixture
    def store(self):
        """Create a MemoryStore backed by a temporary SQLite database (no Qdrant)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            yield MemoryStore(db_path=db_path)

    # ── search cache ───────────────────────────────────────────────────

    def test_cache_search_result_writes_to_db(self, store):
        """cache_search_result should persist to SQLite."""
        results = [
            MockSearchResult(
                title="Result 1", url="https://example.com/1",
                content="Content of result 1", score=0.95,
            ),
        ]
        store.cache_search_result("什么是深度学习", results, answer="深度学习是...")

        # Verify the data is in SQLite directly
        query_hash = _hash_query("什么是深度学习")
        conn = sqlite3.connect(store._db_path)
        row = conn.execute(
            "SELECT query_text, results_json, created_at, ttl_days FROM search_cache WHERE query_hash = ?",
            (query_hash,),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "什么是深度学习"
        assert row[3] == 7  # default TTL

        payload = json.loads(row[1])
        assert len(payload["results"]) == 1
        assert payload["results"][0]["title"] == "Result 1"
        assert payload["answer"] == "深度学习是..."

    def test_cache_search_result_overwrites_existing(self, store):
        """Caching the same query twice should overwrite (INSERT OR REPLACE)."""
        results1 = [MockSearchResult(title="Old", url="", content="old", score=0.5)]
        results2 = [MockSearchResult(title="New", url="", content="new", score=0.9)]

        store.cache_search_result("same query", results1, answer="old answer")
        store.cache_search_result("same query", results2, answer="new answer")

        query_hash = _hash_query("same query")
        conn = sqlite3.connect(store._db_path)
        row = conn.execute(
            "SELECT results_json FROM search_cache WHERE query_hash = ?",
            (query_hash,),
        ).fetchone()
        conn.close()

        payload = json.loads(row[0])
        assert payload["results"][0]["title"] == "New"
        assert payload["answer"] == "new answer"

    def test_get_cached_search_miss(self, store):
        """Uncached query should return None."""
        result = store.get_cached_search("从未搜索过的问题")
        assert result is None

    def test_get_cached_search_expired(self, store):
        """An expired cache entry should return None."""
        # Manually insert an expired entry
        query_hash = _hash_query("过期查询")
        now = time.time()
        conn = sqlite3.connect(store._db_path)
        conn.execute(
            "INSERT INTO search_cache(query_hash, query_text, results_json, created_at, ttl_days) "
            "VALUES(?, ?, ?, ?, ?)",
            (query_hash, "过期查询",
             json.dumps({"results": [], "answer": ""}),
             now - 10 * 86400,  # 10 days ago
             7),  # TTL 7 days → expired
        )
        conn.commit()
        conn.close()

        result = store.get_cached_search("过期查询")
        assert result is None

    # ── plan cache ─────────────────────────────────────────────────────

    def test_cache_and_get_plan(self, store):
        """Caching a plan then retrieving it should work."""
        plan = [
            {"id": 1, "question": "什么是梯度下降"},
            {"id": 2, "question": "梯度下降的变体有哪些"},
        ]
        store.cache_plan("机器学习基础", plan, background="这是背景资料")

        retrieved_plan, bg = store.get_cached_plan("机器学习基础")
        assert retrieved_plan == plan
        assert bg == "这是背景资料"

    def test_get_cached_plan_miss(self, store):
        """Uncached plan should return None."""
        result = store.get_cached_plan("从未规划过的主题")
        assert result is None

    def test_cache_plan_overwrites_existing(self, store):
        """Caching the same topic twice should overwrite."""
        store.cache_plan("同一主题", [{"step": "old"}], background="旧背景")
        store.cache_plan("同一主题", [{"step": "new"}], background="新背景")

        plan, bg = store.get_cached_plan("同一主题")
        assert plan == [{"step": "new"}]
        assert bg == "新背景"

    def test_cache_plan_empty_background(self, store):
        """Caching with empty background should work."""
        store.cache_plan("测试主题", [{"id": 1}], background="")
        plan, bg = store.get_cached_plan("测试主题")
        assert plan == [{"id": 1}]
        assert bg == ""

    def test_cache_search_with_no_answer(self, store):
        """Caching search without an answer should work."""
        results = [MockSearchResult(title="T", url="u", content="c", score=1.0)]
        store.cache_search_result("query", results, answer="")

        query_hash = _hash_query("query")
        conn = sqlite3.connect(store._db_path)
        row = conn.execute(
            "SELECT results_json FROM search_cache WHERE query_hash = ?",
            (query_hash,),
        ).fetchone()
        conn.close()

        payload = json.loads(row[0])
        assert payload["answer"] == ""


class TestMemoryStoreInit:
    """Tests for MemoryStore initialization."""

    @pytest.fixture
    def store(self):
        """Create a MemoryStore backed by a temporary SQLite database (no Qdrant)."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "test_memory.db")
            yield MemoryStore(db_path=db_path)

    def test_init_creates_db_file(self):
        """Initializing MemoryStore should create the SQLite database file."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "subdir" / "new_memory.db")
            MemoryStore(db_path=db_path)
            assert Path(db_path).exists()

    def test_init_accepts_directory_path(self):
        """Passing a directory path should create memory.db inside it."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = MemoryStore(db_path=tmpdir)
            expected = str(Path(tmpdir) / "memory.db")
            assert Path(expected).exists()

    def test_init_without_qdrant_does_not_use_qdrant(self):
        """Without Qdrant credentials, _use_qdrant should be False."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            store = MemoryStore(db_path=str(Path(tmpdir) / "test.db"))
            assert store._use_qdrant is False

    def test_init_creates_all_tables(self, store):
        """All expected tables should exist after init."""
        conn = sqlite3.connect(store._db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        conn.close()

        table_names = {t[0] for t in tables}
        # All 4 tables should exist
        assert "search_cache" in table_names
        assert "plan_cache" in table_names
        assert "sessions" in table_names
        assert "session_messages" in table_names
