# Unit tests for TokenStore (monitoring token usage storage)

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from src.monitoring.store import TokenRecord, TokenStore, get_token_store


class TestTokenRecord:
    """Tests for TokenRecord dataclass."""

    def test_default_values(self):
        """Default field values should be set correctly."""
        record = TokenRecord()
        assert record.id is None
        assert record.model == ""
        assert record.prompt_tokens == 0
        assert record.completion_tokens == 0
        assert record.total_tokens == 0
        assert record.call_type == ""
        assert record.duration_ms == 0.0
        assert record.metadata == {}
        assert isinstance(record.timestamp, float)

    def test_full_constructor(self):
        """All fields should be settable via constructor."""
        ts = time.time()
        record = TokenRecord(
            id=1,
            timestamp=ts,
            model="gpt-4",
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            call_type="qa_answerer",
            duration_ms=1234.5,
            metadata={"source": "test"},
        )
        assert record.model == "gpt-4"
        assert record.total_tokens == 700
        assert record.call_type == "qa_answerer"
        assert record.metadata == {"source": "test"}


class TestTokenStore:
    """Tests for TokenStore SQLite CRUD operations."""

    @pytest.fixture
    def store(self):
        """Create a TokenStore with a temporary database."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "test_tokens.db")
            yield TokenStore(db_path=db_path)

    # ── insert ────────────────────────────────────────────────────────

    def test_insert_returns_row_id(self, store):
        """insert() should return an integer row ID."""
        row_id = store.insert(TokenRecord(model="gpt-4", total_tokens=100))
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_insert_multiple_returns_increasing_ids(self, store):
        """Subsequent inserts should return increasing row IDs."""
        id1 = store.insert(TokenRecord(model="a", total_tokens=10))
        id2 = store.insert(TokenRecord(model="b", total_tokens=20))
        id3 = store.insert(TokenRecord(model="c", total_tokens=30))
        assert id1 < id2 < id3

    # ── stats ─────────────────────────────────────────────────────────

    def test_stats_empty_store(self, store):
        """stats() on empty store should return zeros."""
        s = store.stats()
        assert s["overall"]["calls"] == 0
        assert s["overall"]["total_tokens"] == 0
        assert s["by_model"] == []
        assert s["by_call_type"] == []

    def test_stats_aggregates_correctly(self, store):
        """stats() should correctly sum tokens and count calls."""
        store.insert(TokenRecord(model="gpt-4", total_tokens=100, prompt_tokens=60,
                                  completion_tokens=40, call_type="qa"))
        store.insert(TokenRecord(model="gpt-4", total_tokens=200, prompt_tokens=120,
                                  completion_tokens=80, call_type="qa"))
        store.insert(TokenRecord(model="claude", total_tokens=50, prompt_tokens=30,
                                  completion_tokens=20, call_type="extract"))

        s = store.stats()
        assert s["overall"]["calls"] == 3
        assert s["overall"]["total_tokens"] == 350
        assert s["overall"]["prompt_tokens"] == 210
        assert s["overall"]["completion_tokens"] == 140

    def test_stats_by_model(self, store):
        """stats() should group correctly by model."""
        store.insert(TokenRecord(model="gpt-4", total_tokens=100, call_type="qa"))
        store.insert(TokenRecord(model="gpt-4", total_tokens=300, call_type="qa"))
        store.insert(TokenRecord(model="claude", total_tokens=50, call_type="extract"))

        s = store.stats()
        by_model = {m["model"]: m for m in s["by_model"]}
        assert by_model["gpt-4"]["calls"] == 2
        assert by_model["gpt-4"]["total_tokens"] == 400
        assert by_model["claude"]["calls"] == 1
        assert by_model["claude"]["total_tokens"] == 50

    def test_stats_by_call_type(self, store):
        """stats() should group correctly by call_type."""
        store.insert(TokenRecord(model="gpt-4", total_tokens=100, call_type="qa"))
        store.insert(TokenRecord(model="gpt-4", total_tokens=200, call_type="qa"))
        store.insert(TokenRecord(model="gpt-4", total_tokens=50, call_type="extract"))

        s = store.stats()
        by_type = {t["call_type"]: t for t in s["by_call_type"]}
        assert by_type["qa"]["calls"] == 2
        assert by_type["qa"]["total_tokens"] == 300
        assert by_type["extract"]["calls"] == 1
        assert by_type["extract"]["total_tokens"] == 50

    def test_stats_since_timestamp(self, store):
        """stats(since_ts) should filter records by timestamp."""
        now = time.time()
        # Insert an old record
        store.insert(TokenRecord(
            timestamp=now - 7200, model="old-model", total_tokens=100, call_type="old"
        ))
        # Insert a recent record
        store.insert(TokenRecord(
            timestamp=now, model="new-model", total_tokens=200, call_type="new"
        ))

        # Only recent (within last hour)
        s = store.stats(since_ts=now - 3600)
        assert s["overall"]["calls"] == 1
        assert s["overall"]["total_tokens"] == 200

        # All records
        s_all = store.stats(since_ts=0.0)
        assert s_all["overall"]["calls"] == 2

    # ── recent ────────────────────────────────────────────────────────

    def test_recent_empty_store(self, store):
        """recent() on empty store should return empty list."""
        assert store.recent() == []

    def test_recent_returns_most_recent_first(self, store):
        """recent() should return records ordered by timestamp DESC."""
        now = time.time()
        store.insert(TokenRecord(timestamp=now - 100, model="oldest", total_tokens=10))
        store.insert(TokenRecord(timestamp=now - 50, model="middle", total_tokens=20))
        store.insert(TokenRecord(timestamp=now, model="newest", total_tokens=30))

        records = store.recent()
        assert len(records) == 3
        assert records[0]["model"] == "newest"
        assert records[1]["model"] == "middle"
        assert records[2]["model"] == "oldest"

    def test_recent_respects_limit(self, store):
        """recent(limit=N) should return at most N records."""
        for i in range(10):
            store.insert(TokenRecord(model=f"model-{i}", total_tokens=i))

        records = store.recent(limit=3)
        assert len(records) == 3

    def test_recent_record_fields(self, store):
        """Each record from recent() should have all expected fields."""
        store.insert(TokenRecord(
            model="gpt-4", total_tokens=123, call_type="qa",
            prompt_tokens=100, completion_tokens=23, duration_ms=500.0,
            metadata={"key": "val"},
        ))
        records = store.recent()
        r = records[0]
        assert r["model"] == "gpt-4"
        assert r["total_tokens"] == 123
        assert r["call_type"] == "qa"
        assert r["prompt_tokens"] == 100
        assert r["completion_tokens"] == 23
        assert r["duration_ms"] == 500.0
        assert r["metadata"] == {"key": "val"}
        assert "id" in r
        assert "timestamp" in r


class TestGetTokenStore:
    """Tests for the global singleton accessor."""

    def test_get_token_store_returns_same_instance(self):
        """get_token_store() should return the same singleton."""
        store1 = get_token_store()
        store2 = get_token_store()
        assert store1 is store2

    def test_get_token_store_creates_db(self):
        """get_token_store() should create a working store."""
        store = get_token_store()
        assert store.db_path is not None
        # Should be able to insert and query
        store.insert(TokenRecord(model="test", total_tokens=1))
        assert store.stats()["overall"]["calls"] >= 1
