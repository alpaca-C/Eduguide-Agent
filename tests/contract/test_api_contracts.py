"""
API Contract Tests — verify that HTTP response shapes match what the frontend expects.

These tests don't test business logic (that's unit/integration tests).
They test the *contract*: if you change an SSE event type name, a REST field name,
or the shape of a response, these tests break — and you know the frontend will too.

Principle: every change that breaks the frontend should break at least one test here.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _parse_sse(streaming_response) -> list[dict]:
    events = []
    for line in streaming_response.iter_lines():
        if line and line.startswith("data: "):
            try:
                events.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                pass
    return events


def _mock_supervisor_output():
    return type("SupervisorOutput", (), {
        "reply": "库仑定律描述了静止点电荷之间的相互作用力：F = k·q₁q₂/r²。",
        "rounds": 2,
        "tool_calls": 3,
        "route": "moderate",
    })()


@contextmanager
def _chat_test_client(mock_deps):
    """Context manager: yields a TestClient with supervisor mocked.

    Uses yield (not return) so patches stay active during test execution.
    """
    supervisor = MagicMock()
    supervisor.run = AsyncMock(return_value=_mock_supervisor_output())
    supervisor._memory = MagicMock()
    supervisor._memory.short_term = MagicMock()

    with mock_deps.set(
        **{"src.api.router_chat.supervisor": supervisor}
    ).patch():
        from src.api import app
        yield TestClient(app)


@contextmanager
def _rest_test_client(mock_deps, **extra_patches):
    """Context manager: yields a TestClient with basic REST deps mocked."""
    with mock_deps.set(**extra_patches).patch() if extra_patches else mock_deps.patch():
        from src.api import app
        yield TestClient(app)


# ═══════════════════════════════════════════════════════════════════════
# 1. Chat SSE Event Contract
# ═══════════════════════════════════════════════════════════════════════

class TestChatSSEContract:
    """Frontend parses SSE events by type. If any type name changes, frontend breaks."""

    CHAT_EVENT_TYPES = frozenset({"status", "reply_start", "reply_chunk", "done"})
    EVENT_REQUIRED_FIELDS = {
        "status":      frozenset({"type", "text", "session_id"}),
        "reply_start": frozenset({"type", "session_id"}),
        "reply_chunk": frozenset({"type", "text"}),
        "done":        frozenset({"type", "session_id", "rounds", "tool_calls"}),
    }

    @pytest.fixture
    def client(self, mock_deps):
        with _chat_test_client(mock_deps) as c:
            yield c

    def test_event_type_whitelist(self, client):
        """Every SSE event must use only known event types."""
        resp = client.post("/api/chat", json={
            "question": "什么是库仑定律？", "session_id": "ctr-1",
        })
        assert resp.status_code == 200

        for event in _parse_sse(resp):
            etype = event.get("type")
            assert etype in self.CHAT_EVENT_TYPES, (
                f"Unknown SSE event type '{etype}'. "
                f"Frontend won't know how to handle this. "
                f"Known types: {sorted(self.CHAT_EVENT_TYPES)}"
            )

    def test_status_event_has_required_fields(self, client):
        resp = client.post("/api/chat", json={"question": "hello"})
        status_events = [e for e in _parse_sse(resp) if e["type"] == "status"]
        assert len(status_events) >= 1
        for event in status_events:
            missing = self.EVENT_REQUIRED_FIELDS["status"] - set(event.keys())
            assert not missing, f"status event missing fields: {missing}"

    def test_reply_start_event_has_required_fields(self, client):
        resp = client.post("/api/chat", json={"question": "hello"})
        events = [e for e in _parse_sse(resp) if e["type"] == "reply_start"]
        assert len(events) == 1
        missing = self.EVENT_REQUIRED_FIELDS["reply_start"] - set(events[0].keys())
        assert not missing, f"reply_start event missing fields: {missing}"

    def test_reply_chunk_event_has_text_field(self, client):
        resp = client.post("/api/chat", json={"question": "hello"})
        chunk_events = [e for e in _parse_sse(resp) if e["type"] == "reply_chunk"]
        assert len(chunk_events) >= 1
        for event in chunk_events:
            assert "text" in event, "reply_chunk must have 'text'"
            assert isinstance(event["text"], str)

    def test_done_event_has_metadata_fields(self, client):
        resp = client.post("/api/chat", json={"question": "hello"})
        done_events = [e for e in _parse_sse(resp) if e["type"] == "done"]
        assert len(done_events) == 1
        done = done_events[0]
        missing = self.EVENT_REQUIRED_FIELDS["done"] - set(done.keys())
        assert not missing, f"done event missing fields: {missing}"
        assert isinstance(done["rounds"], int), "rounds must be int"
        assert isinstance(done["tool_calls"], int), "tool_calls must be int"
        assert isinstance(done["session_id"], str), "session_id must be str"

    def test_done_is_the_last_event(self, client):
        resp = client.post("/api/chat", json={"question": "hello"})
        events = _parse_sse(resp)
        assert events[-1]["type"] == "done", (
            f"Last event must be 'done', got '{events[-1]['type']}'"
        )

    def test_events_arrive_in_expected_order(self, client):
        resp = client.post("/api/chat", json={"question": "hello"})
        types = [e["type"] for e in _parse_sse(resp)]
        canonical = [t for t in types if t != "status"]
        assert canonical[0] == "reply_start", f"First non-status event: {canonical[0]}"
        assert canonical[-1] == "done"
        assert "reply_chunk" in canonical, "Must have at least one reply_chunk"


# ═══════════════════════════════════════════════════════════════════════
# 2. Session REST Contract
# ═══════════════════════════════════════════════════════════════════════

class TestSessionContract:
    """Frontend reads session list and detail shapes."""

    LIST_RESPONSE_KEYS = frozenset({"sessions"})
    DETAIL_RESPONSE_KEYS = frozenset({"session_id", "topic", "report", "messages"})
    MESSAGE_KEYS = frozenset({"role", "content"})
    DELETE_RESPONSE_KEYS = frozenset({"deleted"})

    @pytest.fixture
    def client(self, mock_deps):
        mock_deps.store.list_sessions.return_value = [
            {"session_id": "s1", "topic": "test", "created_at": "2026-01-01",
             "message_count": 2},
        ]
        mock_deps.store.get_session.return_value = {
            "session_id": "s1", "topic": "测试", "report": "{}",
        }
        mock_deps.store.get_chat_history.return_value = [
            {"role": "user", "content": "hello"},
        ]
        mock_deps.store.delete_session.return_value = None

        with mock_deps.patch():
            from src.api import app
            yield TestClient(app)

    def test_list_sessions_shape(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        missing = self.LIST_RESPONSE_KEYS - set(data.keys())
        assert not missing, f"list sessions response missing keys: {missing}"

    def test_get_session_shape(self, client):
        resp = client.get("/api/sessions/s1")
        assert resp.status_code == 200
        data = resp.json()
        missing = self.DETAIL_RESPONSE_KEYS - set(data.keys())
        assert not missing, f"get session response missing keys: {missing}"
        for msg in data.get("messages", []):
            msg_missing = self.MESSAGE_KEYS - set(msg.keys())
            assert not msg_missing, f"message missing keys: {msg_missing}"

    def test_delete_session_shape(self, client):
        resp = client.delete("/api/sessions/s1")
        assert resp.status_code == 200
        data = resp.json()
        missing = self.DELETE_RESPONSE_KEYS - set(data.keys())
        assert not missing, f"delete session response missing keys: {missing}"
        assert isinstance(data["deleted"], str)


# ═══════════════════════════════════════════════════════════════════════
# 3. Health & Monitoring Contract
# ═══════════════════════════════════════════════════════════════════════

class TestHealthContract:
    """Monitoring / load-balancer / frontend startup probes."""

    @pytest.fixture
    def client(self, mock_deps):
        mock_deps.store.list_sessions.return_value = []
        with mock_deps.patch():
            from src.api import app
            yield TestClient(app)

    def test_health_returns_status_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_monitoring_stats_shape(self, client):
        resp = client.get("/api/monitoring/stats")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("request", "processing"):
            assert key in data, f"monitoring stats must have '{key}' key"


# ═══════════════════════════════════════════════════════════════════════
# 4. Files REST Contract
# ═══════════════════════════════════════════════════════════════════════

class TestFilesContract:
    """Frontend file list / upload / delete."""

    @pytest.fixture
    def client(self, mock_deps, tmp_path):
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(exist_ok=True)
        (upload_dir / "test.pdf").write_text("fake pdf content")
        fake_uploaded = {"test.pdf": str(upload_dir / "test.pdf")}

        with mock_deps.set(
            **{
                "src.api.router_files.UPLOAD_DIR": upload_dir,
                "src.api.router_files.uploaded_files": fake_uploaded,
                "src.api.deps.uploaded_files": fake_uploaded,
            }
        ).patch():
            from src.api import app
            yield TestClient(app)

    def test_list_files_shape(self, client):
        resp = client.get("/api/files/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data, "files list must have 'files' key"
        assert isinstance(data["files"], list)

    def test_delete_file_shape(self, client):
        resp = client.delete("/api/files/test.pdf")
        assert resp.status_code == 200
        data = resp.json()
        assert "deleted" in data, "delete response must have 'deleted' key"
        assert "remaining" in data, "delete response must have 'remaining' key"


# ═══════════════════════════════════════════════════════════════════════
# 5. Knowledge REST Contract
# ═══════════════════════════════════════════════════════════════════════

class TestKnowledgeContract:
    """Frontend knowledge management page."""

    STATS_KEYS = frozenset({"concepts", "relations", "categories", "documents"})

    @pytest.fixture
    def client(self, mock_deps):
        with mock_deps.patch():
            from src.api import app
            yield TestClient(app)

    def test_stats_shape(self, client):
        resp = client.get("/api/knowledge/stats")
        assert resp.status_code == 200
        data = resp.json()
        for key in self.STATS_KEYS:
            assert key in data, f"knowledge stats missing key: '{key}'"
        assert isinstance(data["concepts"], int)
        assert isinstance(data["relations"], int)
        assert isinstance(data["documents"], list)

    def test_list_documents_shape(self, client):
        resp = client.get("/api/knowledge/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data, "documents list must have 'documents' key"
        assert isinstance(data["documents"], list)


# ═══════════════════════════════════════════════════════════════════════
# 6. Chapter Detect SSE Contract
# ═══════════════════════════════════════════════════════════════════════

class TestChapterDetectContract:
    """Frontend reads chapter detection progress via SSE."""

    COMPLETE_EVENT_KEYS = frozenset({
        "type", "chapters", "total", "files_processed", "total_chapters_found",
    })
    CHAPTER_KEYS = frozenset({
        "label", "title", "filename", "level", "text_preview", "text_length",
    })

    @pytest.fixture
    def client(self, mock_deps, tmp_path):
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(exist_ok=True)
        fake_uploaded = {}
        for name in ("physics.pdf", "math.pdf"):
            path = upload_dir / name
            path.write_text(f"%PDF-1.4 fake content for {name}")
            fake_uploaded[name] = str(path)

        with mock_deps.set(
            **{
                "src.api.deps.UPLOAD_DIR": upload_dir,
                "src.api.deps.uploaded_files": fake_uploaded,
                "src.api.router_chapters.uploaded_files": fake_uploaded,
                "src.api.router_chapters.UPLOAD_DIR": upload_dir,
                "src.api.router_chapters.chapters_cache": {},
            }
        ).patch():
            from src.api import app
            yield TestClient(app)

    def test_detect_returns_empty_list_for_no_files(self, client):
        """If no files, returns plain JSON with empty chapters."""
        resp = client.post("/api/chapters/detect", json={"filepaths": []})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chapters"] == []

    def test_detect_with_files_ends_with_complete_event(self, client):
        """With files, last SSE event must be 'complete'."""
        resp = client.post("/api/chapters/detect", json={
            "filepaths": ["physics.pdf", "math.pdf"],
        })
        assert resp.status_code == 200
        events = _parse_sse(resp)
        assert len(events) >= 1, "Should have at least a 'complete' event"
        assert events[-1]["type"] == "complete", (
            f"Last event must be 'complete', got '{events[-1].get('type')}'"
        )

    def test_complete_event_has_required_fields(self, client):
        resp = client.post("/api/chapters/detect", json={
            "filepaths": ["physics.pdf", "math.pdf"],
        })
        events = _parse_sse(resp)
        complete = events[-1]
        missing = self.COMPLETE_EVENT_KEYS - set(complete.keys())
        assert not missing, f"complete event missing keys: {missing}"

    def test_chapter_entry_has_required_fields(self, client):
        resp = client.post("/api/chapters/detect", json={
            "filepaths": ["physics.pdf", "math.pdf"],
        })
        events = _parse_sse(resp)
        complete = events[-1]
        for chapter in complete.get("chapters", []):
            missing = self.CHAPTER_KEYS - set(chapter.keys())
            assert not missing, f"Chapter entry missing keys: {missing}"


# ═══════════════════════════════════════════════════════════════════════
# 7. Chat Request Schema Contract
# ═══════════════════════════════════════════════════════════════════════

class TestChatRequestContract:
    """Verify request validation — what the frontend sends must be accepted."""

    @pytest.fixture
    def client(self, mock_deps):
        with _chat_test_client(mock_deps) as c:
            yield c

    def test_empty_question_rejected(self, client):
        resp = client.post("/api/chat", json={"question": "   "})
        assert resp.status_code == 400

    def test_unknown_field_is_ignored_not_rejected(self, client):
        """Extra fields must not crash the server (Pydantic ignores extras)."""
        resp = client.post("/api/chat", json={
            "question": "hello",
            "unknown_future_field": "should be ignored",
        })
        assert resp.status_code == 200, (
            "Adding a new optional field should not break existing requests"
        )

    def test_tutor_mode_accepted(self, client):
        resp = client.post("/api/chat", json={
            "question": "这道题怎么做",
            "tutor_mode": True,
        })
        assert resp.status_code == 200

    def test_doc_filter_accepted(self, client):
        resp = client.post("/api/chat", json={
            "question": "什么是梯度下降",
            "doc_filter": ["ml_book.pdf"],
        })
        assert resp.status_code == 200
