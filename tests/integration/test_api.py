# Integration tests for FastAPI endpoints
#
# Uses TestClient with mocked dependencies (store, config, uploaded_files, etc.)
# to achieve zero-cost API-level coverage.
#
# Mock at router level because routers import e.g. `from .deps import store`
# at module load time — patching deps.xxx doesn't retroactively update the
# already-imported reference in each router.

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── Mock store ─────────────────────────────────────────────────────

def _fake_store():
    """Return a MagicMock that behaves like MemoryStore for API tests."""
    store = MagicMock()
    store.list_sessions.return_value = [
        {"id": "abc123", "topic": "test session", "updated_at": "2026-07-10"}
    ]
    store.get_session.return_value = {"id": "abc123", "topic": "电场学习", "report": "[]"}
    store.get_chat_history.return_value = [
        {"role": "user", "content": "什么是电场？"},
        {"role": "assistant", "content": "电场是..."},
    ]
    store.delete_session.return_value = None
    store.add_chat_message.return_value = None
    store.save_session.return_value = None
    return store


# ── Fixture ────────────────────────────────────────────────────────

@pytest.fixture
def api_client(tmp_path):
    """Create a FastAPI TestClient with all deps mocked at router level."""
    from fastapi.testclient import TestClient

    fake_store_obj = _fake_store()
    fake_config = MagicMock()
    fake_upload_dir = tmp_path / "uploads"
    fake_upload_dir.mkdir()
    fake_uploaded: dict[str, str] = {}

    # Patch at ROUTER level — each router does `from .deps import store`,
    # so it holds its own reference. Patching deps.xxx is too late.
    with patch("src.api.router_sessions.store", fake_store_obj), \
         patch("src.api.router_chat.store", fake_store_obj), \
         patch("src.api.router_files.uploaded_files", fake_uploaded), \
         patch("src.api.router_files.UPLOAD_DIR", fake_upload_dir), \
         patch("src.api.deps.store", fake_store_obj), \
         patch("src.api.deps.config", fake_config), \
         patch("src.api.deps.kg", MagicMock()), \
         patch("src.api.deps.vs", MagicMock()), \
         patch("src.api.deps.chapter_agent", MagicMock()), \
         patch("src.api.deps.chapters_cache", {}), \
         patch("src.api.deps.uploaded_files", fake_uploaded), \
         patch("src.api.deps.UPLOAD_DIR", fake_upload_dir), \
         patch("src.api.deps.PROJECT_ROOT", tmp_path):

        from src.api import app
        client = TestClient(app)
        yield client


# ── Health ────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_ok(self, api_client):
        response = api_client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ── Schemas ───────────────────────────────────────────────────────

class TestSchemas:
    """Tests for Pydantic request models."""

    def test_chat_request_valid(self):
        from src.api.schemas import ChatRequest
        req = ChatRequest(question="什么是库仑定律？", session_id="test", doc_filter=["a.pdf"])
        assert req.question == "什么是库仑定律？"
        assert req.session_id == "test"
        assert req.doc_filter == ["a.pdf"]

    def test_chat_request_defaults(self):
        from src.api.schemas import ChatRequest
        req = ChatRequest(question="hello")
        assert req.session_id == ""
        assert req.doc_filter == []

    def test_chapter_detect_request(self):
        from src.api.schemas import ChapterDetectRequest
        req = ChapterDetectRequest(filepaths=["/tmp/a.pdf", "/tmp/b.pdf"])
        assert len(req.filepaths) == 2

    def test_process_request_defaults(self):
        from src.api.schemas import ProcessRequest
        req = ProcessRequest(filepaths=["/tmp/a.pdf"])
        assert req.selected_chapters == []

    def test_save_chapters_request(self):
        from src.api.schemas import SaveChaptersRequest
        req = SaveChaptersRequest(filename="test.pdf", chapters=[{"title": "Ch1"}])
        assert req.filename == "test.pdf"
        assert len(req.chapters) == 1


# ── Sessions ──────────────────────────────────────────────────────

class TestSessionsEndpoint:
    """Tests for GET/DELETE /api/sessions."""

    def test_list_sessions(self, api_client):
        response = api_client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "abc123"

    def test_get_session(self, api_client):
        response = api_client.get("/api/sessions/abc123")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "abc123"
        assert data["topic"] == "电场学习"
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_get_session_not_found(self, api_client):
        # Override the mock to return None for this session
        import src.api.router_sessions as rs
        rs.store.get_session.return_value = None

        response = api_client.get("/api/sessions/nonexistent")
        assert response.status_code == 404

    def test_delete_session(self, api_client):
        response = api_client.delete("/api/sessions/abc123")
        assert response.status_code == 200
        assert response.json() == {"deleted": "abc123"}


# ── Files ─────────────────────────────────────────────────────────

class TestFilesEndpoint:
    """Tests for POST/GET/DELETE /api/files."""

    def test_list_files_empty(self, api_client):
        response = api_client.get("/api/files/list")
        assert response.status_code == 200
        assert response.json() == {"files": []}

    def test_upload_and_list(self, api_client):
        from io import BytesIO

        content = BytesIO(b"test content for document")
        response = api_client.post(
            "/api/files/upload",
            files=[("files", ("test_doc.pdf", content, "application/pdf"))],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "test_doc.pdf" in data["uploaded"]

        response2 = api_client.get("/api/files/list")
        assert "test_doc.pdf" in response2.json()["files"]

    def test_upload_unsupported_format(self, api_client):
        from io import BytesIO

        content = BytesIO(b"binary data")
        response = api_client.post(
            "/api/files/upload",
            files=[("files", ("movie.mp4", content, "video/mp4"))],
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_delete_file_not_found(self, api_client):
        response = api_client.delete("/api/files/nonexistent.pdf")
        assert response.status_code == 404
