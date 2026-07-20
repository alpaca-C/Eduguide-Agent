"""
Integration tests for FastAPI endpoints.

Uses TestClient with mocked dependencies via shared mock_deps fixture
(see tests/conftest.py). Each test gets a clean TestClient with all
deps mocked at the router level.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ── Fixture ────────────────────────────────────────────────────────

@pytest.fixture
def client(mock_deps):
    """Create a TestClient with all deps mocked (zero-cost API coverage)."""
    mock_deps.store.list_sessions.return_value = [
        {"id": "abc123", "topic": "test session", "updated_at": "2026-07-10"},
    ]
    mock_deps.store.get_session.return_value = {
        "id": "abc123", "topic": "电场学习", "report": "[]",
    }
    mock_deps.store.get_chat_history.return_value = [
        {"role": "user", "content": "什么是电场？"},
        {"role": "assistant", "content": "电场是..."},
    ]
    mock_deps.store.delete_session.return_value = None
    mock_deps.store.add_chat_message.return_value = None
    mock_deps.store.save_session.return_value = None

    with mock_deps.patch():
        from src.api import app
        yield TestClient(app)


# ── Health ────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Schemas ───────────────────────────────────────────────────────

class TestSchemas:
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
    def test_list_sessions(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "abc123"

    def test_get_session(self, client):
        resp = client.get("/api/sessions/abc123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "abc123"
        assert len(data["messages"]) == 2

    def test_get_session_not_found(self, client):
        import src.api.router_sessions as rs
        rs.store.get_session.return_value = None
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client):
        resp = client.delete("/api/sessions/abc123")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": "abc123"}


# ── Files ─────────────────────────────────────────────────────────

class TestFilesEndpoint:
    def test_list_files_empty(self, client):
        resp = client.get("/api/files/list")
        assert resp.status_code == 200
        assert resp.json() == {"files": []}

    def test_upload_and_list(self, client):
        content = BytesIO(b"test content for document")
        resp = client.post(
            "/api/files/upload",
            files=[("files", ("test_doc.pdf", content, "application/pdf"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert "test_doc.pdf" in data["uploaded"]

        resp2 = client.get("/api/files/list")
        assert "test_doc.pdf" in resp2.json()["files"]

    def test_upload_unsupported_format(self, client):
        content = BytesIO(b"binary data")
        resp = client.post(
            "/api/files/upload",
            files=[("files", ("movie.mp4", content, "video/mp4"))],
        )
        assert resp.status_code == 400

    def test_delete_file_not_found(self, client):
        resp = client.delete("/api/files/nonexistent.pdf")
        assert resp.status_code == 404
