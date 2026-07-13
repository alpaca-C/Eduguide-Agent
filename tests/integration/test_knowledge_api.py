# Integration tests for /api/knowledge endpoints

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ── Fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def knowledge_client(tmp_path):
    """Create a FastAPI TestClient with knowledge router dependencies mocked."""
    from fastapi.testclient import TestClient

    mock_kg = MagicMock()
    mock_kg.stats.return_value = {
        "concepts": 15, "relations": 8,
        "categories": {"definition": 5, "theorem": 3, "method": 7},
    }
    mock_kg.get_doc_names.return_value = ["电磁学.pdf", "量子力学.pdf"]
    mock_kg.get_all_concepts.return_value = []
    mock_kg.clear.return_value = None

    mock_vs = MagicMock()
    mock_vs.index_chunks.return_value = None
    mock_vs.get_imported_chapter_titles.return_value = set()

    mock_chapter_agent = MagicMock()

    fake_store = MagicMock()

    fake_config = MagicMock()
    fake_config.chunk_size = 800
    fake_config.chunk_overlap = 150

    fake_upload_dir = tmp_path / "uploads"
    fake_upload_dir.mkdir()
    fake_uploaded: dict[str, str] = {}
    fake_chapters_cache: dict = {}

    with patch("src.api.router_knowledge.kg", mock_kg), \
         patch("src.api.router_knowledge.vs", mock_vs), \
         patch("src.api.router_knowledge.chapters_cache", fake_chapters_cache), \
         patch("src.api.router_knowledge.UPLOAD_DIR", fake_upload_dir), \
         patch("src.api.router_knowledge.uploaded_files", fake_uploaded), \
         patch("src.api.deps.kg", mock_kg), \
         patch("src.api.deps.vs", mock_vs), \
         patch("src.api.deps.chapter_agent", mock_chapter_agent), \
         patch("src.api.deps.chapters_cache", fake_chapters_cache), \
         patch("src.api.deps.uploaded_files", fake_uploaded), \
         patch("src.api.deps.UPLOAD_DIR", fake_upload_dir), \
         patch("src.api.deps.store", fake_store), \
         patch("src.api.deps.config", fake_config), \
         patch("src.api.deps.PROJECT_ROOT", tmp_path), \
         patch("src.api.router_sessions.store", fake_store), \
         patch("src.api.router_chat.store", fake_store), \
         patch("src.api.router_files.uploaded_files", fake_uploaded), \
         patch("src.api.router_files.UPLOAD_DIR", fake_upload_dir):

        from src.api import app
        yield TestClient(app)


class TestKnowledgeStats:
    """Tests for GET /api/knowledge/stats."""

    def test_stats_returns_knowledge_data(self, knowledge_client):
        response = knowledge_client.get("/api/knowledge/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["concepts"] == 15
        assert data["relations"] == 8
        assert "categories" in data
        assert "documents" in data


class TestKnowledgeDocuments:
    """Tests for GET /api/knowledge/documents."""

    def test_list_documents(self, knowledge_client):
        response = knowledge_client.get("/api/knowledge/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "电磁学.pdf" in data["documents"]
        assert "量子力学.pdf" in data["documents"]


class TestKnowledgeClear:
    """Tests for DELETE /api/knowledge/clear."""

    def test_clear_returns_cleared(self, knowledge_client):
        response = knowledge_client.delete("/api/knowledge/clear")
        assert response.status_code == 200
        assert response.json() == {"status": "cleared"}


class TestKnowledgeDeleteChapter:
    """Tests for DELETE /api/knowledge/chapters/{label}."""

    def test_delete_chapter_invalid_label_returns_400(self, knowledge_client):
        """Invalid label format should return 400."""
        response = knowledge_client.delete("/api/knowledge/chapters/bad-label-no-brackets")
        assert response.status_code == 400
        assert "Invalid label format" in response.json()["detail"]

    def test_delete_chapter_valid_label(self, knowledge_client):
        """Valid label should trigger vs.remove_by_chapter and kg.remove_by_doc."""
        import src.api.router_knowledge as rk
        rk.vs.remove_by_chapter.return_value = 5
        rk.kg.remove_by_doc.return_value = 3

        # URL encode the brackets: [file.pdf] title
        response = knowledge_client.delete(
            "/api/knowledge/chapters/%5B电磁学.pdf%5D%20第一章"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["chunks_removed"] == 5
        assert data["concepts_removed"] == 3


class TestKnowledgeProcess:
    """Tests for POST /api/knowledge/process (SSE streaming)."""

    def test_process_no_files_returns_400(self, knowledge_client):
        """Empty filepaths should return 400."""
        response = knowledge_client.post("/api/knowledge/process", json={
            "filepaths": [],
            "selected_chapters": [],
        })
        assert response.status_code == 400

    def test_process_no_chapters_returns_error_in_sse(self, knowledge_client):
        """No selected chapters should produce an error SSE event."""
        # Need a file to exist for it to get past the file check
        import src.api.router_knowledge as rk
        test_file = rk.UPLOAD_DIR / "test.txt"
        test_file.write_text("test content", encoding="utf-8")
        rk.uploaded_files["test.txt"] = str(test_file)

        response = knowledge_client.post("/api/knowledge/process", json={
            "filepaths": ["test.txt"],
            "selected_chapters": [],
        })
        assert response.status_code == 200

        # Read SSE — should get error about no chapters
        events = []
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                try:
                    events.append(json.loads(line[len("data: "):]))
                except json.JSONDecodeError:
                    pass

        # Should have an error event
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) >= 1
