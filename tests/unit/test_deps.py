# Unit tests for api/deps.py — chapter DB helpers and app context

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.deps import (
    _get_chapters_conn,
    _load_chapters,
    _save_chapters,
    get_app_context,
    CHAPTERS_DB,
)


class TestChapterDB:
    """Tests for _load_chapters and _save_chapters helpers."""

    @pytest.fixture(autouse=True)
    def _temp_chapters_db(self, tmp_path, monkeypatch):
        """Redirect CHAPTERS_DB to a temp path for test isolation."""
        test_db = tmp_path / "test_chapters.db"
        monkeypatch.setattr(
            "src.api.deps.CHAPTERS_DB", test_db,
        )
        # Also patch the CHAPTERS_DB used by _get_chapters_conn
        import src.api.deps as deps_module
        original_db = deps_module.CHAPTERS_DB
        deps_module.CHAPTERS_DB = test_db
        yield
        deps_module.CHAPTERS_DB = original_db

    def test_save_and_load_single_file(self):
        """Save chapters for one file, then load them back."""
        chapters = [
            {"title": "第一章", "start_page": 1, "end_page": 10},
            {"title": "第二章", "start_page": 11, "end_page": 20},
        ]
        _save_chapters("test.pdf", chapters)

        loaded = _load_chapters("test.pdf")
        assert len(loaded) == 2
        assert loaded[0]["title"] == "第一章"
        assert loaded[1]["title"] == "第二章"

    def test_load_nonexistent_file_returns_empty(self):
        """Loading chapters for a non-existent file should return empty list."""
        result = _load_chapters("never_saved.pdf")
        assert result == []

    def test_save_overwrites_existing(self):
        """Saving the same filename twice should overwrite."""
        _save_chapters("doc.pdf", [{"title": "旧章节"}])
        _save_chapters("doc.pdf", [{"title": "新章节"}, {"title": "另一个"}])

        loaded = _load_chapters("doc.pdf")
        assert len(loaded) == 2
        assert loaded[0]["title"] == "新章节"

    def test_load_all_files(self):
        """_load_chapters() without filename should return all files."""
        _save_chapters("a.pdf", [{"title": "A1"}])
        _save_chapters("b.pdf", [{"title": "B1"}, {"title": "B2"}])

        all_chapters = _load_chapters()  # No filename → all
        assert isinstance(all_chapters, dict)
        assert len(all_chapters) == 2
        assert "a.pdf" in all_chapters
        assert "b.pdf" in all_chapters
        assert len(all_chapters["a.pdf"]) == 1
        assert len(all_chapters["b.pdf"]) == 2

    def test_save_empty_chapters_list(self):
        """Saving an empty chapters list should work."""
        _save_chapters("empty.pdf", [])
        loaded = _load_chapters("empty.pdf")
        assert loaded == []

    def test_save_chapters_with_unicode(self):
        """Chinese characters should be preserved correctly."""
        chapters = [
            {"title": "第一章 静电场的基本规律", "start_page": 1},
            {"title": "第二章 静电场中的导体", "start_page": 42},
        ]
        _save_chapters("电磁学.pdf", chapters)

        loaded = _load_chapters("电磁学.pdf")
        assert loaded[0]["title"] == "第一章 静电场的基本规律"
        assert loaded[1]["start_page"] == 42


class TestGetAppContext:
    """Tests for get_app_context() FastAPI dependency."""

    def test_returns_context_when_initialized(self):
        """Should return the current AppContext when init_context was called."""
        with patch("src.api.deps.get_context") as mock_get:
            mock_ctx = MagicMock()
            mock_get.return_value = mock_ctx

            result = get_app_context()
            assert result is mock_ctx
            mock_get.assert_called_once()

    def test_raises_when_not_initialized(self):
        """Should propagate RuntimeError when context is not initialized."""
        with patch("src.api.deps.get_context", side_effect=RuntimeError("not init")):
            with pytest.raises(RuntimeError, match="not init"):
                get_app_context()


class TestGetChaptersConn:
    """Tests for _get_chapters_conn()."""

    def test_creates_table_on_first_call(self, tmp_path, monkeypatch):
        """First call should create the doc_chapters table."""
        test_db = tmp_path / "chapters.db"
        monkeypatch.setattr("src.api.deps.CHAPTERS_DB", test_db)

        import src.api.deps as deps_module
        original = deps_module.CHAPTERS_DB
        deps_module.CHAPTERS_DB = test_db
        try:
            conn = _get_chapters_conn()
            # Check table exists
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}
            assert "doc_chapters" in table_names
            conn.close()
        finally:
            deps_module.CHAPTERS_DB = original

    def test_returns_existing_connection(self, tmp_path, monkeypatch):
        """Multiple calls should work without error."""
        test_db = tmp_path / "chapters2.db"
        monkeypatch.setattr("src.api.deps.CHAPTERS_DB", test_db)

        import src.api.deps as deps_module
        original = deps_module.CHAPTERS_DB
        deps_module.CHAPTERS_DB = test_db
        try:
            conn1 = _get_chapters_conn()
            conn2 = _get_chapters_conn()
            conn1.close()
            conn2.close()
        finally:
            deps_module.CHAPTERS_DB = original
