"""
Root conftest — shared fixtures for all tests.

Provides:
  - mock_deps: one-stop context manager that patches all router-level deps
  - mock_config: Configuration fixture for unit tests
  - LLM-canned-response fixtures (re-exported from tests/steps)
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# ── Module-level: prevent ChatOpenAI from making real network calls ─────────
# Applied at import time so any code that creates ChatOpenAI during test
# collection/execution gets a mock instead of a real network connection.
_mock_llm = MagicMock()
_mock_llm.return_value.ainvoke = MagicMock(return_value=MagicMock(content="mock"))
_patcher = patch("langchain_openai.ChatOpenAI", _mock_llm)
try:
    _patcher.start()
except Exception:
    pass  # langchain_openai may not be installed in all test environments

from pathlib import Path

import pytest


# ═══════════════════════════════════════════════════════════════════
# Shared mock factory
# ═══════════════════════════════════════════════════════════════════

def _patch_all_deps(
    *,
    store=None,
    fake_uploaded=None,
    tmp_path=None,
    extra_patches: dict | None = None,
):
    """
    Return a merged dict of (target, value) entries that patches every
    module-level dependency router modules import from src.api.deps.

    Router modules hold their own references (``from .deps import store``),
    so we must patch at the *router* level, not just deps itself.

    NOTE: Not every router imports every dep (e.g. router_files does NOT
    import ``store``). The PatchHelper.start() silently skips attributes
    that don't exist on the target module.
    """
    if store is None:
        store = MagicMock()
    if fake_uploaded is None:
        fake_uploaded = {}
    if tmp_path is None:
        tmp_path = Path(".")

    fake_upload_dir = tmp_path / "uploads"
    fake_upload_dir.mkdir(exist_ok=True)

    # Build a mock MemoryManager that delegates to the same mock store.
    mock_mm = MagicMock()
    mock_mm.short_term.list_sessions = store.list_sessions
    mock_mm.short_term.get_session = store.get_session
    mock_mm.short_term.get_history = store.get_chat_history
    mock_mm.short_term.add_message = store.add_chat_message
    mock_mm.short_term.save_session = store.save_session
    mock_mm.short_term.delete_session = store.delete_session
    mock_mm.short_term.build_context = MagicMock(return_value="")
    # recall() returns a MemoryContext-like object (no episodic — caches are separate)
    mock_mm.recall = MagicMock(return_value=MagicMock(
        chat_history=[], history_context="", episodes=[], available_docs=[],
    ))

    # Mock ExactMatchCache (standalone cache, not part of memory)
    mock_cache = MagicMock()
    mock_cache.find_search.return_value = None
    mock_cache.find_plan.return_value = None

    patches = {
        # -- router-level overrides (each router has its own ``from .deps``) --
        # Both store and memory_manager are patched; routers use either
        "src.api.router_chat.store": store,
        "src.api.router_sessions.store": store,
        # MemoryManager patches for routers that now use it
        "src.api.router_chat.memory_manager": mock_mm,
        "src.api.router_sessions.memory_manager": mock_mm,

        # -- direct deps (fallback) --
        "src.api.deps.store": store,
        "src.api.deps.config": MagicMock(),
        "src.api.deps.kg": MagicMock(),
        "src.api.deps.vs": MagicMock(),
        "src.api.deps.chapter_agent": MagicMock(),
        "src.api.deps.chapters_cache": {},
        "src.api.deps._ctx": MagicMock(),
        "src.api.deps.memory_manager": mock_mm,
        "src.api.deps.exact_cache": mock_cache,
        "src.api.router_chat.exact_cache": mock_cache,

        # -- file paths --
        "src.api.deps.UPLOAD_DIR": fake_upload_dir,
        "src.api.deps.PROJECT_ROOT": tmp_path,
        "src.api.router_files.uploaded_files": fake_uploaded,
        "src.api.router_files.UPLOAD_DIR": fake_upload_dir,
        "src.api.deps.uploaded_files": fake_uploaded,
    }

    if extra_patches:
        patches.update(extra_patches)

    return patches


# ═══════════════════════════════════════════════════════════════════
# Integration-test fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_deps(tmp_path):
    """
    Return a helper object that can be used with ``with mock_deps.patch():``
    to activate all shared mock patches in one block.

    Usage in a test::

        with mock_deps.patch():
            from src.api import app
            client = TestClient(app)
            resp = client.get("/api/health")
    """
    class DepsPatcher:
        def __init__(self, base_path):
            self._base_path = base_path
            self._extra: dict = {}
            self.store = MagicMock()

        def set(self, **extra):
            """Register extra patches before entering the context."""
            self._extra.update(extra)
            return self

        @contextmanager
        def patch(self, *, store=None, **extra):
            from unittest.mock import patch as _patch

            s = store if store is not None else self.store
            targets = _patch_all_deps(
                store=s,
                tmp_path=self._base_path,
                extra_patches={**self._extra, **extra},
            )

            # Start patches, silently skip targets that don't have the attr
            # (e.g. router_files does not import ``store``)
            started: list = []
            for target_str, value in targets.items():
                try:
                    patcher = _patch(target_str, value)
                    patcher.start()
                    started.append(patcher)
                except AttributeError:
                    pass  # target module doesn't have this attribute

            try:
                yield
            finally:
                for p in reversed(started):
                    p.stop()

    return DepsPatcher(tmp_path)


# ═══════════════════════════════════════════════════════════════════
# Unit-test fixtures (re-exported for discoverability)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_config(monkeypatch):
    """Return Configuration with test-safe values (no real API keys)."""
    from src.config import Configuration

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")
    # Use localhost to prevent real network calls during test collection
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "test/embedding")
    monkeypatch.setenv("MONITORING_ENABLED", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    return Configuration()
