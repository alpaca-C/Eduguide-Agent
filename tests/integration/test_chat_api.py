"""
Integration tests for POST /api/chat (SSE streaming chat endpoint).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────

def _mock_qa_result():
    return {
        "reply": "库仑定律是电磁学的基本定律，描述了两个点电荷之间的相互作用力：F = k·q₁q₂/r²。",
        "route": "moderate",
        "rounds": 2,
        "tool_calls": [{"tool": "rag_search", "query": "库仑定律", "result_len": 150}],
    }


def _mock_supervisor_result():
    """Mock SupervisorOutput matching what the endpoint expects."""
    from dataclasses import dataclass
    return type("SupervisorOutput", (), {
        "reply": "库仑定律是电磁学的基本定律，描述了两个点电荷之间的相互作用力：F = k·q₁q₂/r²。",
        "rounds": 2,
        "tool_calls": 3,
        "route": "moderate",
        "session_id": "test-session",
    })()


def _parse_sse_events(streaming_response) -> list[dict]:
    events = []
    for line in streaming_response.iter_lines():
        if line and line.startswith("data: "):
            try:
                events.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                pass
    return events


# ── Fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def chat_client(mock_deps):
    """Create TestClient with Supervisor and store mocked."""
    mock_deps.store.add_chat_message.return_value = None
    mock_deps.store.save_session.return_value = None
    mock_deps.store.get_chat_history.return_value = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    ]

    # Mock supervisor.run instead of the old get_agent path
    mock_supervisor = MagicMock()
    mock_supervisor.run = AsyncMock(return_value=_mock_supervisor_result())
    mock_supervisor._memory = MagicMock()
    mock_supervisor._memory.short_term = mock_deps.store

    with mock_deps.set(
        **{"src.api.router_chat.supervisor": mock_supervisor}
    ).patch():
        from src.api import app
        yield TestClient(app)


# ── Tests ──────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_empty_question_returns_400(self, chat_client):
        resp = chat_client.post("/api/chat", json={"question": "   "})
        assert resp.status_code == 400

    def test_successful_chat_streams_sse_events(self, chat_client):
        resp = chat_client.post("/api/chat", json={
            "question": "什么是库仑定律？",
            "session_id": "test-session-1",
        })
        assert resp.status_code == 200
        events = _parse_sse_events(resp)
        event_types = {e["type"] for e in events}
        assert "status" in event_types
        assert "reply_start" in event_types
        assert "done" in event_types

    def test_chat_response_includes_session_id(self, chat_client):
        resp = chat_client.post("/api/chat", json={
            "question": "你好", "session_id": "my-session-42",
        })
        for event in _parse_sse_events(resp):
            if "session_id" in event:
                assert event["session_id"] == "my-session-42"

    def test_chat_generates_session_id_when_missing(self, chat_client):
        resp = chat_client.post("/api/chat", json={"question": "random question"})
        events = _parse_sse_events(resp)
        status_event = next((e for e in events if e["type"] == "status"), None)
        assert status_event is not None
        assert len(status_event["session_id"]) > 0

    def test_chat_reply_chunks_assemble_to_full_reply(self, chat_client):
        resp = chat_client.post("/api/chat", json={"question": "test question"})
        events = _parse_sse_events(resp)
        chunks = [e["text"] for e in events if e["type"] == "reply_chunk"]
        assert "库仑定律" in "".join(chunks)

    def test_chat_done_event_has_metadata(self, chat_client):
        resp = chat_client.post("/api/chat", json={"question": "test question"})
        events = _parse_sse_events(resp)
        done = next((e for e in events if e["type"] == "done"), None)
        assert done is not None
        assert "rounds" in done
        assert "tool_calls" in done

    def test_chat_passes_doc_filter(self, chat_client):
        """doc_filter should be passed through to SkillInput.params."""
        mock_sup = MagicMock()
        mock_sup.run = AsyncMock(return_value=_mock_supervisor_result())
        mock_sup._memory = MagicMock()
        mock_sup._memory.short_term = MagicMock()

        from unittest.mock import patch
        with patch("src.api.router_chat.supervisor", mock_sup):
            resp = chat_client.post("/api/chat", json={
                "question": "什么是梯度下降",
                "doc_filter": ["ml_book.pdf", "dl_book.pdf"],
            })
            list(resp.iter_lines())

        # Verify SkillInput.params["doc_filter"] was passed to supervisor
        call_args = mock_sup.run.call_args
        skill_input = call_args[0][0]  # first positional arg
        assert skill_input.params["doc_filter"] == {"ml_book.pdf", "dl_book.pdf"}

    def test_chat_handles_qa_failure_gracefully(self, mock_deps):
        """When Supervisor raises, the endpoint should still stream an error."""
        mock_sup = MagicMock()
        mock_sup.run = AsyncMock(side_effect=Exception("QA system down"))
        mock_sup._memory = MagicMock()
        mock_sup._memory.short_term = MagicMock()

        with mock_deps.set(
            **{"src.api.router_chat.supervisor": mock_sup}
        ).patch():
            from src.api import app
            client = TestClient(app)
            resp = client.post("/api/chat", json={"question": "anything"})
            events = _parse_sse_events(resp)
            reply_chunks = [e for e in events if e["type"] == "reply_chunk"]
            assembled = "".join(e["text"] for e in reply_chunks)
            assert "出错" in assembled or "QA system down" in assembled
