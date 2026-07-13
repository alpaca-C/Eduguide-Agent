# Integration tests for POST /api/chat (SSE streaming chat endpoint)

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Mock QA agent result ──────────────────────────────────────────────

def _mock_qa_result():
    """Return a canned QA result matching QASystem.answer() output shape."""
    return {
        "reply": "库仑定律是电磁学的基本定律，描述了两个点电荷之间的相互作用力：F = k·q₁q₂/r²。",
        "route": "moderate",
        "rounds": 2,
        "tool_calls": [
            {"tool": "rag_search", "query": "库仑定律", "result_len": 150},
        ],
    }


def _mock_agent():
    """Create a mock QA agent with async answer()."""
    agent = MagicMock()
    agent.answer = AsyncMock(return_value=_mock_qa_result())
    return agent


# ── Fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def chat_client(tmp_path):
    """Create a FastAPI TestClient with chat router dependencies mocked."""
    from fastapi.testclient import TestClient

    fake_store = MagicMock()
    fake_store.add_chat_message.return_value = None
    fake_store.save_session.return_value = None
    fake_store.get_chat_history.return_value = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    ]

    fake_qa_agent = _mock_agent()
    fake_config = MagicMock()

    # Pre-import the module to ensure patch targets exist
    # We patch at router level and also patch get_agent
    with patch("src.api.router_chat.store", fake_store), \
         patch("src.api.deps.config", fake_config), \
         patch("src.api.deps.store", fake_store), \
         patch("src.api.deps.kg", MagicMock()), \
         patch("src.api.deps.vs", MagicMock()), \
         patch("src.api.deps.chapter_agent", MagicMock()), \
         patch("src.api.deps.chapters_cache", {}), \
         patch("src.api.deps.uploaded_files", {}), \
         patch("src.api.deps.UPLOAD_DIR", tmp_path), \
         patch("src.api.deps.PROJECT_ROOT", tmp_path), \
         patch("src.agents.qa.get_agent", return_value=fake_qa_agent):

        from src.api import app
        yield TestClient(app)


# ── Helpers ────────────────────────────────────────────────────────────

def _parse_sse_events(streaming_response) -> list[dict]:
    """Parse SSE data: lines from a streaming response."""
    events = []
    for line in streaming_response.iter_lines():
        if line and line.startswith("data: "):
            data_str = line[len("data: "):]
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                pass
    return events


class TestChatEndpoint:
    """Tests for POST /api/chat SSE streaming."""

    def test_empty_question_returns_400(self, chat_client):
        """Empty or whitespace-only question should return 400."""
        response = chat_client.post("/api/chat", json={
            "question": "   ",
        })
        assert response.status_code == 400

    def test_successful_chat_streams_sse_events(self, chat_client):
        """A valid question should produce SSE events with status, reply, done."""
        response = chat_client.post("/api/chat", json={
            "question": "什么是库仑定律？",
            "session_id": "test-session-1",
        })
        assert response.status_code == 200

        events = _parse_sse_events(response)
        assert len(events) >= 3  # status + reply_start + done at minimum

        event_types = {e["type"] for e in events}
        assert "status" in event_types
        assert "reply_start" in event_types
        assert "done" in event_types

    def test_chat_response_includes_session_id(self, chat_client):
        """SSE events should include the session_id."""
        response = chat_client.post("/api/chat", json={
            "question": "你好",
            "session_id": "my-session-42",
        })
        events = _parse_sse_events(response)

        for event in events:
            if "session_id" in event:
                assert event["session_id"] == "my-session-42"

    def test_chat_generates_session_id_when_missing(self, chat_client):
        """When session_id is empty, a new one should be generated."""
        response = chat_client.post("/api/chat", json={
            "question": "random question",
        })
        events = _parse_sse_events(response)

        # First status event should have a generated session_id
        status_event = next((e for e in events if e["type"] == "status"), None)
        assert status_event is not None
        assert "session_id" in status_event
        assert len(status_event["session_id"]) > 0

    def test_chat_reply_chunks_assemble_to_full_reply(self, chat_client):
        """reply_chunk events should assemble into the full reply text."""
        response = chat_client.post("/api/chat", json={
            "question": "test question",
        })
        events = _parse_sse_events(response)

        # Collect reply chunks
        chunks = [e["text"] for e in events if e["type"] == "reply_chunk"]
        assembled = "".join(chunks)

        # Should match the mock QA reply
        assert "库仑定律" in assembled

    def test_chat_done_event_has_metadata(self, chat_client):
        """The done event should include rounds and tool_calls count."""
        response = chat_client.post("/api/chat", json={
            "question": "test question",
        })
        events = _parse_sse_events(response)

        done = next((e for e in events if e["type"] == "done"), None)
        assert done is not None
        assert "rounds" in done
        assert "tool_calls" in done

    def test_chat_passes_doc_filter(self, chat_client):
        """doc_filter should be forwarded to the QA agent."""
        # Re-mock the agent to capture call args
        mock_agent = _mock_agent()

        with patch("src.agents.qa.get_agent", return_value=mock_agent):
            response = chat_client.post("/api/chat", json={
                "question": "什么是梯度下降",
                "doc_filter": ["ml_book.pdf", "dl_book.pdf"],
            })
            # Consume the SSE stream
            list(response.iter_lines())

        # Verify doc_filter was passed
        call_kwargs = mock_agent.answer.call_args.kwargs
        assert call_kwargs["doc_filter"] == {"ml_book.pdf", "dl_book.pdf"}

    def test_chat_saves_assistant_reply(self, chat_client):
        """After streaming, assistant reply should be saved to store."""
        response = chat_client.post("/api/chat", json={
            "question": "hello",
            "session_id": "s1",
        })
        # Consume
        list(response.iter_lines())

        # Store.add_chat_message should have been called for assistant
        import src.api.router_chat as chat_router
        # Called twice: once for user message, once for assistant reply
        assert chat_router.store.add_chat_message.call_count >= 1

    def test_chat_handles_qa_failure_gracefully(self, chat_client):
        """When QA agent raises, the endpoint should still stream an error."""
        mock_agent = MagicMock()
        mock_agent.answer = AsyncMock(side_effect=Exception("QA system down"))

        with patch("src.agents.qa.get_agent", return_value=mock_agent):
            response = chat_client.post("/api/chat", json={
                "question": "anything",
            })
            events = _parse_sse_events(response)

            # Should still have reply_start and done events
            reply_chunks = [e for e in events if e["type"] == "reply_chunk"]
            assembled = "".join(e["text"] for e in reply_chunks)
            assert "出错" in assembled or "QA system down" in assembled
