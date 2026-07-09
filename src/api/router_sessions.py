"""Session management — list, get, delete chat sessions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .deps import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions():
    """List all saved sessions."""
    sessions = store.list_sessions()
    return {"sessions": sessions}


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get session details and chat history."""
    data = store.get_session(session_id)
    if not data:
        raise HTTPException(404, "Session not found")
    chat_history = store.get_chat_history(session_id)
    return {
        "session_id": session_id,
        "topic": data.get("topic", ""),
        "report": data.get("report", ""),
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in chat_history
        ],
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its chat messages."""
    store.delete_session(session_id)
    return {"deleted": session_id}
