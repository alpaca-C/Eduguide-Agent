"""QA Chat endpoint — SSE streaming answer with QASystem."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .schemas import ChatRequest
from .deps import store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """Answer a question with SSE streaming."""
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")

    session_id = req.session_id or str(uuid.uuid4())[:12]
    filter_docs = set(req.doc_filter) if req.doc_filter else set()

    try:
        store.add_chat_message(session_id, "user", req.question)
        store.save_session(session_id, req.question[:40], "[]", "")
    except Exception as e:
        logger.warning("Failed to save initial session message: %s", e)

    try:
        chat_history = store.get_chat_history(session_id) if session_id else []
    except Exception as e:
        logger.warning("Failed to load chat history for session %s: %s", session_id, e)
        chat_history = []

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        yield _sse({"type": "status", "text": "正在分析问题...", "session_id": session_id})
        await asyncio.sleep(0.1)

        reply = ""
        tool_calls_log = []
        rounds = 0

        try:
            from .deps import config as app_config
            from src.agents.qa import get_agent
            agent = get_agent(app_config)
            result = await agent.answer(
                req.question,
                doc_filter=filter_docs if filter_docs else None,
                chat_history=chat_history,
            )
            reply = result["reply"]
            rounds = result.get("rounds", 0)
            tool_calls_log = result.get("tool_calls", [])
            if tool_calls_log:
                tools_used = set(tc["tool"] for tc in tool_calls_log)
                yield _sse({
                    "type": "status",
                    "text": f"已检索 {len(tool_calls_log)} 次，正在生成回答...",
                    "session_id": session_id,
                })
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("QA failed: %s", e)
            reply = f"处理出错: {e}"

        # Stream reply character by character
        yield _sse({"type": "reply_start", "session_id": session_id})
        chunk_size = 8
        for i in range(0, len(reply), chunk_size):
            yield _sse({"type": "reply_chunk", "text": reply[i:i + chunk_size]})
            await asyncio.sleep(0.02)

        # Save assistant reply before done event
        try:
            store.add_chat_message(session_id, "assistant", reply)
        except Exception as e:
            logger.warning("Failed to save assistant reply for session %s: %s", session_id, e)

        yield _sse({
            "type": "done",
            "session_id": session_id,
            "rounds": rounds,
            "tool_calls": len(tool_calls_log),
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
