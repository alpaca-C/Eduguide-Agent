"""QA Chat endpoint — SSE streaming answer via Supervisor."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .schemas import ChatRequest
from .deps import supervisor, store  # store kept for backward compat (tests access router_chat.store)
from src.skills.skill_base import SkillInput

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """Answer a question with SSE streaming."""
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")

    session_id = req.session_id or str(uuid.uuid4())[:12]
    filter_docs = set(req.doc_filter) if req.doc_filter else set()

    # Save user message
    try:
        supervisor._memory.short_term.add_message(session_id, "user", req.question)
        supervisor._memory.short_term.save_session(session_id, req.question[:40], "[]", "")
    except Exception as e:
        logger.warning("Failed to save initial session message: %s", e)

    # Build SkillInput — API layer injects QA-specific params
    skill_input = SkillInput(
        question=req.question,
        params={
            "doc_filter": filter_docs if filter_docs else None,
            "tutor_mode": req.tutor_mode,
        },
    )

    def _sse(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_stream():
        from src.harness import set_request_id, _agent_name
        set_request_id()
        _agent_name.set("Supervisor")

        yield _sse({"type": "status", "text": "正在分析问题...", "session_id": session_id})
        await asyncio.sleep(0.1)

        reply = ""
        rounds = 0
        tool_calls_count = 0

        try:
            result = await supervisor.run(skill_input, session_id=session_id)
            reply = result.reply
            rounds = result.rounds
            tool_calls_count = result.tool_calls
            if tool_calls_count:
                yield _sse({
                    "type": "status",
                    "text": f"已检索 {tool_calls_count} 次，正在生成回答...",
                    "session_id": session_id,
                })
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("Supervisor failed: %s", e)
            reply = f"处理出错: {e}"

        # Stream reply
        yield _sse({"type": "reply_start", "session_id": session_id})
        chunk_size = 8
        for i in range(0, len(reply), chunk_size):
            yield _sse({"type": "reply_chunk", "text": reply[i:i + chunk_size]})
            await asyncio.sleep(0.02)

        yield _sse({
            "type": "done",
            "session_id": session_id,
            "rounds": rounds,
            "tool_calls": tool_calls_count,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
