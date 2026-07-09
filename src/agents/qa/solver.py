# DirectSolver — think → act → synthesize for moderate questions

from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...tools import get_tool_registry, ToolResult
from ...prompts.qa.router import SYSTEM_PROMPT
from ...prompts.qa.solver import SYNTHESIS_PROMPT
from .query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class DirectSolver(BaseAgent):
    """Handles moderate-difficulty questions: think → act → synthesize.

    Routes to Planner pipeline if initial search returns insufficient results.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._llm = self._make_llm()
        self._tools = get_tool_registry()
        self._rewriter = QueryRewriter(config)

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        doc_filter = input.metadata.get("doc_filter")
        chat_history = input.metadata.get("chat_history")

        try:
            result = await self._answer(question, doc_filter, chat_history)
            return AgentOutput(success=True, metadata=result)
        except Exception as e:
            return AgentOutput(success=False, error=str(e))

    async def _answer(
        self, question: str, doc_filter: set[str] | None, chat_history: list[dict] | None,
    ) -> dict:
        """Full pipeline: REWRITE → ACT → SYNTHESIZE."""
        observations: list[ToolResult] = []
        tool_call_log: list[dict] = []

        # ── REWRITE: convert question to search-optimized queries ──
        rewritten = await self._rewriter.rewrite(question)
        logger.info("DirectSolver: rewritten '%s' → %d queries: %s",
                     question[:50], len(rewritten), rewritten)

        # ── ACT: execute rag_search for each rewritten query ──────
        for kw in rewritten:
            try:
                result = await self._tools["rag_search"]["func"](
                    query=kw, filter_docs=doc_filter,
                )
                observations.append(result)
                tool_call_log.append({"tool": "rag_search", "query": kw, "result_len": len(result.content)})
            except Exception as e:
                logger.warning("DirectSolver rag_search failed for '%s': %s", kw, e)

        # ── SYNTHESIZE ──────────────────────────────────────────
        obs_text = "\n\n".join(
            f"[{o.tool_name}] {o.content}" for o in observations
        ) if observations else "（未找到相关资料）"

        try:
            synth_resp = await self._llm_retry([
                SystemMessage(content=SYSTEM_PROMPT + "\n\n" + SYNTHESIS_PROMPT.format(
                    observations=obs_text, question=question,
                )),
                HumanMessage(content="请基于以上资料回答学生问题。"),
            ])
            reply = synth_resp.content if hasattr(synth_resp, "content") else str(synth_resp)
        except Exception as e:
            logger.error("DirectSolver synthesis failed: %s", e)
            reply = f"抱歉，回答生成失败: {e}"

        # Detect if result is insufficient → escalate
        escalate = not observations and not reply.strip()

        return {
            "reply": reply,
            "tool_calls": tool_call_log,
            "observations": observations,
            "route": "escalate" if escalate else "done",
        }

    def _tool_descs(self, allowed: set[str]) -> str:
        return "\n".join(
            f"- **{name}**: {info['description']}"
            for name, info in self._tools.items() if name in allowed
        )

    def _build_context(self, question: str, chat_history: list[dict] | None) -> str:
        ctx = f"**学生问题：** {question}\n\n"
        if chat_history:
            ctx += "对话历史：\n"
            for msg in chat_history[-10:]:
                role = "学生" if msg.get("role") == "user" else "答疑老师"
                ctx += f"- {role}: {msg.get('content', '')[:300]}\n"
            ctx += "\n"
        return ctx

    def _parse_tool_calls_from_text(self, text: str, allowed: set[str]) -> list[dict]:
        tool_calls = []
        for tool_name in sorted(allowed):
            pattern = rf'{tool_name}\s*[：:]\s*["""]?\s*(.+?)\s*["""]?(?:\n|$)'
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                query = match.group(1).strip().strip(""'""')
                if query:
                    tool_calls.append({
                        "name": tool_name, "args": {"query": query},
                        "id": f"fallback_{tool_name}_{len(tool_calls)}",
                    })
        return tool_calls

    async def _execute_tools(self, tool_calls: list[dict], doc_filter: set[str] | None) -> list[ToolResult]:
        tasks = []
        for tc in tool_calls:
            name = tc.get("name", "")
            query = tc.get("args", {}).get("query", "")
            if name not in self._tools:
                continue
            if name == "rag_search":
                tasks.append(self._tools[name]["func"](query=query, filter_docs=doc_filter))
            else:
                tasks.append(self._tools[name]["func"](query=query))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, Exception):
                out.append(ToolResult(tool_name="error", query="", content=f"工具执行出错: {r}"))
            else:
                out.append(r)
        return out
