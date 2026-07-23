# DirectSolver — concurrent RAG → synthesize for moderate questions
#
# Rewriter has been moved upstream (orchestrator runs it before Router).
# DirectSolver now receives decomposition + normalized_queries and runs
# all RAG calls concurrently via asyncio.gather.

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

logger = logging.getLogger(__name__)


class DirectSolver(BaseAgent):
    """Handles moderate-difficulty questions: concurrent RAG → synthesize.

    Routes to Planner pipeline if initial search returns insufficient results.
    """

    def __init__(self, config: Configuration, gssc_pipeline=None):
        super().__init__(config)
        self._llm = self._make_llm()
        self._tools = get_tool_registry()
        self._gssc = gssc_pipeline

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        doc_filter = input.metadata.get("doc_filter")
        chat_history = input.metadata.get("chat_history")
        history_ctx = input.metadata.get("history_ctx", "")
        structured_prompt = input.metadata.get("structured_prompt")
        decomposition = input.metadata.get("decomposition", [])
        normalized_queries = input.metadata.get("normalized_queries", [])

        try:
            result = await self._answer(question, doc_filter, chat_history,
                                        history_ctx=history_ctx,
                                        structured_prompt=structured_prompt,
                                        decomposition=decomposition,
                                        normalized_queries=normalized_queries)
            return AgentOutput(success=True, metadata=result)
        except Exception as e:
            return AgentOutput(success=False, error=str(e))

    async def _answer(
        self, question: str, doc_filter: set[str] | None,
        chat_history: list[dict] | None,
        history_ctx: str = "",
        structured_prompt=None,
        decomposition: list[str] | None = None,
        normalized_queries: list[str] | None = None,
    ) -> dict:
        """Concurrent RAG → SYNTHESIZE. Rewriter has already run upstream."""
        from src.harness import _agent_name
        _agent_name.set("DirectSolver")
        observations: list[ToolResult] = []
        tool_call_log: list[dict] = []

        # ── Determine search queries ──
        # Priority: Router decomposition > normalized_queries > original question
        queries: list[str] = []
        if decomposition:
            queries = [d if isinstance(d, str) else str(d) for d in decomposition]
        if not queries and normalized_queries:
            queries = list(normalized_queries)
        if not queries:
            queries = [question]

        logger.info("DirectSolver: %d concurrent RAG queries for '%s'",
                     len(queries), question[:50])

        # ── ACT: execute all rag_search calls CONCURRENTLY ──────
        tasks = [
            self._tools["rag_search"]["func"](query=q, filter_docs=doc_filter)
            for q in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, (q, result) in enumerate(zip(queries, results)):
            if isinstance(result, Exception):
                logger.warning("DirectSolver rag_search failed for '%s': %s", q, result)
                observations.append(ToolResult(
                    tool_name="rag_search", query=q,
                    content=f"搜索出错: {result}",
                ))
            else:
                observations.append(result)
                tool_call_log.append({
                    "tool": "rag_search", "query": q,
                    "result_len": len(result.content) if hasattr(result, "content") else 0,
                })

        # ── EMPTY-RETRIEVAL GUARD ──────────────────────────────
        # Filter out empty / not-configured / error results
        valid_obs = [
            o for o in observations
            if o and not o.is_error and o.content.strip()
            and "未找到相关内容" not in o.content
            and "未初始化" not in o.content
        ]
        if not valid_obs:
            logger.info(
                "DirectSolver: all %d observations empty/invalid — refusing to synthesize",
                len(observations),
            )
            return {
                "reply": "抱歉，教材中未找到与您问题相关的内容。建议：\n1. 确认相关教材已上传并处理\n2. 尝试用更具体的术语重新提问\n3. 检查是否选对了教材章节",
                "tool_calls": tool_call_log,
                "observations": observations,
                "route": "empty_result",
            }

        # ── SYNTHESIZE ──────────────────────────────────────────
        obs_text = "\n\n".join(
            f"[{o.tool_name}] {o.content}" for o in valid_obs
        )
        # GSSC compress: enforce token budget via configured Compressor
        LIMIT = 4000  # fallback chars when GSSC unavailable
        if self._gssc is not None:
            try:
                from src.context_builder.schema import StructuredPrompt
                from src.context_builder.compress import Compressor
                hard_limit = getattr(self._config, 'context_hard_limit', 2000)
                prompt = StructuredPrompt(sections={"evidence": obs_text, "task": question})
                compressed = Compressor(hard_limit=hard_limit).compress(prompt)
                if compressed.compressed:
                    obs_text = compressed.sections.get("evidence", obs_text)
                    logger.info("DirectSolver: GSSC compressed obs → %d chars", len(obs_text))
            except Exception as e:
                logger.warning("DirectSolver: GSSC compress failed: %s", e)
        elif len(obs_text) > LIMIT:
            obs_text = obs_text[:LIMIT] + "\n...（已截断）"

        # Build context with conversation history (if available)
        history_block = self._build_context(question, chat_history,
                                            history_ctx=history_ctx)

        try:
            synth_resp = await self._llm_retry([
                SystemMessage(content=SYSTEM_PROMPT + "\n\n" + SYNTHESIS_PROMPT.format(
                    observations=obs_text, question=question,
                )),
                HumanMessage(content=history_block + "请基于以上资料回答学生问题。如果资料不足以回答，请明确告知学生而不是编造内容。"),
            ])
            reply = synth_resp.content if hasattr(synth_resp, "content") else str(synth_resp)
        except Exception as e:
            logger.error("DirectSolver synthesis failed: %s", e)
            reply = f"抱歉，回答生成失败: {e}"

        # Detect if result is insufficient → escalate
        escalate = not reply.strip()

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

    def _build_context(self, question: str, chat_history: list[dict] | None,
                       history_ctx: str = "") -> str:
        ctx = f"**学生问题：** {question}\n\n"
        # Prefer pre-computed history_ctx (from MemoryManager) over raw chat_history
        if history_ctx:
            ctx += history_ctx + "\n\n"
        elif chat_history:
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
