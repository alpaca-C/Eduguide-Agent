# QA Orchestrator — 3-tier routing with conversation memory
#
# Tier 0: trivial  → direct answer (1 LLM)
# Tier 1: moderate → DirectSolver (2-3 LLM)
# Tier 2: complex  → Planner→Executor→Reflector loop (≤3 rounds)
#
# Conversation memory:
#   chat_history is loaded from SQLite per session and passed to all sub-agents.
#   When history exceeds COMPRESS_THRESHOLD messages, older turns are compressed
#   into a summary to stay within the model's context window.

from __future__ import annotations

import logging

from ...config import Configuration
from ...tools import ToolResult

from .router import QuestionRouter
from .solver import DirectSolver
from .planner import Planner
from .executor import Executor
from .reflector import Reflector

logger = logging.getLogger(__name__)

MAX_COMPLEX_ROUNDS = 3
MAX_MODERATE_ROUNDS = 2
MAX_HISTORY_CHARS = 4000  # Max chars of chat_history to inject into prompts
COMPRESS_THRESHOLD = 12   # Messages before compressing older ones into summary
MAX_OBS_CHARS = 6000      # Max chars of accumulated observations


class QASystem:
    """Orchestrates the 5-agent QA pipeline with 3-tier difficulty routing."""

    def __init__(self, config: Configuration):
        self._config = config
        self._router = QuestionRouter(config)
        self._solver = DirectSolver(config)
        self._planner = Planner(config)
        self._executor = Executor(config)
        self._reflector = Reflector(config)
        # Separate LLM for history compression (cheap, fast)
        self._compress_llm = None

    # ── Public API ─────────────────────────────────────────────────

    async def answer(
        self, question: str,
        doc_filter: set[str] | None = None,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """Main entry point. Returns {reply, rounds, tool_calls, route}."""
        # Compress history for injection into prompts
        history_ctx = self._build_history_context(chat_history)

        # Tier 0: Classify difficulty (with conversation context)
        route_result = await self._router.run(self._mk_input(
            question=question, chat_history=history_ctx,
        ))
        difficulty = route_result.metadata.get("difficulty", "moderate")

        # Trivial: direct answer
        if difficulty == "trivial":
            reply = await self._router.direct_answer(question)
            return {"reply": reply, "rounds": 0, "tool_calls": [], "route": "trivial"}

        # Reuse Router's decomposition as seed for Planner (both paths)
        seed_decomposition = route_result.metadata.get("decomposition", [])

        # Moderate: DirectSolver
        if difficulty == "moderate":
            return await self._handle_moderate(question, doc_filter, chat_history,
                                               seed_decomposition=seed_decomposition)

        # Complex: Planner pipeline
        return await self._handle_complex(question, doc_filter, chat_history,
                                          seed_decomposition=seed_decomposition)

    # ── History context builder ────────────────────────────────────

    def _build_history_context(self, chat_history: list[dict] | None) -> str:
        """Convert chat_history to a compact context string.

        When history exceeds COMPRESS_THRESHOLD messages, older turns
        are summarized. Total output capped at MAX_HISTORY_CHARS.
        """
        if not chat_history:
            return ""

        total = len(chat_history)
        if total <= 6:
            # Short history: include all messages directly
            parts = []
            for msg in chat_history[-6:]:
                role = "学生" if msg.get("role") == "user" else "答疑老师"
                parts.append(f"- {role}: {msg.get('content', '')[:300]}")
            ctx = "【对话历史】\n" + "\n".join(parts)

        elif total <= COMPRESS_THRESHOLD:
            # Medium history: include last 10, truncate older
            parts = []
            for msg in chat_history[-10:]:
                role = "学生" if msg.get("role") == "user" else "答疑老师"
                parts.append(f"- {role}: {msg.get('content', '')[:200]}")
            ctx = f"【对话历史（最近 10 轮，共 {total} 轮）】\n" + "\n".join(parts)

        else:
            # Long history: compress older messages into summary
            recent = chat_history[-8:]
            older = chat_history[:-8]
            summary = self._summarize_sync(older)
            parts = [f"【历史摘要】{summary}"]
            for msg in recent:
                role = "学生" if msg.get("role") == "user" else "答疑老师"
                parts.append(f"- {role}: {msg.get('content', '')[:200]}")
            ctx = f"【对话历史（共 {total} 轮，早期已摘要）】\n" + "\n".join(parts)

        if len(ctx) > MAX_HISTORY_CHARS:
            ctx = ctx[:MAX_HISTORY_CHARS] + "\n...（历史已截断）"
        return ctx

    def _summarize_sync(self, messages: list[dict]) -> str:
        """Simple rule-based summary (no LLM call needed for basic compression).

        Extracts topics from user questions to give the model context about
        what was discussed earlier in the conversation.
        """
        questions = [m.get("content", "")[:100] for m in messages if m.get("role") == "user"]
        if not questions:
            return "（之前的对话）"
        topics = "；".join(q[:60] for q in questions[-5:])
        return f"之前讨论过：{topics}"

    # ── Moderate path ─────────────────────────────────────────────

    async def _handle_moderate(
        self, question: str, doc_filter: set[str] | None,
        chat_history: list[dict] | None,
        seed_decomposition: list[str] | None = None,
    ) -> dict:
        """DirectSolver with up to MAX_MODERATE_ROUNDS retry on escalation."""
        tool_call_log: list[dict] = []

        for round_num in range(1, MAX_MODERATE_ROUNDS + 1):
            logger.info("[QA] moderate round %d/%d", round_num, MAX_MODERATE_ROUNDS)
            result = await self._solver.run(self._mk_input(
                question=question, doc_filter=doc_filter, chat_history=chat_history,
            ))
            if not result.success:
                return {"reply": f"处理出错: {result.error}", "rounds": round_num, "tool_calls": tool_call_log, "route": "error"}

            meta = result.metadata
            tool_call_log.extend(meta.get("tool_calls", []))

            if meta.get("route") == "done":
                return {"reply": meta["reply"], "rounds": round_num, "tool_calls": tool_call_log, "route": "moderate"}
            logger.info("[QA] moderate escalated to complex at round %d", round_num)
            break

        return await self._handle_complex(question, doc_filter, chat_history, seed_decomposition)

    # ── Complex path ──────────────────────────────────────────────

    async def _handle_complex(
        self, question: str, doc_filter: set[str] | None,
        chat_history: list[dict] | None,
        seed_decomposition: list[str] | None = None,
    ) -> dict:
        """Planner → Executor → Reflector loop with conversation context.

        Args:
            seed_decomposition: Optional initial sub-questions from Router.
                Passed to Planner to seed decomposition (avoids re-deriving from scratch).
        """
        history_ctx = self._build_history_context(chat_history)
        tool_call_log: list[dict] = []
        feedback = ""
        last_answer = ""
        all_obs_text = ""  # Accumulated but truncated

        for round_num in range(1, MAX_COMPLEX_ROUNDS + 1):
            logger.info("[QA] complex round %d/%d", round_num, MAX_COMPLEX_ROUNDS)

            # ── PLAN (with history + feedback + router seed) ─────
            plan = await self._planner.plan(
                question, feedback=feedback,
                history_ctx=history_ctx,
                seed_decomposition=seed_decomposition,
            )
            if not plan:
                logger.warning("[QA] planner returned empty plan at round %d", round_num)
                if last_answer:
                    return {"reply": last_answer, "rounds": round_num, "tool_calls": tool_call_log, "route": "complex"}
                fallback = await self._handle_moderate(question, doc_filter, chat_history,
                                                         seed_decomposition=seed_decomposition)
                fallback["route"] = "complex_fallback"
                return fallback

            logger.info("[QA] planner: %d sub-questions", len(plan))

            # ── EXECUTE ──────────────────────────────────────────
            exec_results = await self._executor.execute(plan, doc_filter)
            for sub_result in exec_results:
                for r in sub_result.get("results", []):
                    if hasattr(r, "tool_name"):
                        tool_call_log.append({
                            "sub_id": sub_result.get("id"),
                            "tool": r.tool_name, "query": r.query,
                        })

            # Build observation text (truncated to avoid context overflow)
            obs_parts = []
            for sub in exec_results:
                obs_parts.append(f"\n### 子问题 {sub.get('id')}: {sub.get('question')}")
                for r in sub.get("results", []):
                    obs_parts.append(f"[{r.tool_name}] {r.content[:600]}")
            obs_text = "\n".join(obs_parts)

            # Accumulate with truncation
            if all_obs_text:
                all_obs_text = all_obs_text[-MAX_OBS_CHARS // 2:] + "\n---\n" + obs_text
            else:
                all_obs_text = obs_text
            if len(all_obs_text) > MAX_OBS_CHARS:
                all_obs_text = "...（早期搜索结果已截断）\n" + all_obs_text[-MAX_OBS_CHARS:]

            # ── SOLVE (with history context) ─────────────────────
            answer = await self._planner.solve(question, obs_text, history_ctx=history_ctx)
            last_answer = answer

            # ── REFLECT ──────────────────────────────────────────
            if round_num < MAX_COMPLEX_ROUNDS:
                verdict = await self._reflector.review(
                    question, answer, obs_text,
                    history_ctx=history_ctx,
                )
                if verdict.get("verdict") == "SUFFICIENT":
                    logger.info("[QA] reflector: SUFFICIENT at round %d", round_num)
                    return {"reply": answer, "rounds": round_num, "tool_calls": tool_call_log, "route": "complex"}

                # Build structured feedback for next planner iteration
                missing = verdict.get("missing", [])
                queries = verdict.get("suggested_queries", [])
                issues = verdict.get("issues", [])
                fb_parts = []
                if missing:
                    fb_parts.append(f"缺失知识点：{'；'.join(missing)}")
                if queries:
                    fb_parts.append(f"建议搜索查询：{'；'.join(queries)}")
                if issues:
                    fb_parts.append(f"回答问题：{'；'.join(issues)}")
                feedback = "。".join(fb_parts)
                logger.info("[QA] reflector INSUFFICIENT: %s", feedback[:200])

        return {"reply": last_answer, "rounds": MAX_COMPLEX_ROUNDS, "tool_calls": tool_call_log, "route": "complex"}

    @staticmethod
    def _mk_input(**metadata) -> object:
        from ..base import AgentInput
        return AgentInput(metadata=metadata)


# ===========================================================================
# Singleton & backward-compat
# ===========================================================================

_agent: QASystem | None = None


def get_agent(config: Configuration) -> QASystem:
    """Get or create the QASystem singleton."""
    global _agent
    if _agent is None:
        _agent = QASystem(config)
    return _agent


async def answer_question(
    question: str,
    relevant_chunks: list[str] | None = None,
    graph_context: str = "",
    config: Configuration | None = None,
    chat_history: list[dict] | None = None,
    doc_filter: set[str] | None = None,
) -> str:
    if config is None:
        raise ValueError("config is required")
    agent = get_agent(config)
    result = await agent.answer(question, doc_filter=doc_filter, chat_history=chat_history)
    return result["reply"]
