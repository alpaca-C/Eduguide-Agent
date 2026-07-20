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
from ...context_builder import ContextRouter, RouterContext, PlannerContext, SolverContext, ReflectorContext

from .router import QuestionRouter
from .solver import DirectSolver
from .planner import Planner
from .executor import Executor
from .reflector import Reflector

logger = logging.getLogger(__name__)

MAX_COMPLEX_ROUNDS = 3
MAX_MODERATE_ROUNDS = 2
MAX_INLINE_FIXES = 2      # Max inline knowledge/reasoning fixes per complex round
MAX_HISTORY_CHARS = 4000  # Max chars of chat_history to inject into prompts
COMPRESS_THRESHOLD = 12   # Messages before compressing older ones into summary
MAX_OBS_CHARS = 6000      # Max chars of accumulated observations


class QASystem:
    """Orchestrates the 5-agent QA pipeline with 3-tier difficulty routing."""

    def __init__(self, config: Configuration, gssc_pipeline=None, rag_skill=None):
        self._config = config
        self._router = QuestionRouter(config)
        self._solver = DirectSolver(config)
        self._planner = Planner(config)
        self._executor = Executor(config)
        self._reflector = Reflector(config)
        self._gssc = gssc_pipeline  # GSSC context builder (optional)
        self._ctx_router = ContextRouter()  # typed context builder
        self._rag_skill = rag_skill  # RAG retrieval skill (optional)

    # ── Public API ─────────────────────────────────────────────────

    async def answer(
        self, question: str,
        doc_filter: set[str] | None = None,
        chat_history: list[dict] | None = None,
        memory_context: object = None,  # MemoryContext from MemoryManager.recall()
        tutor_mode: bool = False,
    ) -> dict:
        """Main entry point. Returns {reply, rounds, tool_calls, route}."""
        # ── "举一反三" exercise tutor mode ──
        if tutor_mode:
            return await self._handle_tutor(question, doc_filter, chat_history)

        # Build RouterContext via ContextRouter
        router_ctx = self._ctx_router.build_router(question, memory_context)

        # Build legacy history_ctx for fallback/compat
        if memory_context is not None and hasattr(memory_context, 'history_context'):
            history_ctx = memory_context.history_context
        else:
            history_ctx = self._build_history_context(chat_history)

        # Tier 0: Classify difficulty (with typed context)
        route_result = await self._router.run(self._mk_input(
            question=question,
            chat_history=history_ctx,         # legacy compat
            router_context=router_ctx,         # typed context
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
                                               seed_decomposition=seed_decomposition,
                                               history_ctx=history_ctx)

        # Complex: Planner pipeline
        return await self._handle_complex(question, doc_filter, chat_history,
                                          seed_decomposition=seed_decomposition,
                                          history_ctx=history_ctx,
                                          memory_context=memory_context)

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
        history_ctx: str = "",
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

        return await self._handle_complex(question, doc_filter, chat_history,
                                          seed_decomposition)

    # ── Complex path ──────────────────────────────────────────────

    async def _handle_complex(
        self, question: str, doc_filter: set[str] | None,
        chat_history: list[dict] | None,
        seed_decomposition: list[str] | None = None,
        history_ctx: str = "",
        memory_context=None,
    ) -> dict:
        """Planner → Executor → Reflector loop with conversation context.

        Args:
            seed_decomposition: Optional initial sub-questions from Router.
            history_ctx: Pre-computed context (legacy path).
            memory_context: MemoryContext from MemoryManager.recall().
        """
        if not history_ctx:
            history_ctx = self._build_history_context(chat_history)
        tool_call_log: list[dict] = []
        feedback = ""
        last_answer = ""
        all_obs_text = ""  # Accumulated but truncated

        for round_num in range(1, MAX_COMPLEX_ROUNDS + 1):
            logger.info("[QA] complex round %d/%d", round_num, MAX_COMPLEX_ROUNDS)

            # ── PLAN (with typed PlannerContext) ─────
            planner_ctx = self._ctx_router.build_planner(
                question, memory_context,
                seed_decomposition=seed_decomposition,
                feedback=feedback,
                current_round=round_num,
            )
            plan = await self._planner.plan(
                question, feedback=feedback,
                history_ctx=history_ctx,
                seed_decomposition=seed_decomposition,
                planner_ctx=planner_ctx,
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

            # ── Build solver context for this round ─────────────
            solver_ctx = self._ctx_router.build_solver(
                question, plan=plan, search_results=exec_results,
            )

            # ── SOLVE ────────────────────────────────────────────
            answer = await self._planner.solve(
                question, obs_text, history_ctx=history_ctx,
                solver_ctx=solver_ctx,
            )
            last_answer = answer

            # ── REFLECT (with inline fix loop) ───────────────────
            if round_num < MAX_COMPLEX_ROUNDS:
                for fix_round in range(MAX_INLINE_FIXES + 1):
                    # Build fresh reflector context with current answer
                    reflector_ctx = self._ctx_router.build_reflector(
                        question, answer=answer, observations=obs_text,
                    )
                    verdict = await self._reflector.review(
                        question, answer, obs_text,
                        history_ctx=history_ctx,
                        reflector_ctx=reflector_ctx,
                    )
                    if verdict.get("verdict") == "SUFFICIENT":
                        logger.info("[QA] reflector: SUFFICIENT at round %d (fix=%d)",
                                     round_num, fix_round)
                        return {"reply": answer, "rounds": round_num,
                                "tool_calls": tool_call_log, "route": "complex"}

                    ins_type = verdict.get("insufficiency_type", "plan")
                    logger.info("[QA] reflector INSUFFICIENT: type=%s round=%d fix=%d/%d",
                                 ins_type, round_num, fix_round, MAX_INLINE_FIXES)

                    # ── Plan type or fix exhausted → full re-plan next round ──
                    if ins_type == "plan" or fix_round >= MAX_INLINE_FIXES:
                        if self._rag_skill is not None:
                            self._rag_skill.mark_unsatisfied(question)
                        feedback = self._build_feedback(verdict)
                        logger.info("[QA] feedback → next planner round: %s", feedback[:200])
                        break  # exit inner loop → next main round

                    # ── Knowledge type → re-execute with better queries ──
                    if ins_type == "knowledge":
                        queries = verdict.get("suggested_queries", [])
                        if not queries:
                            # No queries to search → fallback to plan
                            logger.info("[QA] knowledge type but no queries → fallback to plan")
                            if self._rag_skill is not None:
                                self._rag_skill.mark_unsatisfied(question)
                            feedback = self._build_feedback(verdict)
                            break

                        # Escalate search tier for these queries
                        if self._rag_skill is not None:
                            for q in queries:
                                self._rag_skill.mark_unsatisfied(q)

                        # Build mini-plan from suggested queries and execute
                        mini_plan = self._queries_to_plan(queries)
                        new_results = await self._executor.execute(mini_plan, doc_filter)
                        for sub_result in new_results:
                            for r in sub_result.get("results", []):
                                if hasattr(r, "tool_name"):
                                    tool_call_log.append({
                                        "sub_id": sub_result.get("id"),
                                        "tool": r.tool_name, "query": r.query,
                                    })

                        # Accumulate new observations
                        new_parts = []
                        for sub in new_results:
                            new_parts.append(
                                f"\n### 补搜 {sub.get('id')}: {sub.get('question')}"
                            )
                            for r in sub.get("results", []):
                                new_parts.append(f"[{r.tool_name}] {r.content[:600]}")
                        new_obs = "\n".join(new_parts)
                        all_obs_text = (
                            all_obs_text[-MAX_OBS_CHARS // 2:]
                            + "\n---\n" + new_obs
                        )
                        if len(all_obs_text) > MAX_OBS_CHARS:
                            all_obs_text = (
                                "...（早期搜索结果已截断）\n"
                                + all_obs_text[-MAX_OBS_CHARS:]
                            )

                        # Re-solve with accumulated observations
                        solver_ctx = self._ctx_router.build_solver(
                            question, plan=plan, search_results=exec_results,
                        )
                        answer = await self._planner.solve(
                            question, all_obs_text, history_ctx=history_ctx,
                            solver_ctx=solver_ctx,
                        )
                        last_answer = answer
                        continue  # re-reflect in inner loop

                    # ── Reasoning type → re-solve only ────────────
                    if ins_type == "reasoning":
                        issues = verdict.get("issues", [])
                        fb = (
                            "；".join(issues)
                            if issues
                            else "请修正综合推理逻辑，更充分地利用搜索资料中的信息。"
                        )
                        solver_ctx = self._ctx_router.build_solver(
                            question, plan=plan, search_results=exec_results,
                        )
                        answer = await self._planner.solve(
                            question, all_obs_text, history_ctx=history_ctx,
                            solver_ctx=solver_ctx,
                            reasoning_feedback=fb,
                        )
                        last_answer = answer
                        continue  # re-reflect in inner loop

        return {"reply": last_answer, "rounds": MAX_COMPLEX_ROUNDS,
                "tool_calls": tool_call_log, "route": "complex"}

    # ── Exercise Tutor path ────────────────────────────────────────

    async def _handle_tutor(
        self, question: str, doc_filter: set[str] | None,
        chat_history: list[dict] | None,
    ) -> dict:
        """Socratic exercise tutor: search → guided response (no direct answer).

        Loads the "exercise_tutor" skill prompt, runs rag_search to retrieve
        relevant textbook knowledge, then generates a Socratic-style guided
        response that leads the student to discover the answer themselves.
        """
        import src.skills.exercise_tutor  # noqa: F401 — trigger register_skill()
        from src.skills import get_skill
        skill = get_skill("exercise_tutor")
        if skill is None:
            return {"reply": "举一反三功能未初始化，请检查 skills 配置。",
                    "rounds": 0, "tool_calls": [], "route": "error"}

        # Phase 1: Retrieve relevant knowledge from textbooks
        observations_text = ""
        tool_call_log: list[dict] = []

        if "rag_search" in skill.tools:
            try:
                result = await self._solver.run(self._mk_input(
                    question=question, doc_filter=doc_filter,
                    chat_history=chat_history,
                ))
                if result.success:
                    meta = result.metadata
                    for obs in meta.get("observations", []):
                        if hasattr(obs, "content"):
                            observations_text += obs.content + "\n\n"
                    tool_call_log = meta.get("tool_calls", [])
            except Exception as e:
                logger.warning("Exercise tutor search failed: %s", e)
                observations_text = "（暂时无法检索教材内容，请直接引导）"

        if not observations_text.strip():
            observations_text = "（未找到相关教材内容，请基于通用知识引导）"

        # Phase 2: Generate Socratic guided response
        history_ctx = self._build_history_context(chat_history)
        prompt = skill.system_prompt.format(
            question=question,
            observations=observations_text,
            chat_history=history_ctx or "（新对话）",
        )

        try:
            llm = self._solver._make_llm(temperature=0.3)   # Slight warmth for natural tutoring
            from langchain_core.messages import HumanMessage, SystemMessage
            resp = await llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content="请开始引导学生。"),
            ])
            reply = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.error("Exercise tutor LLM failed: %s", e)
            reply = f"抱歉，生成引导式回答时出错: {e}"

        return {"reply": reply, "rounds": 1, "tool_calls": tool_call_log, "route": "tutor"}

    @staticmethod
    def _mk_input(**metadata) -> object:
        from ..base import AgentInput
        return AgentInput(metadata=metadata)

    # ── Inline fix helpers ────────────────────────────────────────

    @staticmethod
    def _queries_to_plan(queries: list[str]) -> list[dict]:
        """Convert suggested search queries into a mini-plan for Executor.

        Each query becomes a standalone sub-question with no dependencies,
        using rag_search as the tool.
        """
        if not queries:
            return []
        return [
            {
                "id": 900 + i,
                "question": q,
                "keywords": [q],
                "tool": "rag_search",
                "depends_on": [],
            }
            for i, q in enumerate(queries[:5])  # cap at 5 queries
        ]

    @staticmethod
    def _build_feedback(verdict: dict) -> str:
        """Build structured feedback string from Reflector verdict."""
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
        return "。".join(fb_parts)


# ===========================================================================
# Singleton & backward-compat
# ===========================================================================

_agent: QASystem | None = None


def get_agent(config: Configuration, gssc_pipeline=None, rag_skill=None) -> QASystem:
    """Get or create the QASystem singleton."""
    global _agent
    if _agent is None:
        _agent = QASystem(config, gssc_pipeline=gssc_pipeline, rag_skill=rag_skill)
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
