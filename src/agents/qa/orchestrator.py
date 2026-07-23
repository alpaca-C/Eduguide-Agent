# QA Orchestrator — 3-tier routing with conversation memory
#
# Rewriter runs first (normalizes question for better classification),
# then Router classifies by task structure (parallel→moderate, sequential→complex).
# Moderate uses concurrent RAG (asyncio.gather). Complex unchanged.
#
# Tier 0: trivial  → direct answer (1 LLM)
# Tier 1: moderate → DirectSolver (concurrent RAG + synthesize)
# Tier 2: complex  → Planner→Executor→Reflector loop (≤3 rounds)

from __future__ import annotations

import asyncio
import logging

from ...config import Configuration
from ...tools import ToolResult, get_tool_registry
from ...context_builder import ContextRouter  # deprecated, kept for fallback
from ...context_builder.schema import StructuredPrompt

from .router import QuestionRouter
from .solver import DirectSolver
from .planner import Planner
from .executor import Executor
from .reflector import Reflector
from .query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)

MAX_COMPLEX_ROUNDS = 3
MAX_INLINE_FIXES = 2      # Max inline knowledge/reasoning fixes per complex round
MAX_HISTORY_CHARS = 4000  # Max chars of chat_history to inject into prompts
COMPRESS_THRESHOLD = 12   # Messages before compressing older ones into summary
MAX_OBS_CHARS = 6000      # Max chars of accumulated observations


class QASystem:
    """Orchestrates the 5-agent QA pipeline with 3-tier difficulty routing."""

    def __init__(self, config: Configuration, gssc_pipeline=None, rag_skill=None,
                 memory_manager=None):
        self._config = config
        self._router = QuestionRouter(config)
        self._solver = DirectSolver(config, gssc_pipeline=gssc_pipeline)
        self._planner = Planner(config)
        self._executor = Executor(config)
        self._reflector = Reflector(config)
        self._rewriter = QueryRewriter(config)  # runs before Router for all questions
        self._gssc = gssc_pipeline        # GSSC Gather→Select→Structure→Compress
        self._ctx_router = ContextRouter()  # deprecated, fallback when GSSC unavailable
        self._rag_skill = rag_skill
        self._memory = memory_manager      # MemoryManager, for _record_episode()

    # ── Public API ─────────────────────────────────────────────────

    async def answer(
        self, question: str,
        doc_filter: set[str] | None = None,
        chat_history: list[dict] | None = None,
        memory_context: object = None,  # MemoryContext from MemoryManager.recall()
        tutor_mode: bool = False,       # ignored — routing done by Supervisor
        session_id: str = "",
        user_id: str = "",              # for episodic memory recording
    ) -> dict:
        """Main entry point. Returns {reply, rounds, tool_calls, route}.

        Flow: Rewriter (normalize) → Router (classify) → dispatch.
        """
        self._user_id = user_id  # stored for _record_episode in sub-paths
        # Build legacy history_ctx
        if memory_context is not None and hasattr(memory_context, 'history_context'):
            history_ctx = memory_context.history_context
        else:
            history_ctx = self._build_history_context(chat_history)

        # ── ① Rewriter + Router: run concurrently (no dependency) ──
        # Router classifies without rewriter keywords — the question + GSSC
        # context + history is sufficient. Rewriter keywords still feed into
        # moderate path's RAG search.
        rewriter_task = self._run_rewriter(
            question, history_ctx, memory_context, session_id,
        )
        router_task = self._route_question(
            question, history_ctx, memory_context, session_id,
        )
        normalized_queries, route_result = await asyncio.gather(
            rewriter_task, router_task,
        )
        difficulty = route_result.metadata.get("difficulty", "moderate")

        # Trivial: direct answer
        if difficulty == "trivial":
            reply = await self._router.direct_answer(question, history_ctx=history_ctx)
            self._record_episode(question, "trivial", [], 0, True, session_id=session_id)
            return {"reply": reply, "rounds": 0, "tool_calls": [], "route": "trivial"}

        # Router decomposition → moderate execution plan / complex seed
        decomposition = route_result.metadata.get("decomposition", [])

        # Moderate: concurrent RAG
        if difficulty == "moderate":
            return await self._handle_moderate(
                question, doc_filter, chat_history,
                decomposition=decomposition,
                normalized_queries=normalized_queries,
                history_ctx=history_ctx,
                memory_context=memory_context,
                session_id=session_id,
            )

        # Complex: Planner pipeline (unchanged, seed_decomposition preserved)
        return await self._handle_complex(question, doc_filter, chat_history,
                                          seed_decomposition=decomposition,
                                          history_ctx=history_ctx,
                                          memory_context=memory_context,
                                          session_id=session_id)

    # ── GSSC context builder helpers ───────────────────────────────

    async def _build_structured(
        self, question: str, memory_context=None, session_id: str = "",
        difficulty: str = "", current_round: int = 0,
        feedback: str = "", planner_output=None, search_results=None,
        current_answer: str = "",
    ) -> StructuredPrompt | None:
        """Run GSSC pipeline to build structured context. Returns None if GSSC unavailable."""
        if self._gssc is None:
            return None
        try:
            return await self._gssc.run(
                question=question,
                session_id=session_id,
                difficulty=difficulty,
                current_round=current_round,
                feedback=feedback,
                planner_output=planner_output,
                search_results=search_results,
                current_answer=current_answer,
                memory_context=memory_context,
            )
        except Exception as e:
            logger.warning("GSSC pipeline failed, falling back to legacy: %s", e)
            return None

    async def _run_rewriter(
        self, question: str, history_ctx: str, memory_context=None, session_id: str = "",
    ) -> list[str]:
        """Run QueryRewriter before Router. Returns 3-5 normalized search keywords."""
        try:
            rewriter_structured = await self._build_structured(
                question, memory_context, session_id, difficulty="", current_round=0,
            )
            keywords = await self._rewriter.rewrite(
                question, history=history_ctx,
                structured_prompt=rewriter_structured,
            )
            logger.info("Rewriter: '%s' → %d keywords: %s",
                         question[:50], len(keywords), keywords)
            return keywords
        except Exception as e:
            logger.warning("Rewriter failed, using original question: %s", e)
            return [question]

    async def _route_question(
        self, question: str, history_ctx: str, memory_context=None, session_id: str = "",
    ):
        """Classify difficulty using GSSC (preferred) or legacy ContextRouter."""
        structured = await self._build_structured(
            question, memory_context, session_id, difficulty="", current_round=0,
        )
        if structured is not None:
            return await self._router.run(self._mk_input(
                question=question,
                chat_history=history_ctx,
                structured_prompt=structured,
            ))
        # Legacy fallback
        router_ctx = self._ctx_router.build_router(question, memory_context)
        return await self._router.run(self._mk_input(
            question=question,
            chat_history=history_ctx,
            router_context=router_ctx,
        ))

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
        decomposition: list[str] | None = None,
        normalized_queries: list[str] | None = None,
        history_ctx: str = "",
        memory_context=None,
        session_id: str = "",
    ) -> dict:
        """Single-pass concurrent RAG → synthesize. Escalates on empty results."""
        tool_call_log: list[dict] = []

        result = await self._solver.run(self._mk_input(
            question=question, doc_filter=doc_filter,
            chat_history=chat_history, history_ctx=history_ctx,
            decomposition=decomposition or [],
            normalized_queries=normalized_queries or [],
        ))
        if not result.success:
            return {"reply": f"处理出错: {result.error}", "rounds": 1,
                    "tool_calls": tool_call_log, "route": "error"}

        meta = result.metadata
        tool_call_log.extend(meta.get("tool_calls", []))

        if meta.get("route") == "done":
            self._record_episode(question, "moderate", tool_call_log, 1,
                                 True, session_id=session_id)
            return {"reply": meta["reply"], "rounds": 1,
                    "tool_calls": tool_call_log, "route": "moderate"}

        # Empty or insufficient results → escalate to complex
        logger.info("[QA] moderate escalated to complex (empty results)")
        return await self._handle_complex(question, doc_filter, chat_history,
                                          seed_decomposition=decomposition or [],
                                          history_ctx=history_ctx,
                                          memory_context=memory_context,
                                          session_id=session_id)

    # ── Complex path ──────────────────────────────────────────────

    async def _handle_complex(
        self, question: str, doc_filter: set[str] | None,
        chat_history: list[dict] | None,
        seed_decomposition: list[str] | None = None,
        history_ctx: str = "",
        memory_context=None,
        session_id: str = "",
    ) -> dict:
        """Planner → Executor → Reflector loop with conversation context.

        Args:
            seed_decomposition: Optional initial sub-questions from Router.
            history_ctx: Pre-computed context (legacy path).
            memory_context: MemoryContext from MemoryManager.recall().
            session_id: Current session for GSSC Gatherer.
        """
        if not history_ctx:
            history_ctx = self._build_history_context(chat_history)
        tool_call_log: list[dict] = []
        feedback = ""
        last_answer = ""
        all_obs_text = ""  # Accumulated but truncated
        final_verdict = None  # For episode recording

        for round_num in range(1, MAX_COMPLEX_ROUNDS + 1):
            logger.info("[QA] complex round %d/%d", round_num, MAX_COMPLEX_ROUNDS)

            # ── PLAN (GSSC preferred, legacy fallback) ─────
            planner_structured = await self._build_structured(
                question, memory_context, session_id,
                difficulty="complex", current_round=round_num,
                feedback=feedback,
            )
            plan_kwargs = dict(
                question=question, feedback=feedback,
                history_ctx=history_ctx,
                seed_decomposition=seed_decomposition,
            )
            if planner_structured is not None:
                plan_kwargs["structured_prompt"] = planner_structured
            else:
                plan_kwargs["planner_ctx"] = self._ctx_router.build_planner(
                    question, memory_context,
                    seed_decomposition=seed_decomposition,
                    feedback=feedback, current_round=round_num,
                )
            plan = await self._planner.plan(**plan_kwargs)
            if not plan:
                logger.warning("[QA] planner returned empty plan at round %d", round_num)
                if last_answer:
                    return {"reply": last_answer, "rounds": round_num, "tool_calls": tool_call_log, "route": "complex"}
                fallback = await self._handle_moderate(question, doc_filter, chat_history,
                                                         seed_decomposition=seed_decomposition,
                                                         history_ctx=history_ctx,
                                                         memory_context=memory_context,
                                                         session_id=session_id)
                fallback["route"] = "complex_fallback"
                return fallback

            logger.info("[QA] planner: %d sub-questions", len(plan))

            # ── EXECUTE ──────────────────────────────────────────
            exec_results = await self._executor.execute(plan, doc_filter, history_ctx=history_ctx)
            for sub_result in exec_results:
                for r in sub_result.get("results", []):
                    if hasattr(r, "tool_name"):
                        tool_call_log.append({
                            "sub_id": sub_result.get("id"),
                            "tool": r.tool_name, "query": r.query,
                        })

            # Build observation text
            obs_parts = []
            for sub in exec_results:
                obs_parts.append(f"\n### 子问题 {sub.get('id')}: {sub.get('question')}")
                for r in sub.get("results", []):
                    obs_parts.append(f"[{r.tool_name}] {r.content[:800]}")
            obs_text = "\n".join(obs_parts)

            # Accumulate with truncation
            if all_obs_text:
                all_obs_text = all_obs_text[-MAX_OBS_CHARS // 2:] + "\n---\n" + obs_text
            else:
                all_obs_text = obs_text
            if len(all_obs_text) > MAX_OBS_CHARS:
                all_obs_text = "...（早期搜索结果已截断）\n" + all_obs_text[-MAX_OBS_CHARS:]

            # ── Build solver context via GSSC (with search results) ──
            solver_structured = await self._build_structured(
                question, memory_context, session_id,
                difficulty="complex", current_round=round_num,
                planner_output=plan, search_results=exec_results,
            )

            # ── SOLVE ────────────────────────────────────────────
            solve_kwargs = dict(
                question=question, observations=obs_text, history_ctx=history_ctx,
            )
            if solver_structured is not None:
                solve_kwargs["structured_prompt"] = solver_structured
            else:
                solve_kwargs["solver_ctx"] = self._ctx_router.build_solver(
                    question, plan=plan, search_results=exec_results, history_ctx=history_ctx,
                )
            answer = await self._planner.solve(**solve_kwargs)
            last_answer = answer

            # ── REFLECT (with inline fix loop) ───────────────────
            if round_num < MAX_COMPLEX_ROUNDS:
                for fix_round in range(MAX_INLINE_FIXES + 1):
                    # Build reflector context via GSSC (with current answer)
                    reflector_structured = await self._build_structured(
                        question, memory_context, session_id,
                        difficulty="complex", current_round=round_num,
                        search_results=exec_results, current_answer=answer,
                    )
                    review_kwargs = dict(
                        question=question, answer=answer, observations=obs_text,
                        history_ctx=history_ctx,
                    )
                    if reflector_structured is not None:
                        review_kwargs["structured_prompt"] = reflector_structured
                    else:
                        review_kwargs["reflector_ctx"] = self._ctx_router.build_reflector(
                            question, answer=answer, observations=obs_text,
                        )
                    verdict = await self._reflector.review(**review_kwargs)
                    if verdict.get("verdict") == "SUFFICIENT":
                        logger.info("[QA] reflector: SUFFICIENT at round %d (fix=%d)",
                                     round_num, fix_round)
                        self._record_episode(
                            question, "complex", tool_call_log, round_num, True,
                            session_id=session_id, verdict=verdict,
                        )
                        return {"reply": answer, "rounds": round_num,
                                "tool_calls": tool_call_log, "route": "complex"}

                    final_verdict = verdict  # save for episode recording on failure
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
                                new_parts.append(f"[{r.tool_name}] {r.content[:800]}")
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

                        # Re-solve with accumulated observations (GSSC)
                        solver_structured = await self._build_structured(
                            question, memory_context, session_id,
                            difficulty="complex", current_round=round_num,
                            planner_output=plan, search_results=exec_results,
                        )
                        solve_kwargs = dict(
                            question=question, observations=all_obs_text,
                            history_ctx=history_ctx,
                        )
                        if solver_structured is not None:
                            solve_kwargs["structured_prompt"] = solver_structured
                        else:
                            solve_kwargs["solver_ctx"] = self._ctx_router.build_solver(
                                question, plan=plan, search_results=exec_results,
                            )
                        answer = await self._planner.solve(**solve_kwargs)
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
                        solver_structured = await self._build_structured(
                            question, memory_context, session_id,
                            difficulty="complex", current_round=round_num,
                            planner_output=plan, search_results=exec_results,
                        )
                        solve_kwargs = dict(
                            question=question, observations=all_obs_text,
                            history_ctx=history_ctx, reasoning_feedback=fb,
                        )
                        if solver_structured is not None:
                            solve_kwargs["structured_prompt"] = solver_structured
                        else:
                            solve_kwargs["solver_ctx"] = self._ctx_router.build_solver(
                                question, plan=plan, search_results=exec_results,
                            )
                        answer = await self._planner.solve(**solve_kwargs)
                        last_answer = answer
                        continue  # re-reflect in inner loop

        self._record_episode(question, "complex", tool_call_log, MAX_COMPLEX_ROUNDS,
                             False, session_id=session_id, verdict=final_verdict)
        return {"reply": last_answer, "rounds": MAX_COMPLEX_ROUNDS,
                "tool_calls": tool_call_log, "route": "complex"}

    # ── Episode recording ────────────────────────────────────────

    def _record_episode(self, question: str, route: str, tool_calls: list[dict],
                        rounds: int, success: bool, session_id: str = "",
                        verdict: dict | None = None):
        """Record a structured episode from QA outcome.

        Populates diagnosis + lesson fields from Reflector verdict when available.
        """
        if self._memory is None or self._memory.episodic is None:
            return
        try:
            ep_data: dict = {
                "task": {"goal": question, "type": route},
                "context": {},
                "actions": [{"type": "qa", "tool_calls": tool_calls[:10]}],
                "observations": [
                    f"路由: {route}",
                    f"轮数: {rounds}",
                    f"工具调用: {len(tool_calls)} 次",
                ],
                "outcome": {"success": success},
                "reflection": {},
            }
            if verdict is not None:
                ins_type = verdict.get("insufficiency_type", "")
                ep_data["insufficiency_type"] = ins_type
                ep_data["failure_stage"] = self._ins_type_to_stage(ins_type)
                ep_data["missing_aspects"] = verdict.get("missing", [])
                ep_data["suggested_queries"] = verdict.get("suggested_queries", [])
                if ins_type == "plan":
                    ep_data["lesson"] = "任务分解不完整，需要检查子问题覆盖维度"
                    ep_data["corrected_behavior"] = "分解后显式检查：定义/公式/对比/应用四个维度"
                elif ins_type == "knowledge":
                    ep_data["lesson"] = "搜索词未命中，需要换同义词或更精确的术语"
                    ep_data["corrected_behavior"] = f"尝试搜索: {'; '.join(verdict.get('suggested_queries', [])[:3])}"
                elif ins_type == "reasoning":
                    ep_data["lesson"] = "资料已充足但综合推理有缺陷"
                    issues = verdict.get("issues", [])
                    ep_data["corrected_behavior"] = f"修正: {'; '.join(issues[:3])}" if issues else "更充分地利用搜索资料"
            elif success:
                ep_data["failure_stage"] = "none"
                ep_data["good_pattern"] = f"{route} 路径一次成功"
                ep_data["lesson"] = f"{route} 路径处理此问题有效"

            self._memory.episodic.record(ep_data, session_id=session_id,
                                         user_id=getattr(self, '_user_id', ''))
        except Exception as e:
            logger.warning("Failed to record episode: %s", e)

    @staticmethod
    def _ins_type_to_stage(ins_type: str) -> str:
        """Map Reflector insufficiency_type to failure_stage."""
        return {"plan": "planner.plan", "knowledge": "rewriter",
                "reasoning": "planner.solve"}.get(ins_type, "")

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


def get_agent(config: Configuration, gssc_pipeline=None, rag_skill=None,
              memory_manager=None) -> QASystem:
    """Get or create the QASystem singleton."""
    global _agent
    if _agent is None:
        _agent = QASystem(config, gssc_pipeline=gssc_pipeline, rag_skill=rag_skill,
                          memory_manager=memory_manager)
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
