# ContextRouter — builds agent-specific typed contexts from MemoryContext.
#
# DEPRECATED: GSSCPipeline.run() now produces StructuredPrompt with all
# sections (role_policies, task, state, evidence, context, output_format).
# Each agent reads what it needs directly. ContextRouter is kept only as
# a fallback when GSSC is unavailable (e.g. CI).
#
# Different agents need different slices of memory. The ContextRouter
# extracts the right fields from MemoryContext + runtime state and
# packages them into the typed context dataclass for each agent.

from __future__ import annotations

import logging

from .contexts import (
    RouterContext, SolverContext, PlannerContext, ReflectorContext,
    RewriterContext, BaseContext,
)

logger = logging.getLogger(__name__)


class ContextRouter:
    """Routes memory + runtime state → agent-specific typed contexts."""

    # ── Public API ──────────────────────────────────────────────────────

    def build_router(
        self,
        question: str,
        memory_context=None,       # MemoryContext from MemoryManager.recall()
        user_intent: str = "",
    ) -> RouterContext:
        """Build context for QuestionRouter — classify + decompose."""
        recent = ""
        if memory_context is not None:
            recent = getattr(memory_context, 'history_context', '')

        return RouterContext(
            question=question,
            recent_history=recent,
            user_intent=user_intent,
        )

    def build_rewriter(
        self,
        question: str,
        memory_context=None,
        history_ctx: str = "",
    ) -> RewriterContext:
        """Build context for QueryRewriter — NL → search keywords.

        Extracts conversation history from memory_context (preferred)
        or falls back to the raw history_ctx string.
        """
        history = history_ctx
        if not history and memory_context is not None:
            history = getattr(memory_context, 'history_context', '')
        return RewriterContext(question=question, history=history)

    def build_solver(
        self,
        question: str,
        plan: list[dict] | None = None,
        search_results: list | None = None,
        doc_filter: set[str] | None = None,
        history_ctx: str = "",
    ) -> SolverContext:
        """Build context for DirectSolver — search + synthesize."""
        observations, evidence, citations = self._extract_from_results(search_results)

        return SolverContext(
            question=question,
            plan=plan or [],
            observations=observations,
            history=history_ctx,
            evidence=evidence,
            citations=citations,
        )

    def build_planner(
        self,
        question: str,
        memory_context=None,
        seed_decomposition: list[str] | None = None,
        feedback: str = "",
        current_round: int = 1,
    ) -> PlannerContext:
        """Build context for Planner — decompose complex questions."""
        history = ""
        if memory_context is not None:
            history = getattr(memory_context, 'history_context', '')

        candidates = []
        if memory_context is not None:
            candidates = getattr(memory_context, 'available_docs', [])

        constraints = []
        if current_round <= 2:
            constraints.append("前 2 轮只使用本地教材检索 (rag_search)")
        else:
            constraints.append("第 3 轮可以使用网络搜索 (web_search) 补充")
        if seed_decomposition:
            constraints.append(f"Router 建议的子问题: {'; '.join(seed_decomposition[:5])}")
        if feedback:
            constraints.append(f"上一轮反馈: {feedback}")

        return PlannerContext(
            question=question,
            history=history,
            retrieved_candidates=candidates,
            constraints=constraints,
        )

    def build_reflector(
        self,
        question: str,
        answer: str,
        observations: str = "",
    ) -> ReflectorContext:
        """Build context for Reflector — review answer quality."""
        return ReflectorContext(
            question=question,
            answer=answer,
            evidence=observations,
            evaluation_rules=[
                "回答是否完整覆盖了问题的所有方面",
                "回答是否有教材原文支撑（引用来源）",
                "是否存在事实错误或逻辑漏洞",
                "对于复杂问题，是否分解为子问题逐一解答",
                "如果信息不足，是否诚实告知而非编造",
            ],
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_from_results(search_results: list | None) -> tuple[str, list[str], list[str]]:
        """Extract observations, evidence, and citations from search results."""
        if not search_results:
            return "", [], []

        obs_parts = []
        evidence = []
        citations = []

        for sr in search_results:
            sub_q = sr.get("question", "")
            if sub_q:
                obs_parts.append(f"\n### 子问题 {sr.get('id', '')}: {sub_q}")

            for r in sr.get("results", []):
                content = getattr(r, "content", str(r))
                if not content.strip():
                    continue

                # Truncate per-result content
                truncated = content[:600]
                obs_parts.append(f"[{getattr(r, 'tool_name', 'search')}] {truncated}")

                # Extract source citations
                meta = getattr(r, "metadata", {}) or {}
                doc = meta.get("doc_filename", "")
                if doc:
                    citations.append(doc)

                # Extract evidence snippets (first meaningful sentence)
                first_line = content.strip().split("\n")[0][:200]
                if first_line and "未找到" not in first_line:
                    evidence.append(first_line)

        return "\n".join(obs_parts), evidence[:10], list(dict.fromkeys(citations))[:10]
