# Executor — concurrent tool execution for Planner's sub-questions

from __future__ import annotations

import asyncio
import logging

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...tools import get_tool_registry, ToolResult
from .query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class Executor(BaseAgent):
    """Executes search queries concurrently.

    Enriches plan keywords with QueryRewriter output for better recall.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._tools = get_tool_registry()
        self._rewriter = QueryRewriter(config)

    async def run(self, input: AgentInput) -> AgentOutput:
        plan = input.metadata.get("plan", [])
        doc_filter = input.metadata.get("doc_filter")
        history_ctx = input.metadata.get("history_ctx", "")
        try:
            results = await self.execute(plan, doc_filter, history_ctx=history_ctx)
            return AgentOutput(success=True, metadata={"results": results})
        except Exception as e:
            return AgentOutput(success=False, error=str(e))

    async def execute(
        self, plan: list[dict], doc_filter: set[str] | None = None,
        history_ctx: str = "",
    ) -> list[dict]:
        """Execute planned searches in dependency-ordered rounds.

        Each plan item: {
            "id": 1, "question": "...", "keywords": ["kw1", "kw2"],
            "tool": "rag_search", "target_doc": "book.pdf",
            "depends_on": []           # optional: sub-question IDs this depends on
        }

        Sub-questions with no depends_on run first (concurrently within the round).
        Results from each round enrich the search queries of subsequent rounds.
        Backward-compatible: plan items without depends_on all run concurrently.

        Returns: [{ "id": 1, "question": "...", "results": [ToolResult, ...] }]
        """
        from src.harness import _agent_name
        _agent_name.set("Executor")

        if not plan:
            return []

        # ── Group into topological rounds ─────────────────────────
        rounds = self._topological_rounds(plan)
        logger.info("Executor: %d sub-questions in %d round(s)",
                     len(plan), len(rounds))

        resolved: dict[int, str] = {}   # sub_id → context summary
        by_sub: dict[int, dict] = {}    # sub_id → {id, question, results}

        for round_idx, round_ids in enumerate(rounds):
            if round_idx > 0:
                logger.info("Executor: round %d (%d sub-questions, enriched with prior results)",
                             round_idx + 1, len(round_ids))

            tasks = []
            task_meta: list[dict] = []

            for sub in plan:
                sub_id = sub.get("id", 0)
                if sub_id not in round_ids:
                    continue
                tool_name = sub.get("tool", "rag_search")
                # ★ Route rag_search → rag_skill when available (auto-escalation)
                if tool_name == "rag_search" and "rag_skill" in self._tools:
                    tool_name = "rag_skill"
                if tool_name not in self._tools:
                    logger.warning("Executor: unknown tool '%s' for sub %d", tool_name, sub_id)
                    continue

                # ── Enrich keywords ──────────────────────────────
                keywords = list(sub.get("keywords", []))
                sub_question = sub.get("question", "")

                # 1) Context enrichment: use prior-round results if this sub has dependencies
                deps = sub.get("depends_on", [])
                if deps and resolved:
                    ctx = "\n".join(resolved.get(d, "") for d in deps if d in resolved)
                    if ctx.strip():
                        try:
                            extra = await self._enrich_query_with_context(sub_question, ctx)
                            for ek in extra:
                                if ek not in keywords:
                                    keywords.append(ek)
                            logger.debug("Executor: context-enriched sub %d from %d deps → %d keywords",
                                         sub_id, len(deps), len(keywords))
                        except Exception:
                            pass

                # 2) Fallback: QueryRewriter if keywords still sparse
                if len(keywords) < 3 and sub_question:
                    try:
                        extra = await self._rewriter.rewrite(sub_question, history=history_ctx)
                        for ek in extra:
                            if ek not in keywords:
                                keywords.append(ek)
                        logger.debug("Executor: rewriter-enriched sub %d → %d keywords",
                                     sub_id, len(keywords))
                    except Exception:
                        pass

                for kw in keywords[:8]:
                    task_args = {"query": kw}
                    if tool_name == "rag_search":
                        task_args["filter_docs"] = doc_filter
                    tasks.append(self._tools[tool_name]["func"](**task_args))
                    task_meta.append({
                        "sub_id": sub_id,
                        "keyword": kw,
                        "tool": tool_name,
                        "target_doc": sub.get("target_doc", ""),
                    })

            if not tasks:
                continue

            # ── Execute round concurrently ────────────────────────
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # ── Group & store context for next rounds ─────────────
            for meta, result in zip(task_meta, results):
                sid = meta["sub_id"]
                if sid not in by_sub:
                    sub = next((s for s in plan if s.get("id") == sid), {})
                    by_sub[sid] = {
                        "id": sid,
                        "question": sub.get("question", ""),
                        "results": [],
                    }
                if isinstance(result, Exception):
                    by_sub[sid]["results"].append(ToolResult(
                        tool_name=meta["tool"], query=meta["keyword"],
                        content=f"搜索出错: {result}",
                    ))
                else:
                    by_sub[sid]["results"].append(result)

            # Store context summaries for newly resolved sub-questions
            for sid in round_ids:
                if sid in by_sub:
                    resolved[sid] = self._summarize_for_context(
                        by_sub[sid].get("question", ""),
                        by_sub[sid].get("results", []),
                    )

        return list(by_sub.values())

    # ── Dependency ordering ─────────────────────────────────────

    @staticmethod
    def _topological_rounds(plan: list[dict]) -> list[list[int]]:
        """Group sub-question IDs into topological rounds based on depends_on.

        Round 0: sub-questions with no unresolved dependencies (concurrent).
        Round 1: sub-questions that depend only on Round 0 results.
        ...
        Falls back to single-round (all concurrent) if depends_on is absent or circular.
        """
        deps: dict[int, set[int]] = {}
        for sub in plan:
            sid = sub.get("id", 0)
            deps[sid] = set(sub.get("depends_on", []) or [])

        # If no sub-question declares any dependency, single round
        if not any(deps.values()):
            return [list(deps.keys())]

        rounds: list[list[int]] = []
        remaining = set(deps.keys())

        while remaining:
            ready = {sid for sid in remaining if not (deps[sid] & remaining)}
            if not ready:
                # Circular dependency or reference to non-existent sub:
                # flush remaining as final round to avoid infinite loop
                logger.warning("Executor: circular/missing deps detected, flushing %d as final round",
                               len(remaining))
                rounds.append(list(remaining))
                break
            rounds.append(list(ready))
            remaining -= ready

        return rounds

    # ── Context enrichment ──────────────────────────────────────

    async def _enrich_query_with_context(self, question: str, context: str) -> list[str]:
        """Build search queries enriched with answers from prior sub-questions.

        Combines the current question with context from resolved dependencies
        and asks QueryRewriter to generate more targeted keywords.
        """
        truncated = context[:400]  # Keep compact to avoid query bloat
        combined = f"{question}\n[前置知识：{truncated}]"
        return await self._rewriter.rewrite(combined)

    @staticmethod
    def _summarize_for_context(question: str, results: list[ToolResult]) -> str:
        """Compact summary of search results for passing to the next round.

        Extracts key text snippets to help downstream sub-questions build
        better search queries, without bloating the enrichment context.
        """
        parts = [f"Q: {question}"]
        for r in results:
            if hasattr(r, "content") and r.content:
                # Take first meaningful chunk from each result
                content = r.content.strip()[:250]
                if content:
                    parts.append(content)
        return "\n".join(parts)[:500]
