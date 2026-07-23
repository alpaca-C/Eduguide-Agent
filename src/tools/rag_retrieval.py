# RAG Retrieval Strategy — centrally managed two-tier retrieval.
#
# NOT a Supervisor Skill (doesn't implement Skill ABC). This is a retrieval
# strategy controller registered as a TOOL so Executor can call it.
#
# Default:  rag_search      (Dense + Cross-Encoder)  — fast, 80% of queries
# Escalate: rag_fullsearch  (Dense + Sparse + Graph + Cross-Encoder)
#           triggered when Reflector feedback or user marks answer bad.
#
# The strategy tracks which queries got poor feedback and auto-escalates
# on re-search within the same session.

from __future__ import annotations

import logging

from . import ToolResult, register_tool, get_tool_registry

logger = logging.getLogger(__name__)

_strategy_instance = None


class RAGRetrievalStrategy:
    """Central handler for RAG retrieval — chooses fast vs full based on state.

    Usage:
        strategy = RAGRetrievalStrategy()
        result = await strategy.search("高斯定理")

        # After bad feedback from Reflector:
        strategy.mark_unsatisfied("高斯定理")
        result = await strategy.search("高斯定理")  # → auto-escalates to full
    """

    def __init__(self):
        self._unsatisfied: set[str] = set()  # queries that got bad feedback
        self._escalate_session: set[str] = set()  # sessions where ANY query was bad

    # ── Public API ──────────────────────────────────────────────────────

    async def search(
        self, query: str, top_k: int = 5,
        filter_docs: set[str] | None = None,
        force_full: bool = False,
    ) -> ToolResult:
        """Retrieve with auto-escalation.

        Default: rag_search (Dense + CE)
        Auto-escalate: rag_fullsearch when query was previously unsatisfied
        """
        should_full = force_full or self._should_escalate(query)

        if should_full:
            logger.info("RAGRetrievalStrategy: FULLSEARCH for '%s' (unsatisfied=%s, force=%s)",
                         query[:60], query in self._unsatisfied, force_full)
            from .rag_search import rag_fullsearch
            result = await rag_fullsearch(query, top_k=top_k, filter_docs=filter_docs)
        else:
            from .rag_search import rag_search
            result = await rag_search(query, top_k=top_k, filter_docs=filter_docs)

        return result

    def mark_unsatisfied(self, query: str, session_id: str = ""):
        """Record that this query got poor feedback. Future searches auto-escalate."""
        self._unsatisfied.add(query.strip().lower())
        if session_id:
            self._escalate_session.add(session_id)
        logger.info("RAGRetrievalStrategy: marked '%s' as unsatisfied", query[:60])

    def mark_satisfied(self, query: str):
        """Clear the escalation flag for a query."""
        q = query.strip().lower()
        self._unsatisfied.discard(q)
        logger.info("RAGRetrievalStrategy: cleared '%s'", query[:60])

    def reset(self):
        """Clear all escalation state."""
        self._unsatisfied.clear()
        self._escalate_session.clear()

    # ── Internal ────────────────────────────────────────────────────────

    def _should_escalate(self, query: str) -> bool:
        """Check if this query or similar ones should use full search."""
        q = query.strip().lower()
        if q in self._unsatisfied:
            return True
        # Fuzzy match: if any unsatisfied query is a substring or vice versa
        for uq in self._unsatisfied:
            if uq in q or q in uq:
                return True
        return False


def get_rag_strategy() -> RAGRetrievalStrategy:
    """Get or create the global RAGRetrievalStrategy singleton."""
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = RAGRetrievalStrategy()
    return _strategy_instance


# ── Tool registration (so Executor can call it) ─────────────────────────
#
# rag_search  and rag_fullsearch are separate low-level tools (src/tools/rag_search.py).
# rag_skill  is this smart dispatcher — Executor uses it as the single entry point,
# and it auto-chooses between rag_search (fast) and rag_fullsearch (full) based on
# feedback history.

async def _rag_skill_search(query: str, top_k: int = 5,
                            filter_docs: set[str] | None = None) -> ToolResult:
    """Tool wrapper: delegates to RAGRetrievalStrategy."""
    strategy = get_rag_strategy()
    return await strategy.search(query, top_k=top_k, filter_docs=filter_docs)


register_tool(
    name="rag_skill",
    description="智能文档检索（默认Dense+CE，不满意时自动升级到全量检索）。",
    func=_rag_skill_search,
)
