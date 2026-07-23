# GSSCPipeline — Gather → Select → Structure → Compress
#
# The main orchestrator that chains the four stages into a single call.
# Used by the QA orchestrator to build structured prompts for each agent.

from __future__ import annotations

import logging

from .gather import Gatherer
from .select import Selector
from .structure import Structurer
from .compress import Compressor
from .schema import StructuredPrompt, Fragment

logger = logging.getLogger(__name__)


class GSSCPipeline:
    """Gather → Select → Structure → Compress 完整流水线。

    Usage:
        pipeline = GSSCPipeline(memory_manager, tool_registry)
        prompt = await pipeline.run(
            question="什么是高斯定理",
            session_id="abc123",
            difficulty="complex",
            search_results=[...],
            feedback="需要补充矢量性",
            current_round=2,
        )
        # prompt.to_prompt() → 最终的 LLM 输入
    """

    def __init__(
        self,
        memory_manager=None,
        tool_registry: dict | None = None,
        token_budget: int = 3000,
        hard_limit: int = 4000,
        relevance_weight: float = 0.6,
        recency_weight: float = 0.4,
        min_score: float = 0.10,
    ):
        self.gatherer = Gatherer(memory_manager, tool_registry)
        self.selector = Selector(
            token_budget=token_budget,
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            min_score=min_score,
        )
        self.structurer = Structurer()
        self.compressor = Compressor(hard_limit=hard_limit)

    async def run(
        self,
        question: str,
        session_id: str = "",
        difficulty: str = "moderate",
        planner_output: list[dict] | None = None,
        search_results: list | None = None,
        feedback: str = "",
        current_round: int = 1,
        memory_context=None,          # MemoryContext from MemoryManager.recall()
        current_answer: str = "",      # Answer under review (for Reflector)
    ) -> StructuredPrompt:
        """Execute the full GSSC pipeline.

        Args:
            question: User's current question.
            session_id: Session for conversation history.
            difficulty: "trivial" | "moderate" | "complex".
            planner_output: Planner's sub-question plan (complex path).
            search_results: Executor's search results from current round.
            feedback: Reflector's feedback from previous round.
            current_round: Current reflection round (1, 2, or 3).
            memory_context: Pre-built MemoryContext from MemoryManager.recall().
                Passed through to EpisodicSource to avoid redundant ChromaDB queries.
            current_answer: The answer being reviewed (for Reflector).
        """
        # ── G: Gather ─────────────────────────────────────────────────
        fragments: list[Fragment] = await self.gatherer.gather(
            question, session_id,
            difficulty=difficulty,
            planner_output=planner_output,
            search_results=search_results,
            feedback=feedback,
            current_round=current_round,
            memory_context=memory_context,
            current_answer=current_answer,
        )

        # ── S: Select ─────────────────────────────────────────────────
        selected = self.selector.select(fragments, question)

        # ── S: Structure ──────────────────────────────────────────────
        prompt = self.structurer.structure(selected)

        # ── C: Compress ───────────────────────────────────────────────
        prompt = self.compressor.compress(prompt)

        logger.info(
            "GSSC: %d fragments → %d selected → %d sections, %d chars%s",
            len(fragments), len(selected),
            sum(1 for v in prompt.sections.values() if v != "（无）"),
            prompt.total_chars(),
            " (compressed)" if prompt.compressed else "",
        )

        return prompt
