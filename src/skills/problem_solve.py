# ProblemSolveSkill — wraps the full QA pipeline as a Supervisor-callable skill.
#
# Internal: QASystem.answer() → Router → Planner → Executor → Solver → Reflector.
# Supervisor only gives it SkillInput and receives SkillOutput.

from __future__ import annotations

import logging

from .skill_base import Skill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


class ProblemSolveSkill(Skill):
    """全流程教材答疑——包装现有的 QASystem 为 Supervisor 可调用的 Skill。"""

    def __init__(self, qa_system):
        """Wrap an existing QASystem instance."""
        self._qa = qa_system

    @property
    def name(self) -> str:
        return "problem_solve"

    @property
    def description(self) -> str:
        return (
            "完整的教材答疑流程：问题分类 → 多步规划 → 并发检索 → 结果综合 → 反思审核。"
            "适合所有需要基于教材回答的问题。"
        )

    async def execute(self, input: SkillInput) -> SkillOutput:
        """Execute the full QA pipeline.

        Reads QA-specific fields from input.params:
          - doc_filter: set[str] — limit search to specific documents
          - tutor_mode: bool    — Socratic tutoring mode
          - chat_history: list  — raw chat messages (optional)
        """
        doc_filter = input.params.get("doc_filter")
        tutor_mode = input.params.get("tutor_mode", False)
        chat_history = input.params.get("chat_history", [])
        memory_ctx = input.memory_context

        try:
            result = await self._qa.answer(
                question=input.question,
                doc_filter=doc_filter,
                chat_history=chat_history,
                memory_context=memory_ctx,
                tutor_mode=tutor_mode,
            )
            return SkillOutput(
                reply=result.get("reply", ""),
                rounds=result.get("rounds", 0),
                tool_calls=result.get("tool_calls", []),
                route=result.get("route", "unknown"),
                success=True,
            )
        except Exception as e:
            logger.error("ProblemSolveSkill failed: %s", e)
            return SkillOutput(
                reply=f"处理出错: {e}",
                success=False,
                error=str(e),
            )
