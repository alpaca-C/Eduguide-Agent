# Supervisor — thin dispatch layer.
#
# Responsibilities:
#   1. Recall memory (via MemoryManager) and inject into SkillInput
#   2. Route to the right Skill (currently only problem_solve)
#   3. Save assistant reply to short-term memory
#   4. Return SkillOutput
#
# Does NOT know skill-specific fields — those live in SkillInput.params.

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..skills.skill_base import SkillInput, SkillOutput

logger = logging.getLogger(__name__)


@dataclass
class SupervisorOutput:
    """Lightweight wrapper — API layer consumes this."""
    reply: str
    session_id: str = ""
    rounds: int = 0
    tool_calls: int = 0
    route: str = ""


class Supervisor:
    """薄调度层——取记忆 → 注入 context → 喂 skill → 返回结果。

    不知道任何 skill-specific 字段。params 由 API 层填入。
    """

    def __init__(self, memory_manager, skills: dict[str, object]):
        self._memory = memory_manager
        self._skills = skills  # {skill_name: Skill}

    async def run(self, input: SkillInput, session_id: str = "") -> SupervisorOutput:
        """Handle a user question end-to-end.

        1. Recall + inject memory
        2. Route to skill
        3. Save reply
        4. Return
        """
        # ── 1. Recall memory, inject into input ──────────────────────
        if self._memory is not None:
            try:
                input.memory_context = await self._memory.recall(
                    input.question, session_id,
                )
            except Exception as e:
                logger.warning("Memory recall failed: %s", e)

        # ── 2. Route to skill ───────────────────────────────────────
        skill_name = input.params.get("skill", "problem_solve")
        skill = self._skills.get(skill_name)
        if skill is None:
            return SupervisorOutput(reply=f"未知技能: {skill_name}")

        result = await skill.execute(input)

        # ── 3. Save assistant reply to short-term memory ─────────────
        if self._memory is not None and session_id:
            try:
                self._memory.short_term.add_message(session_id, "assistant", result.reply)
            except Exception as e:
                logger.warning("Failed to save assistant reply: %s", e)

        # ── 4. Return ────────────────────────────────────────────────
        return SupervisorOutput(
            reply=result.reply,
            session_id=session_id,
            rounds=result.rounds,
            tool_calls=len(result.tool_calls),
            route=result.route,
        )
