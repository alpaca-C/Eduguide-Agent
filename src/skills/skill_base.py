# Skill base — SkillInput, SkillOutput, Skill (ABC)
#
# Skills are the units of work Supervisor delegates to.
# SkillInput carries universal fields (question, memory_context) +
# skill-specific params (dict, filled by API layer).
#
# Trigger types:
#   "default" — always available, Supervisor's fallback route
#   "toggle"  — user explicitly activates via frontend toggle / API flag
#   "auto"    — reserved: LLM selects by matching description to question

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class SkillInput:
    """Input to a skill execution.

    question + memory_context are universal.
    params carries skill-specific fields (e.g. doc_filter, tutor_mode).
    """
    question: str
    memory_context: object | None = None   # Supervisor injects this
    params: dict = field(default_factory=dict)


@dataclass
class SkillOutput:
    """Output from a skill execution."""
    reply: str
    rounds: int = 0
    tool_calls: list = field(default_factory=list)
    route: str = ""
    success: bool = True
    error: str = ""


class Skill(ABC):
    """Unified skill — unit of work for Supervisor.

    Each skill declares:
      - name, description: identity for registration and future LLM selection
      - trigger: how this skill gets activated
      - execute(): the actual work
      - (optional) system_prompt: for community-skill-style self-contained prompts
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def trigger(self) -> str:
        """How this skill is activated.

        "default" — always available, Supervisor routes here unless overridden.
        "toggle"  — user explicitly activates (frontend toggle / API flag).
        "auto"    — reserved for LLM-based selection by description.
        """
        return "default"

    @property
    def examples(self) -> list[str]:
        """Example questions this skill handles well.

        Used by Supervisor's LLM router to match user questions to skills.
        e.g. ["什么是库仑定律", "高斯定理怎么推导", "电场和磁场有什么区别"]
        """
        return []

    @property
    def system_prompt(self) -> str:
        """Optional self-contained prompt (community-skill pattern).
        Override in toggle-type skills that carry their own prompt.
        """
        return ""

    @abstractmethod
    async def execute(self, input: SkillInput) -> SkillOutput: ...
