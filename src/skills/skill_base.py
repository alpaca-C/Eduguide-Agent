# Skill base — SkillInput, SkillOutput, Skill (ABC)
#
# Skills are the units of work Supervisor delegates to.
# SkillInput carries universal fields (question, memory_context) +
# skill-specific params (dict, filled by API layer).

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
    """Abstract skill — unit of work for Supervisor."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def execute(self, input: SkillInput) -> SkillOutput: ...
