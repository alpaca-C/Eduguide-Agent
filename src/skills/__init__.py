"""Skills module — Supervisor-delegable units of work.

A Skill packages metadata (name, description, trigger, examples) + execute().
Supervisor uses SkillRegistry to discover and select skills dynamically.
Supervisor does NOT know skill internals (agents, tools, prompts).

Trigger types:
  "default" — always available (e.g. problem_solve)
  "toggle"  — user activates via frontend toggle / API flag (e.g. exercise_tutor)
  "auto"    — reserved: LLM selects by matching description
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

# Re-export the unified ABC and data classes
from .skill_base import Skill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


# ── Skill metadata (lightweight, for Supervisor's LLM router) ──────────

@dataclass
class SkillMeta:
    """Public metadata for a skill. Supervisor reads this to route.

    Does NOT expose internal agents, tools, prompts, or config —
    Supervisor only sees what it needs to make a routing decision.
    """
    name: str
    description: str
    trigger: str                        # "default" | "toggle" | "auto"
    examples: list[str] = field(default_factory=list)  # representative questions


# ── SkillRegistry ──────────────────────────────────────────────────────

class SkillRegistry:
    """Central registry of all skills. Supervisor reads from this.

    Skills register themselves with metadata + execute().
    Supervisor calls get_all_meta() to build the LLM routing prompt —
    it sees SkillMeta but never touches Skill instances directly.
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """Register a skill instance."""
        self._skills[skill.name] = skill
        logger.info("Skill registered: %s [%s] — %s",
                     skill.name, skill.trigger, skill.description[:80])

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all_meta(self) -> list[SkillMeta]:
        """Return metadata for all registered skills.

        Supervisor uses this to build the LLM routing prompt.
        Skill internals (agents, tools, prompts) are NOT exposed.
        """
        return [
            SkillMeta(
                name=s.name,
                description=s.description,
                trigger=s.trigger,
                examples=s.examples,
            )
            for s in self._skills.values()
        ]

    def get_by_trigger(self, trigger: str) -> list[Skill]:
        """Get all skills with a given trigger type."""
        return [s for s in self._skills.values() if s.trigger == trigger]

    def list_names(self) -> list[str]:
        """Return registered skill names (for logging/debug)."""
        return list(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


# ── Backward-compat wrappers (some code may call these) ────────────────

_default_registry: SkillRegistry | None = None


def get_default_registry() -> SkillRegistry:
    """Get or create the global default SkillRegistry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry


def register_skill(skill: Skill):
    """Register a skill in the default registry."""
    get_default_registry().register(skill)


def get_skill(name: str) -> Skill | None:
    """Get a registered skill by name from the default registry."""
    return get_default_registry().get(name)
