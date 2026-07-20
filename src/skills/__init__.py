"""Skills module — lightweight abstraction over prompt + tool chain.

A Skill packages a system prompt with a list of tools into a reusable
unit that Planner/Executor can invoke by name. Compared to raw tools,
skills carry richer context (the prompt) and can encapsulate multi-step
interaction patterns (e.g. Socratic tutoring across multiple rounds).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Skill:
    """A reusable capability: prompt + tool chain + execution constraints.

    Attributes:
        name: Unique skill identifier (e.g. "exercise_tutor").
        description: One-line summary — used by LLM to select skills.
        system_prompt: The system prompt template for this skill.
            Supports {question}, {observations}, {chat_history} placeholders.
        tools: Tool names this skill needs (from tool_registry).
    """
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)


# Registry of available skills
_skill_registry: dict[str, Skill] = {}


def register_skill(skill: Skill):
    """Register a skill for the agent to use."""
    _skill_registry[skill.name] = skill


def get_skill(name: str) -> Skill | None:
    """Get a registered skill by name. Returns None if not found."""
    return _skill_registry.get(name)


def get_skill_registry() -> dict[str, Skill]:
    """Get all registered skills."""
    return dict(_skill_registry)
