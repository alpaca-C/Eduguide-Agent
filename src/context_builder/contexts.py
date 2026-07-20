# Per-agent typed context dataclasses.
#
# Each agent gets exactly the fields it needs — no more, no less.
# to_prompt() renders the context into a structured markdown block
# that PromptBuilder injects as a separate section.

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class BaseContext(ABC):
    """Abstract context — all agent contexts implement to_prompt()."""

    @abstractmethod
    def to_prompt(self) -> str:
        """Render context as a structured prompt block."""
        ...


# ── Router ──────────────────────────────────────────────────────────────────

@dataclass
class RouterContext(BaseContext):
    """Context for QuestionRouter — classify difficulty + decompose."""
    question: str
    recent_history: str = ""       # compressed recent conversation
    user_intent: str = ""          # inferred: "ask_definition" | "solve_problem" | ...

    def to_prompt(self) -> str:
        parts = []
        if self.recent_history.strip():
            parts.append(f"### 对话历史\n{self.recent_history}")
        if self.user_intent.strip():
            parts.append(f"### 用户意图\n{self.user_intent}")
        return "\n\n".join(parts)


# ── Solver ──────────────────────────────────────────────────────────────────

@dataclass
class SolverContext(BaseContext):
    """Context for DirectSolver — search + synthesize."""
    question: str
    plan: list[dict] = field(default_factory=list)       # sub-questions to search
    observations: str = ""          # retrieved document chunks
    evidence: list[str] = field(default_factory=list)    # key evidence snippets
    citations: list[str] = field(default_factory=list)   # source references

    def to_prompt(self) -> str:
        parts = []
        if self.plan:
            plan_text = "\n".join(
                f"- {s.get('id', i+1)}. {s.get('question', '')}"
                for i, s in enumerate(self.plan)
            )
            parts.append(f"### 搜索计划\n{plan_text}")
        if self.observations.strip():
            parts.append(f"### 检索结果\n{self.observations}")
        if self.evidence:
            ev_text = "\n".join(f"- {e}" for e in self.evidence[:10])
            parts.append(f"### 关键证据\n{ev_text}")
        if self.citations:
            cite_text = "\n".join(f"- {c}" for c in self.citations[:10])
            parts.append(f"### 引用来源\n{cite_text}")
        return "\n\n".join(parts)


# ── Planner ─────────────────────────────────────────────────────────────────

@dataclass
class PlannerContext(BaseContext):
    """Context for Planner — decompose complex questions."""
    question: str
    history: str = ""                       # conversation history
    retrieved_candidates: list[str] = field(default_factory=list)  # relevant doc names
    constraints: list[str] = field(default_factory=list)           # "前2轮只用RAG"

    def to_prompt(self) -> str:
        parts = []
        if self.history.strip():
            parts.append(f"### 对话历史\n{self.history}")
        if self.retrieved_candidates:
            cand_text = "\n".join(f"- {c}" for c in self.retrieved_candidates)
            parts.append(f"### 可用资料\n{cand_text}")
        if self.constraints:
            cons_text = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"### 约束条件\n{cons_text}")
        return "\n\n".join(parts)


# ── Reflector ───────────────────────────────────────────────────────────────

@dataclass
class ReflectorContext(BaseContext):
    """Context for Reflector — review answer quality."""
    question: str
    answer: str                         # the answer being reviewed
    evidence: str = ""                  # the observations used to generate the answer
    evaluation_rules: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        parts = []
        if self.answer.strip():
            # Truncate for reviewer context (reviewer needs question + evidence more than full answer)
            preview = self.answer[:500] + "..." if len(self.answer) > 500 else self.answer
            parts.append(f"### 待审核回答\n{preview}")
        if self.evidence.strip():
            # Truncate evidence for review
            ev = self.evidence[:1500] + "..." if len(self.evidence) > 1500 else self.evidence
            parts.append(f"### 支撑证据\n{ev}")
        if self.evaluation_rules:
            rules_text = "\n".join(f"- {r}" for r in self.evaluation_rules)
            parts.append(f"### 评判标准\n{rules_text}")
        return "\n\n".join(parts)
