# GSSC Schema — Fragment, ScoredFragment, StructuredPrompt
#
# Gather → Select → Structure → Compress 流水线的数据结构。

from __future__ import annotations

from dataclasses import dataclass, field

# ── Template ────────────────────────────────────────────────────────────────

# {section_name} placeholders filled by Structurer
TEMPLATE = """## Role & Policies
{role_policies}

## Task
{task}

## State
{state}

## Evidence
{evidence}

## Context
{context}

## Output Format
{output_format}"""

TEMPLATE_SECTIONS = ["role_policies", "task", "state", "evidence", "context", "output_format"]


# ── Fragment (Gather 阶段产出) ──────────────────────────────────────────────

@dataclass
class Fragment:
    """从某个来源汇集的一个候选信息碎片。"""
    source: str            # "user_question"|"conversation"|"search_result"
                           # |"kg_concept"|"episodic"|"system_policy"|"tool"
    content: str           # 文本内容
    priority: int = 0      # 0=低, 1=中, 2=高
    metadata: dict = field(default_factory=dict)
    # metadata 常见字段: timestamp, doc_name, relevance_score, episode_id, round


# ── ScoredFragment (Select 阶段产出) ────────────────────────────────────────

@dataclass
class ScoredFragment:
    """经过评分和筛选的碎片。"""
    fragment: Fragment
    relevance: float = 0.0   # 0..1 与当前问题的相关性
    recency: float = 0.0     # 0..1 新近性 (1 = 最新)
    total_score: float = 0.0


# ── StructuredPrompt (Structure + Compress 阶段产出) ───────────────────────

@dataclass
class StructuredPrompt:
    """组织进固定模板的结构化 Prompt。"""
    sections: dict[str, str] = field(default_factory=dict)
    # sections keys: role_policies, task, state, evidence, context, output_format
    token_estimate: int = 0
    compressed: bool = False

    def to_prompt(self) -> str:
        """渲染最终的 Prompt 文本。"""
        return TEMPLATE.format(**{
            name: self.sections.get(name, "（无）")
            for name in TEMPLATE_SECTIONS
        })

    def total_chars(self) -> int:
        return sum(len(v) for v in self.sections.values())
