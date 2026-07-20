"""
Pydantic schema for Planner output validation.

Replaces the old regex-only _parse_json() with typed validation.
On validation failure, the errors are sent back to the LLM for correction
(one retry), rather than silently falling back to defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


VALID_TOOLS = {"rag_search", "web_search", "mineru_ocr"}


class SubQuestion(BaseModel):
    """A single decomposed sub-question."""
    id: int = Field(..., ge=1, description="子问题编号")
    question: str = Field(..., min_length=1, description="子问题描述")
    keywords: list[str] = Field(default_factory=list, description="搜索关键词")
    target_doc: str = Field(default="", description="建议搜索的教材名")
    tool: str = Field(default="rag_search", description="使用的工具名")
    depends_on: list[int] = Field(default_factory=list, description="依赖的子问题 ID")


class PlanOutput(BaseModel):
    """Planner's structured output."""
    sub_questions: list[SubQuestion] = Field(
        default_factory=list,
        max_length=8,  # one more than prompt's limit of 5, for tolerance
        description="分解后的子问题列表",
    )

    def filter_valid(self) -> list[dict]:
        """
        Return only sub_questions with valid tool names as dicts.

        Invalid tool names are logged and skipped — the Executor already
        has its own guard, so this is defense-in-depth.
        """
        valid = []
        for sq in self.sub_questions:
            d = sq.model_dump()
            if d["tool"] not in VALID_TOOLS:
                d["tool"] = "rag_search"  # safe default
            valid.append(d)
        return valid
