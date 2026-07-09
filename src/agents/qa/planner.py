# Planner — decompose complex questions + summarize results

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...tools.rag_search import get_doc_names
from ...prompts.qa.router import SYSTEM_PROMPT
from ...prompts.qa.planner import PLAN_PROMPT, SOLVE_PROMPT

logger = logging.getLogger(__name__)


class Planner(BaseAgent):
    """Decomposes complex questions into sub-questions, then synthesizes results.

    Works with Executor (runs sub-question searches) and Reflector (reviews output).
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._llm = self._make_llm()

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        try:
            plan = await self.plan(question, feedback="")
            return AgentOutput(success=True, metadata={"plan": plan})
        except Exception as e:
            return AgentOutput(success=False, error=str(e))

    async def plan(self, question: str, feedback: str = "", history_ctx: str = "",
                   seed_decomposition: list[str] | None = None) -> list[dict]:
        """Decompose question into sub-questions. Incorporates reviewer feedback + history.

        Args:
            question: The student's original question.
            feedback: Reviewer feedback from a previous round.
            history_ctx: Formatted conversation history context.
            seed_decomposition: Optional initial sub-questions from Router, used to
                seed the Planner so it refines rather than starting from scratch.
        """
        doc_names = self._get_doc_list()
        doc_list = f"共{len(doc_names)}本：{', '.join(doc_names)}" if doc_names else "（暂无已上传教材）"

        # Build seed section from Router's decomposition
        seed_section = ""
        if seed_decomposition:
            seed_lines = [f"  {i+1}. {s}" for i, s in enumerate(seed_decomposition)]
            seed_section = (
                "**初步分解（来自前置分析，可直接使用或改进）：**\n"
                + "\n".join(seed_lines)
                + "\n\n请审查以上分解是否合理，补充或调整子问题后输出最终分解。\n"
            )

        feedback_section = ""
        if feedback:
            feedback_section = (
                f"**上一轮审核反馈：**\n{feedback}\n"
                f"请根据反馈补充遗漏的子问题。\n"
            )

        prompt = PLAN_PROMPT.format(
            question=question, doc_list=doc_list,
            seed_section=seed_section,
            feedback_section=feedback_section,
        )
        if history_ctx:
            prompt += f"\n\n{history_ctx}"

        resp = await self._llm_retry([
            SystemMessage(content=prompt),
            HumanMessage(content="请分解以上问题。"),
        ])
        text = resp.content if hasattr(resp, "content") else str(resp)
        data = self._parse_json(text)
        return data.get("sub_questions", [])

    async def solve(self, question: str, observations: str, history_ctx: str = "") -> str:
        """Synthesize answer from sub-question results, with conversation context."""
        prompt = SYSTEM_PROMPT + "\n\n" + SOLVE_PROMPT.format(
            observations=observations, question=question,
        )
        if history_ctx:
            prompt += f"\n\n{history_ctx}"

        resp = await self._llm_retry([
            SystemMessage(content=prompt),
            HumanMessage(content="请综合以上搜索结果回答学生原始问题。"),
        ])
        return resp.content if hasattr(resp, "content") else str(resp)

    @staticmethod
    def _get_doc_list() -> list[str]:
        try:
            return get_doc_names()
        except Exception:
            return []

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}
