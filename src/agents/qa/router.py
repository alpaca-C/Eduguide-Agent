# QuestionRouter — 问题分类 + 难度判断

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...prompts.qa.router import SYSTEM_PROMPT, ROUTER_PROMPT

logger = logging.getLogger(__name__)


class QuestionRouter(BaseAgent):
    """Analyzes question difficulty and type. Returns structured JSON.

    Output categories:
    - trivial:   greeting, chitchat, common sense → answer directly
    - moderate:  single-concept lookup → DirectSolver
    - complex:   multi-step reasoning, cross-doc comparison → Planner pipeline
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._llm = self._make_llm()

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        history_ctx = input.metadata.get("chat_history", "")
        try:
            prompt = ROUTER_PROMPT.format(question=question)
            if history_ctx:
                prompt += f"\n\n{history_ctx}"
            resp = await self._llm_retry([
                SystemMessage(content=prompt),
                HumanMessage(content="请分析以上问题的难度和类型。"),
            ])
            text = resp.content if hasattr(resp, "content") else str(resp)
            data = self._parse_json(text)
            return AgentOutput(success=True, metadata={
                "difficulty": data.get("difficulty", "moderate"),
                "reason": data.get("reason", ""),
                "target_docs": data.get("target_docs", []),
                "decomposition": data.get("decomposition", []),
            })
        except Exception as e:
            logger.warning("QuestionRouter failed: %s, defaulting to moderate", e)
            return AgentOutput(success=True, metadata={
                "difficulty": "moderate",
                "reason": f"Router failed: {e}",
                "target_docs": [],
                "decomposition": [],
            })

    async def direct_answer(self, question: str) -> str:
        """Answer a trivial question directly (1 LLM call, no tools)."""
        resp = await self._llm_retry([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ])
        return resp.content if hasattr(resp, "content") else str(resp)

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}
