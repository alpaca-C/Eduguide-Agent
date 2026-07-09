# Reflector — structured review with search suggestions

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...prompts.qa.router import SYSTEM_PROMPT
from ...prompts.qa.reflector import REFLECT_PROMPT

logger = logging.getLogger(__name__)


class Reflector(BaseAgent):
    """Reviews answer quality and returns structured feedback.

    Output contains specific gaps and suggested search queries — Planner
    can use these directly in the next iteration rather than guessing.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        # Use a separate LLM instance — review needs careful reasoning
        self._llm = self._make_llm()

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        answer = input.metadata.get("answer", "")
        observations = input.metadata.get("observations", "")

        try:
            verdict = await self.review(question, answer, observations)
            return AgentOutput(success=True, metadata=verdict)
        except Exception as e:
            logger.error("Reflector review failed: %s", e)
            # On failure: assume sufficient (don't infinite loop)
            return AgentOutput(success=True, metadata={
                "verdict": "SUFFICIENT",
                "missing": [], "suggested_queries": [], "issues": [],
                "reason": f"Review failed: {e}",
            })

    async def review(self, question: str, answer: str, observations: str, history_ctx: str = "") -> dict:
        """Return structured verdict with gaps and suggested search queries."""
        prompt = SYSTEM_PROMPT + "\n\n" + REFLECT_PROMPT.format(
            question=question, answer=answer, observations=observations,
        )
        if history_ctx:
            prompt += f"\n\n{history_ctx}"

        resp = await self._llm_retry([
            SystemMessage(content=prompt),
            HumanMessage(content="请判断回答是否充分。"),
        ])
        text = resp.content if hasattr(resp, "content") else str(resp)
        data = self._parse_json(text)

        logger.info(
            "Reflector verdict=%s missing=%d queries=%d",
            data.get("verdict", "?"),
            len(data.get("missing", [])),
            len(data.get("suggested_queries", [])),
        )
        return data

    @staticmethod
    def _parse_json(text: str) -> dict:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {
            "verdict": "SUFFICIENT",
            "missing": [], "suggested_queries": [], "issues": [],
            "reason": "JSON parse failed, assuming sufficient",
        }
