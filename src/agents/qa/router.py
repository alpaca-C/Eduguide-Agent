# QuestionRouter — 问题分类 + 难度判断

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...prompts.qa.router import SYSTEM_PROMPT, ROUTER_PROMPT
from ...context_builder import RouterContext, PromptBuilder
from ...context_builder.schema import StructuredPrompt

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
        self._llm = self._make_llm(max_tokens=200)  # classification task, ~50 tokens output

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        # Rewriter output — injected into Router prompt for better classification
        keywords = input.metadata.get("rewritten_keywords", [])
        keywords_str = ", ".join(keywords) if keywords else "（无）"
        # Accept StructuredPrompt (GSSC, preferred), RouterContext, or legacy string
        structured = input.metadata.get("structured_prompt")
        router_ctx = input.metadata.get("router_context")
        legacy_history = input.metadata.get("chat_history", "")

        try:
            if structured is not None and isinstance(structured, StructuredPrompt):
                messages = [
                    SystemMessage(content=ROUTER_PROMPT.format(
                        question=question, keywords=keywords_str,
                    ) + "\n\n" + structured.to_prompt()),
                    HumanMessage(content="请分析以上问题的难度和类型。"),
                ]
            elif router_ctx is not None and isinstance(router_ctx, RouterContext):
                messages = PromptBuilder.build(
                    system=ROUTER_PROMPT.format(question=question, keywords=keywords_str),
                    context=router_ctx,
                    user="请分析以上问题的难度和类型。",
                )
            else:
                # Legacy path: string concatenation
                prompt = ROUTER_PROMPT.format(question=question, keywords=keywords_str)
                if legacy_history:
                    prompt += f"\n\n{legacy_history}"
                messages = [
                    SystemMessage(content=prompt),
                    HumanMessage(content="请分析以上问题的难度和类型。"),
                ]

            resp = await self._llm_retry(messages)
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

    async def direct_answer(self, question: str, history_ctx: str = "") -> str:
        """Answer a trivial question directly (1 LLM call, no tools)."""
        prompt = SYSTEM_PROMPT
        if history_ctx:
            prompt += f"\n\n{history_ctx}\n\n请结合以上对话历史回答学生当前的问题。"
        resp = await self._llm_retry([
            SystemMessage(content=prompt),
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
