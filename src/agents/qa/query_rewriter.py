# QueryRewriter — convert NL questions into search-optimized keyword queries
#
# Called before rag_search to improve FTS5 (sparse) and KG (graph) recall.
# Dense retrieval handles NL queries fine, but sparse/graph need good keywords.

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...prompts.qa.query_rewriter import REWRITE_PROMPT
from ...context_builder import RewriterContext, PromptBuilder
from ...context_builder.schema import StructuredPrompt

logger = logging.getLogger(__name__)


class QueryRewriter(BaseAgent):
    """Expands a student question into 3-5 search-optimized keywords.

    Accepts StructuredPrompt (GSSC, preferred), RewriterContext (typed), or raw history.
    Cost: 1 fast LLM call (~200 tokens output). Adds ~1s latency.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._llm = self._make_llm(temperature=0.0, max_tokens=150)

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        structured = input.metadata.get("structured_prompt")
        rewriter_ctx = input.metadata.get("rewriter_context")
        try:
            queries = await self.rewrite(question, structured_prompt=structured,
                                         rewriter_ctx=rewriter_ctx)
            return AgentOutput(success=True, metadata={"queries": queries})
        except Exception as e:
            logger.warning("QueryRewriter failed: %s, using original question", e)
            return AgentOutput(success=True, metadata={"queries": [question]})

    async def rewrite(self, question: str,
                      rewriter_ctx: RewriterContext | None = None,
                      history: str = "",
                      structured_prompt: StructuredPrompt | None = None) -> list[str]:
        """Generate 3-5 optimized search queries from a natural language question.

        Args:
            question: The student's question.
            structured_prompt: GSSC StructuredPrompt (preferred).
            rewriter_ctx: Typed RewriterContext (legacy).
            history: Legacy raw history string (fallback).
        """
        if structured_prompt is not None:
            messages = [
                SystemMessage(content=REWRITE_PROMPT.format(question=question)
                              + "\n\n" + structured_prompt.to_prompt()),
                HumanMessage(content="请基于上下文生成搜索查询词。"),
            ]
        elif rewriter_ctx is not None:
            messages = PromptBuilder.build(
                system=REWRITE_PROMPT.format(question=question),
                context=rewriter_ctx,
                user="请基于对话上下文生成搜索查询词。",
            )
        elif history:
            prompt = REWRITE_PROMPT.replace(
                "**学生问题：** {question}",
                f"**对话历史：**\n{history}\n\n**学生问题：** {{question}}",
            ).format(question=question)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content="请生成搜索查询词。"),
            ]
        else:
            prompt = REWRITE_PROMPT.format(question=question)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content="请生成搜索查询词。"),
            ]

        resp = await self._llm_retry(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return self._parse_queries(text, question)

    @staticmethod
    def _parse_queries(text: str, fallback: str) -> list[str]:
        """Parse LLM output into a list of query strings. One per line."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Filter out lines that are too short or look like headers/notes
        queries = [l for l in lines if len(l) >= 2 and not l.startswith("#") and not l.startswith("- ")]
        # Remove numbering prefixes like "1. " or "1) "
        cleaned = []
        for q in queries[:6]:
            import re
            q = re.sub(r'^[\d]+[\.\)、]\s*', '', q).strip()
            if q and len(q) >= 2:
                cleaned.append(q)
        return cleaned[:5] if cleaned else [fallback]
