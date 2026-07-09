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

logger = logging.getLogger(__name__)


class QueryRewriter(BaseAgent):
    """Expands a student question into 3-5 search-optimized keywords.

    Cost: 1 fast LLM call (~200 tokens output). Adds ~1s latency.
    Benefit: Significantly improves FTS5 and KG recall for Chinese questions.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._llm = self._make_llm(temperature=0.0, max_tokens=150)

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        try:
            queries = await self.rewrite(question)
            return AgentOutput(success=True, metadata={"queries": queries})
        except Exception as e:
            logger.warning("QueryRewriter failed: %s, using original question", e)
            return AgentOutput(success=True, metadata={"queries": [question]})

    async def rewrite(self, question: str) -> list[str]:
        """Generate 3-5 optimized search queries from a natural language question."""
        resp = await self._llm_retry([
            SystemMessage(content=REWRITE_PROMPT.format(question=question)),
            HumanMessage(content="请生成搜索查询词。"),
        ])
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
