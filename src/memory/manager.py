# MemoryManager — unified memory facade (short-term + episodic + semantic)
#
# Single entry point for all memory operations in the QA system.
# Composes ShortTermMemory, EpisodicMemory, and SemanticMemory.
#
# Caches (exact-match + semantic) live in src/cache/ — separate concern.
#
# Usage:
#   mm = MemoryManager(short_term, episodic, semantic)
#   ctx = await mm.recall(question, session_id)
#   mm.short_term.add_message(sid, role, content)
#   episodes = mm.episodic.recall("RAG优化")
#   result = await mm.semantic.search(query)

from __future__ import annotations

import logging

from .context import MemoryContext
from .short_term import ShortTermMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """统一记忆入口 — 管理三层记忆。

    1. ShortTermMemory  — 会话内的对话历史
    2. EpisodicMemory   — 跨会话的经验/教训积累
    3. SemanticMemory   — 文档知识（向量 + 知识图谱）

    QASystem 每次回答问题前调用 recall() 获取 MemoryContext。
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
    ):
        self.short_term = short_term
        self.episodic = episodic
        self.semantic = semantic

    # ── Unified recall ──────────────────────────────────────────────────

    async def recall(
        self,
        question: str,
        session_id: str | None = None,
    ) -> MemoryContext:
        """Recall all relevant memories for a question.

        1. Short-term: loads conversation history and compresses it
        2. Episodic:  semantically searches past experiences for relevant lessons
        3. Semantic:  lists available document names (lightweight)
        """
        # 1) Short-term: conversation history
        chat_history: list[dict] = []
        history_context = ""
        if session_id:
            try:
                chat_history = self.short_term.get_history(session_id)
            except Exception as e:
                logger.warning("Failed to load chat history for %s: %s", session_id, e)
        history_context = self.short_term.build_context(chat_history)

        # 2) Episodic: relevant past experiences
        episodes = []
        if self.episodic is not None:
            try:
                episodes = self.episodic.recall(question, top_k=3)
            except Exception as e:
                logger.warning("Episodic recall failed: %s", e)

        # 3) Semantic: available document names (lightweight)
        available_docs: list[str] = []
        if self.semantic is not None:
            try:
                available_docs = self.semantic.get_doc_names()
            except Exception as e:
                logger.warning("Semantic doc listing failed: %s", e)

        return MemoryContext(
            question=question,
            session_id=session_id,
            chat_history=chat_history,
            history_context=history_context,
            episodes=episodes,
            available_docs=available_docs,
        )
