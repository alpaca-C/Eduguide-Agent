# Gatherer — Stage 1 of GSSC: collect candidate fragments from all sources.
#
# Each Source.fetch() returns a list[Fragment]. The Gatherer runs all sources
# concurrently and flattens the results.

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

from .schema import Fragment

logger = logging.getLogger(__name__)


# ── Source interface ────────────────────────────────────────────────────────

class Source(ABC):
    """Abstract source of candidate fragments."""

    def __init__(self, priority: int = 1):
        self.priority = priority

    @abstractmethod
    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        """Collect fragments from this source. May be async (e.g. semantic search)."""
        ...


# ── Built-in sources ────────────────────────────────────────────────────────

class SystemPolicySource(Source):
    """Role & Policies — system instructions, always included (priority 2)."""

    def __init__(self):
        super().__init__(priority=2)

    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        return [
            Fragment(
                source="system_policy",
                content=(
                    "你是一个教材答疑助手，基于学生上传的教材内容回答问题。"
                    "回答应该准确、有据可查，引用具体的教材章节。"
                    "对于复杂问题，先分解再逐步回答。"
                    "如果教材中没有相关内容，诚实告知并建议网络搜索。"
                ),
                priority=self.priority,
                metadata={"type": "role"},
            ),
            Fragment(
                source="system_policy",
                content=(
                    "本轮难度: {difficulty}。当前第 {current_round} 轮。"
                    .format(**runtime) if runtime else ""
                ),
                priority=1,
                metadata={"type": "status"},
            ),
        ]


class UserQuestionSource(Source):
    """The user's current question — always highest priority."""

    def __init__(self):
        super().__init__(priority=2)

    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        feedback = runtime.get("feedback", "")
        content = f"学生提问: {question}"
        if feedback:
            content += f"\n上一轮反馈（需补充）: {feedback}"
        return [Fragment(
            source="user_question",
            content=content,
            priority=self.priority,
            metadata={"question": question},
        )]


class ConversationSource(Source):
    """Recent conversation history from ShortTermMemory."""

    def __init__(self, memory_manager):
        super().__init__(priority=1)
        self._mm = memory_manager

    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        if not session_id:
            return []
        try:
            history = self._mm.short_term.get_history(session_id)
            ctx = self._mm.short_term.build_context(history)
            if not ctx.strip():
                return []
            return [Fragment(
                source="conversation",
                content=ctx,
                priority=self.priority,
                metadata={"message_count": len(history), "timestamp": time.time()},
            )]
        except Exception as e:
            logger.warning("ConversationSource failed: %s", e)
            return []


class EpisodicSource(Source):
    """Relevant past experiences from EpisodicMemory."""

    def __init__(self, memory_manager):
        super().__init__(priority=1)
        self._mm = memory_manager

    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        try:
            episodes = self._mm.episodic.recall(question, top_k=3)
        except Exception as e:
            logger.warning("EpisodicSource failed: %s", e)
            return []

        fragments = []
        for ep in episodes:
            parts = [f"历史经验: {ep.task_goal}"]
            if ep.observations:
                parts.append(f"观察: {'; '.join(ep.observations[:3])}")
            lesson = ep.reflection.get("lesson", "")
            if lesson:
                parts.append(f"教训: {lesson}")
            fragments.append(Fragment(
                source="episodic",
                content="\n".join(parts),
                priority=self.priority,
                metadata={
                    "episode_id": ep.id,
                    "task_type": ep.task_type,
                    "timestamp": ep.created_at,
                },
            ))
        return fragments


class SearchResultSource(Source):
    """Search results from the current round's Executor output."""

    def __init__(self):
        super().__init__(priority=1)

    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        search_results = runtime.get("search_results") or []
        if not search_results:
            return []

        fragments = []
        for sr in search_results:
            # sr is a sub-result dict: {id, question, results: [ToolResult, ...]}
            sub_q = sr.get("question", "")
            results = sr.get("results", [])
            for r in results:
                content = getattr(r, "content", str(r))
                if not content.strip():
                    continue
                fragments.append(Fragment(
                    source="search_result",
                    content=f"[子问题: {sub_q}]\n{content[:800]}",
                    priority=self.priority,
                    metadata={
                        "sub_id": sr.get("id"),
                        "tool": getattr(r, "tool_name", ""),
                    },
                ))
        return fragments


class KGConceptSource(Source):
    """Knowledge graph concepts found during search."""

    def __init__(self):
        super().__init__(priority=0)

    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        search_results = runtime.get("search_results") or []
        fragments = []
        for sr in search_results:
            for r in sr.get("results", []):
                meta = getattr(r, "metadata", {}) or {}
                concepts = meta.get("concepts_found", 0)
                if concepts:
                    fragments.append(Fragment(
                        source="kg_concept",
                        content=f"知识图谱: 找到 {concepts} 个相关概念",
                        priority=0,
                        metadata={"sub_id": sr.get("id")},
                    ))
        return fragments


class ToolSource(Source):
    """Available tools and their descriptions."""

    def __init__(self, tool_registry: dict):
        super().__init__(priority=0)
        self._tools = tool_registry

    async def fetch(self, question: str, session_id: str, **runtime) -> list[Fragment]:
        if not self._tools:
            return []
        lines = ["可用工具:"]
        for name, info in self._tools.items():
            lines.append(f"- {name}: {info.get('description', '无描述')}")
        return [Fragment(
            source="tool",
            content="\n".join(lines),
            priority=0,
            metadata={"tool_count": len(self._tools)},
        )]


# ── Gatherer ────────────────────────────────────────────────────────────────

class Gatherer:
    """Stage 1: collect candidate fragments from all sources concurrently."""

    def __init__(self, memory_manager, tool_registry: dict | None = None):
        self._sources: list[Source] = []

        # Always-on sources
        self._sources.append(SystemPolicySource())
        self._sources.append(UserQuestionSource())

        # Memory-dependent sources
        if memory_manager is not None:
            self._sources.append(ConversationSource(memory_manager))
            self._sources.append(EpisodicSource(memory_manager))

        # Runtime sources (fed by Executor output)
        self._sources.append(SearchResultSource())
        self._sources.append(KGConceptSource())

        # Tool registry
        self._sources.append(ToolSource(tool_registry or {}))

    async def gather(self, question: str, session_id: str = "", **runtime) -> list[Fragment]:
        """Run all sources concurrently, return all fragments."""
        tasks = [src.fetch(question, session_id, **runtime) for src in self._sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        fragments: list[Fragment] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Gatherer: source '%s' failed: %s",
                               type(self._sources[i]).__name__, r)
            else:
                fragments.extend(r)

        logger.info("Gatherer: collected %d fragments from %d sources",
                     len(fragments), len(self._sources))
        return fragments
