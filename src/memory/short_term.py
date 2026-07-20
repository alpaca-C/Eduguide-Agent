# ShortTermMemory — conversation history management
#
# Wraps MemoryStore session methods and provides the history compression
# logic (moved from QASystem._build_history_context).
#
# Usage:
#   stm = ShortTermMemory(store)
#   history = stm.get_history(session_id)
#   ctx = stm.build_context(history)        # compressed for prompt injection

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Constants (moved from QASystem)
MAX_HISTORY_CHARS = 4000   # Max chars of compressed history for prompts
COMPRESS_THRESHOLD = 12    # Messages before compressing older ones into summary


class ShortTermMemory:
    """Conversation history manager.

    Thin wrapper over MemoryStore's session methods, plus the history
    compression logic previously embedded in the QA orchestrator.
    """

    def __init__(self, store):
        """Wrap an existing MemoryStore instance.

        Args:
            store: MemoryStore from src.memory.store.
        """
        self._store = store

    # ── Delegated session methods ───────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str):
        """Add a chat message to a session and bump updated_at."""
        return self._store.add_chat_message(session_id, role, content)

    def get_history(self, session_id: str) -> list[dict]:
        """Get chat history for a session ordered by time."""
        return self._store.get_chat_history(session_id)

    def save_session(self, session_id: str, topic: str,
                     plan: list[dict] | None = None, report: str = ""):
        return self._store.save_session(session_id, topic, plan, report)

    def get_session(self, session_id: str) -> dict | None:
        return self._store.get_session(session_id)

    def list_sessions(self) -> list[dict]:
        return self._store.list_sessions()

    def delete_session(self, session_id: str):
        return self._store.delete_session(session_id)

    # ── History compression ─────────────────────────────────────────────

    def build_context(self, chat_history: list[dict] | None) -> str:
        """Convert chat_history to a compact context string for prompt injection.

        When history exceeds COMPRESS_THRESHOLD messages, older turns
        are summarized. Total output capped at MAX_HISTORY_CHARS.

        Moved from QASystem._build_history_context().
        """
        if not chat_history:
            return ""

        total = len(chat_history)

        if total <= 6:
            # Short history: include all messages directly
            parts = []
            for msg in chat_history[-6:]:
                role = "学生" if msg.get("role") == "user" else "答疑老师"
                parts.append(f"- {role}: {msg.get('content', '')[:300]}")
            ctx = "【对话历史】\n" + "\n".join(parts)

        elif total <= COMPRESS_THRESHOLD:
            # Medium history: include last 10, truncate older
            parts = []
            for msg in chat_history[-10:]:
                role = "学生" if msg.get("role") == "user" else "答疑老师"
                parts.append(f"- {role}: {msg.get('content', '')[:200]}")
            ctx = f"【对话历史（最近 10 轮，共 {total} 轮）】\n" + "\n".join(parts)

        else:
            # Long history: compress older messages into summary
            recent = chat_history[-8:]
            older = chat_history[:-8]
            summary = self._summarize(older)
            parts = [f"【历史摘要】{summary}"]
            for msg in recent:
                role = "学生" if msg.get("role") == "user" else "答疑老师"
                parts.append(f"- {role}: {msg.get('content', '')[:200]}")
            ctx = f"【对话历史（共 {total} 轮，早期已摘要）】\n" + "\n".join(parts)

        if len(ctx) > MAX_HISTORY_CHARS:
            ctx = ctx[:MAX_HISTORY_CHARS] + "\n...（历史已截断）"
        return ctx

    @staticmethod
    def _summarize(messages: list[dict]) -> str:
        """Simple rule-based summary — no LLM call needed.

        Extracts topics from user questions to give the model context
        about what was discussed earlier in the conversation.
        """
        questions = [m.get("content", "")[:100] for m in messages if m.get("role") == "user"]
        if not questions:
            return "（之前的对话）"
        topics = "；".join(q[:60] for q in questions[-5:])
        return f"之前讨论过：{topics}"
