# Memory module — unified memory management for the Document QA System.
#
# Three memory layers:
#   ShortTermMemory  — conversation history (SQLite session_messages)
#   EpisodicMemory   — past experiences and learnings (SQLite + ChromaDB)
#   SemanticMemory   — document knowledge (ChromaDB + FTS5 + KnowledgeGraph)
#
# Caches (exact-match + semantic) live in src/cache/ — separate concern.
#
# Backward-compat (deprecated, prefer MemoryManager):
#   set_memory_store / get_memory_store

from .context import MemoryContext, SemanticResult
from .short_term import ShortTermMemory
from .episodic import EpisodicMemory, Episode
from .semantic import SemanticMemory
from .manager import MemoryManager

# ── Backward-compat: global MemoryStore singleton ────────────────────────

_memory_store = None


def set_memory_store(store):
    global _memory_store
    _memory_store = store


def get_memory_store():
    return _memory_store


__all__ = [
    "MemoryManager",
    "MemoryContext",
    "SemanticResult",
    "ShortTermMemory",
    "SemanticMemory",
    "set_memory_store",
    "get_memory_store",
]
