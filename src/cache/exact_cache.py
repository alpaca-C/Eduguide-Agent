# ExactMatchCache — SQLite exact-match search cache + plan cache
#
# Performance acceleration layer. NOT part of the memory system.
# Cleared cache = no functional loss, only performance impact.
#
# Usage:
#   cache = ExactMatchCache(store)          # store = MemoryStore
#   hit = cache.find_search("高斯定理")      # (results, answer) or None
#   cache.store_search(query, results, ans)
#   hit = cache.find_plan("数据库范式")      # (plan, background) or None
#   cache.store_plan(topic, plan, bg)

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ExactMatchCache:
    """SQLite 精确匹配缓存 — 搜索缓存 + 规划缓存。

    Thin wrapper over MemoryStore's cache methods. Exists as a separate
    concern from the memory system (short-term / semantic).
    """

    def __init__(self, store):
        """Wrap a MemoryStore for its cache methods.

        Args:
            store: MemoryStore from src.memory.store.
        """
        self._store = store

    # ── Search cache ────────────────────────────────────────────────

    def store_search(self, query: str, results: list, answer: str = ""):
        """Cache search result by exact query hash."""
        self._store.cache_search_result(query, results, answer)

    def find_search(self, query: str) -> Optional[tuple[list, str]]:
        """Look up cached search result. Returns (results, answer) or None."""
        return self._store.get_cached_search(query)

    # ── Plan cache ──────────────────────────────────────────────────

    def store_plan(self, topic: str, plan: list[dict], background: str = ""):
        """Cache planner output by exact topic hash."""
        self._store.cache_plan(topic, plan, background)

    def find_plan(self, topic: str) -> Optional[tuple[list[dict], str]]:
        """Look up cached plan. Returns (plan_items, background) or None."""
        return self._store.get_cached_plan(topic)
