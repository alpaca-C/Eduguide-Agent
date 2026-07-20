# Cache module — standalone caching layer (separate from memory)
#
# Two independent caches:
#   ExactMatchCache — SQLite exact-match (SHA256 hash). Performance acceleration.
#   SemanticCache   — Qdrant vector search. Cross-session semantic reuse.
#
# Neither is part of the memory system. Clearing caches = no functional loss.
#
# Usage:
#   from src.cache import ExactMatchCache, SemanticCache
#   cache = ExactMatchCache(store)
#   hit = cache.find_search(query)

from .exact_cache import ExactMatchCache
from .semantic_cache import SemanticCache, enable_read

__all__ = ["ExactMatchCache", "SemanticCache", "enable_read"]
