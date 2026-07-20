# Semantic Cache — Qdrant-based semantic similarity cache
#
# Separate from the memory module. Provides cross-session semantic caching
# for search results and planner outputs using Qdrant vector search.
#
# Currently: write path active, read path disabled (cross-region latency).
# To enable: call enable_read(), or deploy to a regional vector DB (e.g. Zilliz Cloud).
#
# Usage:
#   cache = SemanticCache(qdrant_url, qdrant_api_key)
#   cache.store_search(query, results, answer)
#   hit = cache.find_search(query)        # None if read disabled
#   cache.store_plan(topic, plan)

from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports
_qdrant_client = None
_embedding_model = None
_read_enabled = False  # Toggle to True when deploying to a regional DB


def _get_qdrant_client(url: str, api_key: str):
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=url, api_key=api_key, timeout=20)
    return _qdrant_client


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _embed(text: str) -> list[float]:
    """Convert text to a 384-dim normalized embedding vector."""
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def _hash_query(query: str) -> str:
    """Deterministic query hash for point IDs."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def enable_read():
    """Enable semantic cache reads (call after deploying to a low-latency region)."""
    global _read_enabled
    _read_enabled = True
    logger.info("SemanticCache: read path enabled")


class SemanticCache:
    """Qdrant 语义相似度缓存 — 跨会话复用搜索结果和规划。

    Write path: always active (async upsert to Qdrant, failures logged not raised).
    Read path:  disabled by default (cross-region latency). Call enable_read().
    """

    SEARCH_COLLECTION = "search_cache"
    PLAN_COLLECTION = "plan_cache"
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2

    def __init__(self, qdrant_url: str = "", qdrant_api_key: str = ""):
        self._url = qdrant_url
        self._api_key = qdrant_api_key
        self._active = bool(qdrant_url and qdrant_api_key)

        if self._active:
            self._init_collections()

    # ── Init ──────────────────────────────────────────────────────────

    def _init_collections(self):
        try:
            logger.info("SemanticCache: connecting to Qdrant...")
            client = _get_qdrant_client(self._url, self._api_key)
            from qdrant_client.models import Distance, VectorParams
            for name in [self.SEARCH_COLLECTION, self.PLAN_COLLECTION]:
                try:
                    client.get_collection(name)
                    logger.info("SemanticCache: collection '%s' exists", name)
                except Exception:
                    client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
                    )
                    logger.info("SemanticCache: created collection '%s'", name)
            logger.info("SemanticCache: connected")
        except Exception as e:
            logger.warning("SemanticCache: init failed, cache disabled: %s", e)
            self._active = False

    # ── Search Cache ──────────────────────────────────────────────────

    def store_search(self, query: str, results: list[dict], answer: str = ""):
        """Write a search result to Qdrant for semantic reuse."""
        if not self._active:
            return
        try:
            client = _get_qdrant_client(self._url, self._api_key)
            from qdrant_client.models import PointStruct
            vector = _embed(query)
            qhash = _hash_query(query)
            client.upsert(
                collection_name=self.SEARCH_COLLECTION,
                points=[PointStruct(
                    id=qhash, vector=vector,
                    payload={"query": query, "results": results, "answer": answer},
                )],
            )
        except Exception as e:
            logger.warning("SemanticCache: search upsert failed: %s", e)

    def find_search(self, query: str, threshold: float = 0.85) -> Optional[tuple[list, str]]:
        """Semantically search for a cached search result.

        Returns (results, answer) or None. Blocking — wrap in run_in_executor.
        """
        if not self._active or not _read_enabled:
            return None
        try:
            client = _get_qdrant_client(self._url, self._api_key)
            vector = _embed(query)
            results = client.search(
                collection_name=self.SEARCH_COLLECTION,
                query_vector=vector,
                limit=1,
                score_threshold=threshold,
            )
            if results:
                payload = results[0].payload or {}
                score = results[0].score
                logger.info("SemanticCache: search HIT score=%.3f for '%s'", score, query[:50])
                return payload.get("results", []), payload.get("answer", "")
        except Exception as e:
            logger.warning("SemanticCache: search lookup failed: %s", e)
        return None

    # ── Plan Cache ────────────────────────────────────────────────────

    def store_plan(self, topic: str, plan: list[dict], background: str = ""):
        """Write a planner output to Qdrant for semantic reuse."""
        if not self._active:
            return
        try:
            client = _get_qdrant_client(self._url, self._api_key)
            from qdrant_client.models import PointStruct
            vector = _embed(topic)
            qhash = _hash_query(topic)
            client.upsert(
                collection_name=self.PLAN_COLLECTION,
                points=[PointStruct(
                    id=qhash, vector=vector,
                    payload={"topic": topic, "plan": plan, "background": background},
                )],
            )
        except Exception as e:
            logger.warning("SemanticCache: plan upsert failed: %s", e)

    def find_plan(self, topic: str, threshold: float = 0.85) -> Optional[tuple[list[dict], str]]:
        """Semantically search for a cached plan. Returns (plan_items, background) or None."""
        if not self._active or not _read_enabled:
            return None
        try:
            client = _get_qdrant_client(self._url, self._api_key)
            vector = _embed(topic)
            results = client.search(
                collection_name=self.PLAN_COLLECTION,
                query_vector=vector,
                limit=1,
                score_threshold=threshold,
            )
            if results:
                payload = results[0].payload or {}
                score = results[0].score
                logger.info("SemanticCache: plan HIT score=%.3f for '%s'", score, topic[:50])
                return payload.get("plan", []), payload.get("background", "")
        except Exception as e:
            logger.warning("SemanticCache: plan lookup failed: %s", e)
        return None
