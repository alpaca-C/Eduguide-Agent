# Selector — Stage 2 of GSSC: score, filter, and trim fragments.
#
# Scoring: relevance(question, content) * w_rel + recency(metadata) * w_rec
# Filter:  drop below min_score threshold
# Budget:  stop adding when estimated tokens exceed budget

from __future__ import annotations

import logging
import time

from .schema import Fragment, ScoredFragment

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 2 chars for Chinese/English mixed text
CHARS_PER_TOKEN = 2


class Selector:
    """加权评分 + 去噪过滤 + Token 预算裁剪。"""

    def __init__(
        self,
        token_budget: int = 3000,
        relevance_weight: float = 0.6,
        recency_weight: float = 0.4,
        min_score: float = 0.10,
    ):
        self._budget = token_budget
        self._w_rel = relevance_weight
        self._w_rec = recency_weight
        self._min_score = min_score

    def select(self, fragments: list[Fragment], question: str) -> list[ScoredFragment]:
        """评分 → 排序 → 过滤 → 预算裁剪。"""
        if not fragments:
            return []

        now = time.time()

        # 1) Score each fragment
        scored: list[ScoredFragment] = []
        for f in fragments:
            rel = self._score_relevance(f, question)
            rec = self._score_recency(f, now)
            # Priority boost: each priority level adds 0.05 to total
            total = rel * self._w_rel + rec * self._w_rec + f.priority * 0.05

            if total >= self._min_score:
                scored.append(ScoredFragment(
                    fragment=f, relevance=rel, recency=rec, total_score=total,
                ))

        # 2) Sort by total_score descending
        scored.sort(key=lambda s: s.total_score, reverse=True)

        # 3) Token budget: greedy inclusion
        selected: list[ScoredFragment] = []
        tokens_used = 0
        for s in scored:
            est = max(1, len(s.fragment.content) // CHARS_PER_TOKEN)
            if tokens_used + est > self._budget:
                continue
            selected.append(s)
            tokens_used += est

        logger.info(
            "Selector: %d fragments → %d scored → %d selected (budget=%d, used=%d)",
            len(fragments), len(scored), len(selected), self._budget, tokens_used,
        )
        return selected

    # ── Scoring helpers ─────────────────────────────────────────────────

    @staticmethod
    def _score_relevance(f: Fragment, question: str) -> float:
        """Simple relevance: keyword overlap between question and content.

        For production, replace with embedding cosine similarity or a cheap
        cross-encoder. Current approach is zero-cost and zero-latency.
        """
        if not question:
            return 0.5  # neutral

        q_chars = set(question)
        c_chars = set(f.content[:500])  # first 500 chars only
        if not q_chars:
            return 0.5

        overlap = len(q_chars & c_chars) / len(q_chars)
        # Boost for source types that are inherently relevant
        source_boost = {
            "user_question": 0.3,
            "system_policy": 0.2,
            "conversation": 0.1,
            "episodic": 0.05,
        }.get(f.source, 0.0)

        return min(1.0, overlap + source_boost)

    @staticmethod
    def _score_recency(f: Fragment, now: float) -> float:
        """Score based on timestamp metadata. 1.0 = just now, decays over 24h.

        If no timestamp, default to 0.5 (neutral).
        """
        ts = f.metadata.get("timestamp")
        if ts is None:
            return 0.5

        age_hours = (now - ts) / 3600.0
        if age_hours <= 0:
            return 1.0

        # Exponential decay: half-life of 12 hours
        decay = 2.0 ** (-age_hours / 12.0)
        return max(0.1, decay)
