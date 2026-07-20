# Compressor — Stage 4 of GSSC: smart compression when token budget exceeded.
#
# Compression order (least important first):
#   context → evidence → state → task → role_policies (never compressed below truncation)
#
# Strategy per section:
#   1. Truncate: keep top-N fragments (by score), drop the rest
#   2. Summarize: extract key sentences (rule-based, zero-cost)

from __future__ import annotations

import logging
import re

from .schema import StructuredPrompt, TEMPLATE_SECTIONS

logger = logging.getLogger(__name__)

# Compression order: first to compress first
COMPRESS_ORDER = ["context", "evidence", "state", "task", "role_policies"]

# Per-section max chars before triggering compression within that section
SECTION_CHAR_LIMITS = {
    "context": 800,
    "evidence": 1500,
    "state": 600,
    "task": 600,
    "role_policies": 400,
    "output_format": 200,
}


class Compressor:
    """Token budget的最后防线。先截断，仍超限则规则摘要。"""

    def __init__(self, hard_limit: int = 4000):
        self._limit = hard_limit
        # chars ≈ tokens * 2
        self._char_limit = hard_limit * 2

    def compress(self, prompt: StructuredPrompt) -> StructuredPrompt:
        """Compress if total chars exceed limit. Returns (possibly same) prompt."""
        total = prompt.total_chars()
        if total <= self._char_limit:
            return prompt

        logger.info("Compressor: %d chars exceeds limit %d, compressing...",
                     total, self._char_limit)

        # Phase 1: Truncate each section to its char limit
        for name in COMPRESS_ORDER:
            if name not in prompt.sections:
                continue
            text = prompt.sections[name]
            limit = SECTION_CHAR_LIMITS.get(name, 600)
            if len(text) > limit:
                prompt.sections[name] = self._truncate(text, limit)
                logger.debug("Compressor: truncated '%s' (%d → %d chars)",
                             name, len(text), len(prompt.sections[name]))
            if prompt.total_chars() <= self._char_limit:
                prompt.compressed = True
                return prompt

        # Phase 2: Summarize each section (keep key sentences only)
        for name in COMPRESS_ORDER:
            if name not in prompt.sections:
                continue
            text = prompt.sections[name]
            if len(text) > SECTION_CHAR_LIMITS.get(name, 600) // 2:
                prompt.sections[name] = self._summarize(text)
            if prompt.total_chars() <= self._char_limit:
                prompt.compressed = True
                return prompt

        # Phase 3: Last resort — hard truncate the whole prompt
        prompt = self._hard_truncate(prompt)

        prompt.compressed = True
        logger.info("Compressor: done, final %d chars", prompt.total_chars())
        return prompt

    # ── Truncation ──────────────────────────────────────────────────────

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Keep first max_chars, break at a natural boundary."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # Try to break at last newline
        last_nl = truncated.rfind("\n")
        if last_nl > max_chars // 2:
            truncated = truncated[:last_nl]
        return truncated + "\n...（截断）"

    # ── Summarization (rule-based, zero-cost) ────────────────────────────

    @staticmethod
    def _summarize(text: str) -> str:
        """Extract key sentences: first sentence of each paragraph + any
        sentence containing key indicator words.
        """
        paragraphs = text.split("\n\n")
        key_sentences = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            sentences = re.split(r'[。！？.!?]', para)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                continue
            # Always keep the first sentence of each paragraph
            key_sentences.append(sentences[0] + "。")
            # Keep sentences with key indicators
            for s in sentences[1:]:
                if any(kw in s for kw in [
                    "问题", "关键", "重要", "核心", "定义", "定理", "结论",
                    "教训", "反馈", "缺失", "建议", "必须", "不能",
                ]):
                    key_sentences.append(s + "。")

        result = "；".join(key_sentences[:10])
        if not result:
            result = text[:200] + "..."
        return result + "\n（摘要）"

    # ── Hard truncation (last resort) ────────────────────────────────────

    def _hard_truncate(self, prompt: StructuredPrompt) -> StructuredPrompt:
        """Brute-force: progressively shrink sections until under limit."""
        for name in COMPRESS_ORDER:
            if name not in prompt.sections:
                continue
            prompt.sections[name] = self._truncate(prompt.sections[name], 150)
            if prompt.total_chars() <= self._char_limit:
                break
        return prompt
