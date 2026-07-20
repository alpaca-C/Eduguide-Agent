# Structurer — Stage 3 of GSSC: organize scored fragments into the fixed template.
#
# Maps fragment.source → template section, preserving score order within each section.

from __future__ import annotations

import logging

from .schema import ScoredFragment, StructuredPrompt, TEMPLATE_SECTIONS

logger = logging.getLogger(__name__)

# fragment source → template section name
SOURCE_TO_SECTION: dict[str, str] = {
    "system_policy":   "role_policies",
    "tool":            "role_policies",
    "user_question":   "task",
    "conversation":    "context",
    "search_result":   "evidence",
    "kg_concept":      "evidence",
    "episodic":        "state",
}


class Structurer:
    """Organize scored fragments into the GSSC template.

    Within each section, fragments are kept in score-descending order.
    Empty sections get "（无）" placeholder.
    """

    # ── Public API ──────────────────────────────────────────────────────

    def structure(self, selected: list[ScoredFragment]) -> StructuredPrompt:
        """Build a StructuredPrompt from scored fragments."""
        # Group by section
        buckets: dict[str, list[ScoredFragment]] = {name: [] for name in TEMPLATE_SECTIONS}
        for s in selected:
            section = SOURCE_TO_SECTION.get(s.fragment.source, "context")
            buckets[section].append(s)

        # Build each section (fragments already sorted by total_score from Selector)
        sections: dict[str, str] = {}
        for name in TEMPLATE_SECTIONS:
            items = buckets.get(name, [])
            if not items:
                sections[name] = "（无）"
            else:
                sections[name] = self._render_section(items)

        token_est = sum(len(v) // 2 for v in sections.values())

        return StructuredPrompt(
            sections=sections,
            token_estimate=token_est,
        )

    # ── Rendering ───────────────────────────────────────────────────────

    @staticmethod
    def _render_section(items: list[ScoredFragment]) -> str:
        """Concatenate fragment contents within a section.

        Fragments within a section are separated by double newline.
        """
        parts = []
        for item in items:
            text = item.fragment.content.strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)
