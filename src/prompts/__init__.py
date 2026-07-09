"""Prompt templates for all agents — centralized for easy tuning."""

# Legacy exports (from old single-file qa prompts)
from .qa import (  # type: ignore[import-untyped]
    SYSTEM_PROMPT,
    SIMPLE_CLASSIFY_PROMPT,
    ANSWERER_THINK_RAG_PROMPT,
    ANSWERER_THINK_WEB_PROMPT,
    ANSWERER_SYNTHESIS_PROMPT,
    REVIEWER_PROMPT,
)

from .extractor import EXTRACTOR_SYSTEM_PROMPT
from .chapterizer import CHAPTER_DETECTOR_PROMPT, CHAPTER_REVIEWER_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "SIMPLE_CLASSIFY_PROMPT",
    "ANSWERER_THINK_RAG_PROMPT",
    "ANSWERER_THINK_WEB_PROMPT",
    "ANSWERER_SYNTHESIS_PROMPT",
    "REVIEWER_PROMPT",
    "EXTRACTOR_SYSTEM_PROMPT",
    "CHAPTER_DETECTOR_PROMPT",
    "CHAPTER_REVIEWER_PROMPT",
]
