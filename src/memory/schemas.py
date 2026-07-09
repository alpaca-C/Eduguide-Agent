# Data models for Document QA System

from __future__ import annotations

from typing import TypedDict


class PipelineState(TypedDict):
    """Document processing pipeline state."""
    filepaths: list[str]
    documents: list
    chunks: list
    message: str
    concepts_extracted: int
    relations_extracted: int
    ready: bool
    error: str
