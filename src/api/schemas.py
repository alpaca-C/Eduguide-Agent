"""Shared Pydantic request/response models for all API routers."""

from __future__ import annotations

from pydantic import BaseModel


class ChapterDetectRequest(BaseModel):
    filepaths: list[str]


class ProcessRequest(BaseModel):
    filepaths: list[str]
    selected_chapters: list[str] = []


class SaveChaptersRequest(BaseModel):
    filename: str
    chapters: list[dict] = []


class ChatRequest(BaseModel):
    question: str
    session_id: str = ""
    doc_filter: list[str] = []
