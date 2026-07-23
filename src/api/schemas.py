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
    user_id: str = ""          # 用户标识，用于跨对话情景记忆召回
    doc_filter: list[str] = []
    tutor_mode: bool = False   # 是否启用"举一反三"引导式习题讲解
