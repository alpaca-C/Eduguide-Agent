# Configuration 鈥?Document QA System (local vector store)

from __future__ import annotations

import os
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Document QA System configuration."""

    llm_model_id: str = Field(default="deepseek-chat")
    llm_api_key: str = Field(default="sk-placeholder")
    llm_base_url: str = Field(default="https://api.deepseek.com")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=6000)
    llm_chapter_detect_model: str = Field(default="")
    llm_chapter_review_model: str = Field(default="")
    chapter_detect_timeout: int = Field(default=60)
    # Vision model for image-based PDF chapter detection (VLM fallback)
    # 百炼 DashScope example: "qwen-vl-plus-latest" / "qwen-vl-max-latest"
    llm_vision_model: str = Field(default="")
    llm_vision_base_url: str = Field(default="")
    llm_vision_api_key: str = Field(default="")

    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)
    extract_batch_size: int = Field(default=200)
    extract_chunk_max_chars: int = Field(default=300)
    extract_max_concepts_per_batch: int = Field(default=50)
    extract_concurrency: int = Field(default=3)       # 知识提取并发 batch 数
    chapter_detect_concurrency: int = Field(default=3)  # 章节检测并发文件数

    memory_db_path: str = Field(default="")

    @classmethod
    def from_env(cls, **overrides) -> "Configuration":
        env = os.environ
        data: dict = {}
        for name in ["llm_model_id", "llm_api_key", "llm_base_url", "llm_temperature",
                     "llm_max_tokens", "llm_chapter_detect_model", "llm_chapter_review_model",
                     "chapter_detect_timeout",
                     "llm_vision_model", "llm_vision_base_url", "llm_vision_api_key",
                     "extract_max_concepts_per_batch", "memory_db_path",
                     "extract_concurrency", "chapter_detect_concurrency"]:
            value = env.get(name.upper())
            if value is not None:
                if name == "llm_max_tokens":
                    data[name] = int(value)
                elif name == "llm_temperature":
                    data[name] = float(value)
                elif name in ("chunk_size", "chunk_overlap", "extract_batch_size", "extract_chunk_max_chars", "extract_max_concepts_per_batch", "extract_concurrency", "chapter_detect_concurrency"):
                    data[name] = int(value)
                else:
                    data[name] = value
        data.update(overrides)
        return cls(**data)
