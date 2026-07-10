# Shared fixtures and mocks for integration tests
#
# All tests mock LLM calls to avoid real API costs.
# Strategy: monkeypatch agent._llm_retry() to return canned responses.

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ── Mock LLM response ──────────────────────────────────────────────

class MockLLMResponse:
    """Simulates an AIMessage response from ChatOpenAI."""

    def __init__(self, content: str):
        self.content = content


# ── Reusable canned responses ──────────────────────────────────────

ROUTER_TRIVIAL_JSON = json.dumps({
    "difficulty": "trivial",
    "reason": "问候语",
    "target_docs": [],
    "decomposition": [],
}, ensure_ascii=False)

ROUTER_MODERATE_JSON = json.dumps({
    "difficulty": "moderate",
    "reason": "单一概念查询",
    "target_docs": ["电磁学.pdf"],
    "decomposition": ["什么是库仑定律"],
}, ensure_ascii=False)

ROUTER_COMPLEX_JSON = json.dumps({
    "difficulty": "complex",
    "reason": "跨文档多概念对比",
    "target_docs": ["电磁学.pdf", "量子力学.pdf"],
    "decomposition": [
        "麦克斯韦方程组的物理意义",
        "高斯定理与库仑定律的关系",
        "静电场与磁场的对偶性",
    ],
}, ensure_ascii=False)

REWRITER_OUTPUT = "库仑定律\nCoulomb's law\n静电场 点电荷 相互作用力\n电场强度 库仑力"

PLANNER_PLAN_JSON = json.dumps({
    "sub_questions": [
        {
            "id": 1,
            "question": "什么是库仑定律",
            "keywords": ["库仑定律", "Coulomb's law"],
            "target_doc": "电磁学.pdf",
            "tool": "rag_search",
            "depends_on": [],
        },
        {
            "id": 2,
            "question": "库仑定律在电场计算中的应用",
            "keywords": ["电场计算", "库仑力叠加"],
            "target_doc": "电磁学.pdf",
            "tool": "rag_search",
            "depends_on": [1],
        },
    ],
}, ensure_ascii=False)

PLANNER_SOLVE_OUTPUT = "库仑定律是电磁学的基本定律，描述了两个点电荷之间的相互作用力：F = k·q₁q₂/r²。该定律由法国物理学家库仑于1785年通过扭秤实验得出，是静电学的基石。"

REFLECTOR_SUFFICIENT_JSON = json.dumps({
    "verdict": "SUFFICIENT",
    "missing": [],
    "suggested_queries": [],
    "issues": [],
    "reason": "回答完整准确",
}, ensure_ascii=False)

REFLECTOR_INSUFFICIENT_JSON = json.dumps({
    "verdict": "INSUFFICIENT",
    "missing": ["未提及库仑力的矢量性"],
    "suggested_queries": ["库仑力 矢量叠加", "电场 矢量 方向"],
    "issues": ["回答缺少方向信息"],
    "reason": "库仑力是矢量，回答未提及方向",
}, ensure_ascii=False)


# ── Configuration fixture ──────────────────────────────────────────

@pytest.fixture
def mock_config(monkeypatch):
    """Return a Configuration with test-safe values (no real API keys needed)."""
    from src.config import Configuration

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_ID", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://test.api.example.com")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "test/embedding")
    monkeypatch.setenv("MONITORING_ENABLED", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    return Configuration.from_env()
