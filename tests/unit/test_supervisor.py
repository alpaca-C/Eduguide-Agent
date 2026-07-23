"""Unit tests for Supervisor — routing logic and skill list builder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.supervisor.supervisor import Supervisor, SupervisorOutput
from src.skills import SkillRegistry


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_mock_skill(name, description="desc", trigger="default", examples=None):
    skill = MagicMock()
    skill.name = name
    skill.description = description
    skill.trigger = trigger
    skill.examples = examples or []
    skill.execute = AsyncMock()
    return skill


def _make_skill_result(reply="answer", rounds=1, tool_calls=None, route="moderate"):
    result = MagicMock()
    result.reply = reply
    result.rounds = rounds
    result.tool_calls = tool_calls or []
    result.route = route
    return result


@pytest.fixture
def registry():
    return SkillRegistry()


@pytest.fixture
def config():
    c = MagicMock()
    c.llm_model_id = "test-model"
    c.llm_api_key = "test-key"
    c.llm_base_url = "http://127.0.0.1:1"
    return c


@pytest.fixture
def memory():
    m = MagicMock()
    m.recall = AsyncMock()
    # recall returns a memory context
    ctx = MagicMock()
    ctx.history_context = ""
    ctx.chat_history = []
    ctx.episodes = []
    ctx.available_docs = []
    m.recall.return_value = ctx
    m.short_term = MagicMock()
    return m


@pytest.fixture
def supervisor(registry, config, memory):
    return Supervisor(memory_manager=memory, registry=registry, config=config)


# ═══════════════════════════════════════════════════════════════════════
# SupervisorOutput
# ═══════════════════════════════════════════════════════════════════════

class TestSupervisorOutput:
    def test_defaults(self):
        out = SupervisorOutput(reply="hello")
        assert out.reply == "hello"
        assert out.session_id == ""
        assert out.rounds == 0
        assert out.tool_calls == 0
        assert out.route == ""

    def test_full(self):
        out = SupervisorOutput(
            reply="answer", session_id="s1",
            rounds=2, tool_calls=5, route="moderate",
        )
        assert out.rounds == 2
        assert out.tool_calls == 5
        assert out.route == "moderate"


# ═══════════════════════════════════════════════════════════════════════
# _build_skill_list — pure string formatting from registry
# ═══════════════════════════════════════════════════════════════════════

class TestBuildSkillList:
    def test_empty_registry_returns_empty_string(self, supervisor):
        result = supervisor._build_skill_list()
        assert result == ""

    def test_single_skill_rendered(self, supervisor, registry):
        registry.register(_make_mock_skill(
            "problem_solve", description="教材答疑流程",
            trigger="default", examples=["什么是库仑定律"],
        ))
        result = supervisor._build_skill_list()
        assert "problem_solve" in result
        assert "教材答疑流程" in result
        assert "默认可用" in result

    def test_toggle_skill_has_correct_label(self, supervisor, registry):
        registry.register(_make_mock_skill(
            "exercise_tutor", trigger="toggle", examples=["q1"],
        ))
        result = supervisor._build_skill_list()
        assert "需前端开关激活" in result

    def test_multiple_skills(self, supervisor, registry):
        registry.register(_make_mock_skill("s1", trigger="default"))
        registry.register(_make_mock_skill("s2", trigger="toggle"))
        result = supervisor._build_skill_list()
        assert "s1" in result
        assert "s2" in result

    def test_examples_rendered(self, supervisor, registry):
        registry.register(_make_mock_skill(
            "test_skill", examples=["ex1", "ex2", "ex3", "ex4"],
        ))
        result = supervisor._build_skill_list()
        # only first 3 examples shown
        assert "ex1" in result
        assert "ex2" in result
        assert "ex3" in result
        assert "ex4" not in result  # capped at 3

    def test_no_examples_no_error(self, supervisor, registry):
        registry.register(_make_mock_skill("bare", examples=[]))
        result = supervisor._build_skill_list()
        assert "bare" in result


# ═══════════════════════════════════════════════════════════════════════
# run — routing logic
# ═══════════════════════════════════════════════════════════════════════

class TestRunRouting:
    """Test the four routing priorities in Supervisor.run()."""

    @pytest.mark.asyncio
    async def test_no_skills_returns_error(self, supervisor, memory):
        from src.skills.skill_base import SkillInput
        result = await supervisor.run(SkillInput(question="hello"))
        assert "没有可用的 skill" in result.reply

    @pytest.mark.asyncio
    async def test_single_default_skill_used_directly(self, supervisor, registry):
        skill = _make_mock_skill("problem_solve", trigger="default")
        skill.execute.return_value = _make_skill_result(
            reply="answer", route="moderate",
        )
        registry.register(skill)

        from src.skills.skill_base import SkillInput
        result = await supervisor.run(SkillInput(question="hello"))

        assert result.reply == "answer"
        assert result.route == "moderate"
        skill.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_skill_name_bypasses_routing(self, supervisor, registry):
        skill = _make_mock_skill("custom_skill", trigger="default")
        skill.execute.return_value = _make_skill_result(
            reply="custom reply", route="custom",
        )
        registry.register(skill)
        registry.register(_make_mock_skill("other", trigger="default"))

        from src.skills.skill_base import SkillInput
        result = await supervisor.run(SkillInput(
            question="hello", params={"skill": "custom_skill"},
        ))

        assert result.reply == "custom reply"
        skill.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_toggle_mode_hard_routes(self, supervisor, registry):
        default_skill = _make_mock_skill("problem_solve", trigger="default")
        tutor_skill = _make_mock_skill("exercise_tutor", trigger="toggle")
        tutor_skill.execute.return_value = _make_skill_result(
            reply="引导式回答", route="tutor",
        )
        registry.register(default_skill)
        registry.register(tutor_skill)

        from src.skills.skill_base import SkillInput
        result = await supervisor.run(SkillInput(
            question="这道题怎么做", params={"tutor_mode": True},
        ))

        assert result.reply == "引导式回答"
        assert result.route == "tutor"
        tutor_skill.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_saves_reply_to_memory(self, supervisor, registry, memory):
        skill = _make_mock_skill("problem_solve", trigger="default")
        skill.execute.return_value = _make_skill_result(
            reply="saved answer", route="moderate",
        )
        registry.register(skill)

        from src.skills.skill_base import SkillInput
        await supervisor.run(SkillInput(question="hello"), session_id="sess-1")

        memory.short_term.add_message.assert_called_with(
            "sess-1", "assistant", "saved answer",
        )
