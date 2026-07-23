"""Unit tests for ExerciseTutorSkill — properties and prompt structure."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.skills.exercise_tutor import ExerciseTutorSkill, EXERCISE_TUTOR_PROMPT


@pytest.fixture
def skill():
    config = MagicMock()
    return ExerciseTutorSkill(config)


class TestExerciseTutorProperties:
    def test_name(self, skill):
        assert skill.name == "exercise_tutor"

    def test_trigger_is_toggle(self, skill):
        assert skill.trigger == "toggle"

    def test_description_not_empty(self, skill):
        assert len(skill.description) > 50
        assert "苏格拉底" in skill.description or "引导" in skill.description

    def test_examples_not_empty(self, skill):
        examples = skill.examples
        assert len(examples) >= 2
        assert any("公式" in e or "思路" in e or "卡" in e for e in examples)

    def test_system_prompt_is_the_constant(self, skill):
        assert skill.system_prompt is EXERCISE_TUTOR_PROMPT

    def test_system_prompt_contains_core_principles(self, skill):
        prompt = skill.system_prompt
        assert "绝不直接给出答案" in prompt
        assert "场景 A" in prompt
        assert "场景 B" in prompt
        assert "场景 C" in prompt
        assert "{observations}" in prompt
        assert "{question}" in prompt
        assert "{chat_history}" in prompt
