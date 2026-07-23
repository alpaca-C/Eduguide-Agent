"""Unit tests for src.skills.SkillRegistry and global helper functions.

Tests the registry's core operations (register, get, filter, metadata exposure)
and the isolation principle: Supervisor sees SkillMeta, never Skill internals.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.skills import SkillRegistry, SkillMeta, register_skill, get_skill, get_default_registry


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_mock_skill(name, description="desc", trigger="default", examples=None):
    """Create a minimal mock Skill for registry tests."""
    skill = MagicMock()
    skill.name = name
    skill.description = description
    skill.trigger = trigger
    skill.examples = examples or []
    return skill


@pytest.fixture
def registry():
    """Fresh empty registry for each test."""
    return SkillRegistry()


# ═══════════════════════════════════════════════════════════════════════
# SkillMeta
# ═══════════════════════════════════════════════════════════════════════

class TestSkillMeta:
    def test_create_with_defaults(self):
        meta = SkillMeta(name="test", description="desc", trigger="default")
        assert meta.name == "test"
        assert meta.description == "desc"
        assert meta.trigger == "default"
        assert meta.examples == []

    def test_create_with_examples(self):
        meta = SkillMeta(name="test", description="desc", trigger="toggle",
                         examples=["q1", "q2"])
        assert meta.examples == ["q1", "q2"]


# ═══════════════════════════════════════════════════════════════════════
# SkillRegistry core operations
# ═══════════════════════════════════════════════════════════════════════

class TestRegistryCore:
    """Tests for register, get, __len__, __contains__, list_names."""

    def test_register_and_get(self, registry):
        skill = _make_mock_skill("problem_solve")
        registry.register(skill)
        assert registry.get("problem_solve") is skill

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_len_empty(self, registry):
        assert len(registry) == 0

    def test_len_after_register(self, registry):
        registry.register(_make_mock_skill("s1"))
        registry.register(_make_mock_skill("s2"))
        assert len(registry) == 2

    def test_contains(self, registry):
        registry.register(_make_mock_skill("s1"))
        assert "s1" in registry
        assert "s2" not in registry

    def test_list_names(self, registry):
        registry.register(_make_mock_skill("b"))
        registry.register(_make_mock_skill("a"))
        assert set(registry.list_names()) == {"a", "b"}

    def test_duplicate_name_overwrites(self, registry):
        s1 = _make_mock_skill("same", description="first")
        s2 = _make_mock_skill("same", description="second")
        registry.register(s1)
        registry.register(s2)
        assert registry.get("same") is s2
        assert len(registry) == 1

    def test_multiple_skills(self, registry):
        registry.register(_make_mock_skill("s1"))
        registry.register(_make_mock_skill("s2"))
        registry.register(_make_mock_skill("s3"))
        assert len(registry) == 3
        assert registry.get("s1") is not None
        assert registry.get("s2") is not None
        assert registry.get("s3") is not None


# ═══════════════════════════════════════════════════════════════════════
# get_all_meta — supervisor isolation
# ═══════════════════════════════════════════════════════════════════════

class TestGetAllMeta:
    """get_all_meta returns SkillMeta (not Skill) — Supervisor doesn't see internals."""

    def test_returns_skillmeta_not_skill(self, registry):
        skill = _make_mock_skill("problem_solve", description="教材答疑",
                                 trigger="default", examples=["q1"])
        registry.register(skill)

        meta_list = registry.get_all_meta()
        assert len(meta_list) == 1
        meta = meta_list[0]

        # Must be SkillMeta, not the mock Skill
        assert isinstance(meta, SkillMeta)
        assert not isinstance(meta, type(skill))

    def test_metadata_fields_match_skill(self, registry):
        skill = _make_mock_skill("exercise_tutor", description="习题讲解",
                                 trigger="toggle", examples=["q1", "q2"])
        registry.register(skill)

        meta = registry.get_all_meta()[0]
        assert meta.name == "exercise_tutor"
        assert meta.description == "习题讲解"
        assert meta.trigger == "toggle"
        assert meta.examples == ["q1", "q2"]

    def test_does_not_expose_internal_attributes(self, registry):
        """Skill could have internal attrs (agents, tools, config) — meta mustn't leak them."""
        skill = _make_mock_skill("s1")
        skill._internal_agents = ["Router", "Planner"]  # should be hidden
        skill._secret_config = {"key": "val"}
        registry.register(skill)

        meta = registry.get_all_meta()[0]
        # SkillMeta only has 4 fields
        assert not hasattr(meta, "_internal_agents")
        assert not hasattr(meta, "_secret_config")

    def test_empty_registry_returns_empty_list(self, registry):
        assert registry.get_all_meta() == []

    def test_multiple_skills_meta(self, registry):
        registry.register(_make_mock_skill("a", trigger="default"))
        registry.register(_make_mock_skill("b", trigger="toggle"))
        meta_list = registry.get_all_meta()
        assert len(meta_list) == 2
        assert {m.name for m in meta_list} == {"a", "b"}


# ═══════════════════════════════════════════════════════════════════════
# get_by_trigger — toggle routing
# ═══════════════════════════════════════════════════════════════════════

class TestGetByTrigger:
    """Supervisor uses get_by_trigger for hard routing (e.g., tutor_mode → toggle skills)."""

    def test_filters_by_trigger(self, registry):
        registry.register(_make_mock_skill("default_skill", trigger="default"))
        registry.register(_make_mock_skill("tutor_skill", trigger="toggle"))

        defaults = registry.get_by_trigger("default")
        toggles = registry.get_by_trigger("toggle")

        assert len(defaults) == 1
        assert defaults[0].name == "default_skill"
        assert len(toggles) == 1
        assert toggles[0].name == "tutor_skill"

    def test_none_matching_trigger_returns_empty(self, registry):
        registry.register(_make_mock_skill("s1", trigger="default"))
        assert registry.get_by_trigger("toggle") == []

    def test_all_same_trigger(self, registry):
        for name in ["a", "b", "c"]:
            registry.register(_make_mock_skill(name, trigger="default"))
        assert len(registry.get_by_trigger("default")) == 3

    def test_mixed_triggers(self, registry):
        registry.register(_make_mock_skill("a", trigger="default"))
        registry.register(_make_mock_skill("b", trigger="toggle"))
        registry.register(_make_mock_skill("c", trigger="default"))
        assert len(registry.get_by_trigger("default")) == 2
        assert len(registry.get_by_trigger("toggle")) == 1
        assert len(registry.get_by_trigger("auto")) == 0


# ═══════════════════════════════════════════════════════════════════════
# Global singleton & convenience functions
# ═══════════════════════════════════════════════════════════════════════

class TestGlobalRegistry:
    """Tests for the global default registry and convenience functions.

    Must clean up _default_registry between tests to avoid leakage.
    """

    @pytest.fixture(autouse=True)
    def _reset_global(self):
        """Reset the global singleton before and after each test."""
        import src.skills as skills_mod
        skills_mod._default_registry = None
        yield
        skills_mod._default_registry = None

    def test_get_default_registry_returns_same_instance(self):
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2

    def test_get_default_registry_is_SkillRegistry(self):
        assert isinstance(get_default_registry(), SkillRegistry)

    def test_register_skill_adds_to_default(self):
        skill = _make_mock_skill("global_skill")
        register_skill(skill)
        assert get_skill("global_skill") is skill

    def test_get_skill_nonexistent_returns_none(self):
        assert get_skill("never_registered") is None

    def test_multiple_register_skill(self):
        register_skill(_make_mock_skill("s1"))
        register_skill(_make_mock_skill("s2"))
        assert get_skill("s1") is not None
        assert get_skill("s2") is not None
        assert len(get_default_registry()) == 2
