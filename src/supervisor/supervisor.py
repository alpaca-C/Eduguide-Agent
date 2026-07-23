# Supervisor — LLM-powered skill routing agent.
#
# Responsibilities:
#   1. Recall memory (via MemoryManager) and inject into SkillInput
#   2. Route to the right Skill:
#      a) Toggle active (e.g. tutor_mode) → toggle skill directly (hard route)
#      b) Explicit skill name in params → use that
#      c) LLM reads all skill names + descriptions → selects best match
#      d) Only one default skill → skip LLM, use directly
#   3. Save assistant reply to short-term memory
#   4. Return SupervisorOutput
#
# Does NOT know skill-specific fields — those live in SkillInput.params.

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from ..skills.skill_base import SkillInput, SkillOutput
from ..skills import SkillRegistry, SkillMeta
from ..config import Configuration
from ..prompts.supervisor import SUPERVISOR_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class SupervisorOutput:
    """Lightweight wrapper — API layer consumes this."""
    reply: str
    session_id: str = ""
    rounds: int = 0
    tool_calls: int = 0
    route: str = ""


class Supervisor:
    """调度 Agent——读 SkillRegistry metadata → LLM 选 skill → 执行。

    Supervisor 只通过 SkillMeta 了解每个 skill 的 name / description /
    trigger / examples，不知道 skill 内部的 Agent、Tool、prompt。

    Toggle skills (tutor_mode) 硬路由保留前端控制权。
    仅一个 default skill 时跳过 LLM 节省 token。
    """

    def __init__(self, memory_manager, registry: SkillRegistry, config: Configuration):
        self._memory = memory_manager
        self._registry = registry
        self._config = config
        self._llm = None  # lazy init, only when >1 default skill

        # Startup: log all skills from registry
        for meta in registry.get_all_meta():
            logger.info("Skill: %s [%s] — %s", meta.name, meta.trigger, meta.description[:80])

    # ── LLM factory (lightweight, lazy) ─────────────────────────────

    def _get_llm(self):
        """Create a fast, cheap LLM for classification-only use."""
        if self._llm is None:
            from langchain.chat_models import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self._config.llm_model_id,
                temperature=0.0,
                max_tokens=100,
                openai_api_key=self._config.llm_api_key,
                openai_api_base=self._config.llm_base_url,
            )
        return self._llm

    # ── Skill list builder (dynamically reads registry) ─────────────

    def _build_skill_list(self) -> str:
        """Render all skills from registry for the LLM routing prompt.

        Reads SkillMeta only — never touches Skill instances or internals.
        """
        trigger_labels = {
            "default": "默认可用",
            "toggle": "需前端开关激活",
            "auto": "自动匹配",
        }
        lines = []
        for meta in self._registry.get_all_meta():
            label = trigger_labels.get(meta.trigger, meta.trigger)
            parts = [f"### {meta.name} [{label}]", meta.description]
            if meta.examples:
                parts.append(f"示例: {'; '.join(meta.examples[:3])}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    # ── LLM-based skill selection ───────────────────────────────────

    async def _llm_select(self, question: str, history_ctx: str = "") -> tuple[str, str]:
        """Ask LLM to select the best skill from registry. Returns (skill_name, reason)."""
        prompt = SUPERVISOR_PROMPT.format(
            skill_list=self._build_skill_list(),
            question=question,
            history=history_ctx or "（新对话）",
        )
        try:
            llm = self._get_llm()
            resp = await llm.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content="请选择技能。"),
            ])
            text = resp.content if hasattr(resp, "content") else str(resp)
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                name = data.get("skill", "")
                reason = data.get("reason", "")
                if name in self._registry:
                    logger.info("Supervisor LLM selected: '%s' — %s", name, reason)
                    return name, reason
            logger.warning("Supervisor LLM returned invalid skill: '%s', falling back", text[:100])
        except Exception as e:
            logger.warning("Supervisor LLM selection failed: %s, falling back", e)
        return "", ""

    # ── Main dispatch ───────────────────────────────────────────────

    async def run(self, input: SkillInput, session_id: str = "") -> SupervisorOutput:
        """Handle a user question end-to-end.

        1. Recall + inject memory
        2. Route to skill via registry
        3. Execute skill (Supervisor does NOT know skill internals)
        4. Save reply + return
        """
        # ── 0. Extract user_id, inject session_id ────────────────────
        user_id = input.params.get("user_id", "")
        input.params["session_id"] = session_id

        # ── 1. Recall memory ─────────────────────────────────────────
        history_ctx = ""
        if self._memory is not None:
            try:
                input.memory_context = await self._memory.recall(
                    input.question, session_id, user_id=user_id,
                )
                history_ctx = input.memory_context.history_context
            except Exception as e:
                logger.warning("Memory recall failed: %s", e)

        # ── 2. Route to skill ────────────────────────────────────────
        # Priority:
        #   a) Explicit skill name in params
        #   b) Toggle active (tutor_mode) → hard route via registry
        #   c) Only one default skill → skip LLM
        #   d) Multiple default skills → LLM reads SkillMeta and selects
        skill_name = input.params.get("skill", "")
        skill = None
        reason = ""

        if skill_name:
            skill = self._registry.get(skill_name)
            reason = "显式指定"
        else:
            toggle = input.params.get("tutor_mode", False)
            if toggle:
                toggles = self._registry.get_by_trigger("toggle")
                if toggles:
                    skill = toggles[0]
                    reason = "前端开关激活"

            if skill is None:
                defaults = self._registry.get_by_trigger("default")
                if len(defaults) == 1:
                    skill = defaults[0]
                    reason = "唯一默认技能"
                elif len(defaults) > 1:
                    chosen, llm_reason = await self._llm_select(
                        input.question, history_ctx,
                    )
                    if chosen:
                        skill = self._registry.get(chosen)
                        reason = f"LLM: {llm_reason}"
                    else:
                        skill = defaults[0]
                        reason = "LLM 失败，回退默认"

        if skill is None:
            return SupervisorOutput(reply="没有可用的 skill 处理此请求")

        # ── Visible routing summary ──────────────────────────────────
        # Build skill list for log
        skill_list = ", ".join(
            f"{m.name}[{m.trigger}]" for m in self._registry.get_all_meta()
        )
        # Memory stats
        mem_ctx = getattr(input, 'memory_context', None)
        ep_count = len(getattr(mem_ctx, 'episodes', []) or [])
        msg_count = len(getattr(mem_ctx, 'chat_history', []) or [])
        doc_count = len(getattr(mem_ctx, 'available_docs', []) or [])

        print(
            f"[Supervisor] \"{input.question[:60]}\"\n"
            f"  memory: {msg_count} msgs | {ep_count} episodes | {doc_count} docs\n"
            f"  skills: {skill_list}\n"
            f"  → {skill.name} ({reason})"
        )
        logger.info(
            "Supervisor → %s (%s) | skills=%s | memory=%dmsgs/%dep/%ddocs",
            skill.name, reason, skill_list, msg_count, ep_count, doc_count,
        )
        input.params["tutor_mode"] = (skill.trigger == "toggle")

        # ── 3. Execute skill (Supervisor has NO knowledge of internals) ──
        result = await skill.execute(input)

        # ── 4. Save assistant reply ──────────────────────────────────
        if self._memory is not None and session_id:
            try:
                self._memory.short_term.add_message(session_id, "assistant", result.reply)
            except Exception as e:
                logger.warning("Failed to save assistant reply: %s", e)

        # ── 5. Return ────────────────────────────────────────────────
        return SupervisorOutput(
            reply=result.reply,
            session_id=session_id,
            rounds=result.rounds,
            tool_calls=len(result.tool_calls),
            route=result.route,
        )
