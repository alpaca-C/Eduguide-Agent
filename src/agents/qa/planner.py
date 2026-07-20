# Planner — decompose complex questions + summarize results

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from ..base import BaseAgent, AgentInput, AgentOutput
from ...config import Configuration
from ...tools.rag_search import get_doc_names
from ...prompts.qa.router import SYSTEM_PROMPT
from ...prompts.qa.planner import PLAN_PROMPT, SOLVE_PROMPT
from ...context_builder import PlannerContext, PromptBuilder
from .plan_schema import PlanOutput

logger = logging.getLogger(__name__)

# Max retries for JSON parse + Pydantic validation failures
MAX_PLAN_PARSE_RETRIES = 1


class Planner(BaseAgent):
    """Decomposes complex questions into sub-questions, then synthesizes results.

    Works with Executor (runs sub-question searches) and Reflector (reviews output).
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._llm = self._make_llm()

    async def run(self, input: AgentInput) -> AgentOutput:
        question = input.metadata.get("question", "")
        try:
            plan = await self.plan(question, feedback="")
            return AgentOutput(success=True, metadata={"plan": plan})
        except Exception as e:
            return AgentOutput(success=False, error=str(e))

    async def plan(self, question: str, feedback: str = "", history_ctx: str = "",
                   seed_decomposition: list[str] | None = None,
                   planner_ctx: PlannerContext | None = None) -> list[dict]:
        """Decompose question into sub-questions. Incorporates reviewer feedback + history.

        Args:
            question: The student's original question.
            feedback: Reviewer feedback from a previous round.
            history_ctx: Formatted conversation history context (legacy).
            seed_decomposition: Optional initial sub-questions from Router.
            planner_ctx: Typed PlannerContext (preferred over history_ctx).
        """
        doc_names = self._get_doc_list()
        doc_list = f"共{len(doc_names)}本：{', '.join(doc_names)}" if doc_names else "（暂无已上传教材）"

        # Build seed section from Router's decomposition
        seed_section = ""
        if seed_decomposition:
            seed_lines = [f"  {i+1}. {s}" for i, s in enumerate(seed_decomposition)]
            seed_section = (
                "**初步分解（来自前置分析，可直接使用或改进）：**\n"
                + "\n".join(seed_lines)
                + "\n\n请审查以上分解是否合理，补充或调整子问题后输出最终分解。\n"
            )

        feedback_section = ""
        if feedback:
            feedback_section = (
                f"**上一轮审核反馈：**\n{feedback}\n"
                f"请根据反馈补充遗漏的子问题。\n"
            )

        system_prompt = PLAN_PROMPT.format(
            question=question, doc_list=doc_list,
            tool_list=self._build_tool_list(),
            seed_section=seed_section,
            feedback_section=feedback_section,
        )

        if planner_ctx is not None:
            messages = PromptBuilder.build(
                system=system_prompt,
                context=planner_ctx,
                user="请分解以上问题。",
            )
        elif history_ctx:
            messages = [
                SystemMessage(content=system_prompt + f"\n\n{history_ctx}"),
                HumanMessage(content="请分解以上问题。"),
            ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="请分解以上问题。"),
            ]

        resp = await self._llm_retry(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return await self._parse_and_validate(text, system_prompt)

    async def solve(self, question: str, observations: str, history_ctx: str = "",
                    solver_ctx=None, reasoning_feedback: str = "") -> str:
        """Synthesize answer from sub-question results.

        Args:
            solver_ctx: Typed SolverContext (preferred over raw history_ctx).
            reasoning_feedback: Optional logic review feedback from Reflector.
                Injected into the prompt to guide reasoning fixes without re-searching.
        """
        # Guard: refuse to synthesize if observations are empty / all errors
        if not observations or not observations.strip():
            return "抱歉，教材中未找到与您问题相关的资料。建议尝试更具体的术语或检查教材是否已上传处理。"

        # Check if ALL results are empty/error indicators
        lines = [l for l in observations.split("\n") if l.strip()]
        meaningful = [
            l for l in lines
            if "未找到" not in l
            and "NOT_CONFIGURED" not in l
            and "未初始化" not in l
            and l.strip()
        ]
        if not meaningful:
            return "抱歉，所有检索结果均为空或失败。请确认相关教材已上传并处理完成。"

        # Build reasoning feedback section for injection into prompt
        reasoning_section = ""
        if reasoning_feedback:
            reasoning_section = (
                f"**上一轮逻辑审核反馈（请据此修正推理，但不要编造资料中没有的内容）：**\n"
                f"{reasoning_feedback}\n"
            )

        system_prompt = SYSTEM_PROMPT + "\n\n" + SOLVE_PROMPT.format(
            observations=observations, question=question,
            reasoning_feedback=reasoning_section,
        )

        if solver_ctx is not None:
            from ...context_builder import SolverContext
            messages = PromptBuilder.build(
                system=system_prompt,
                context=solver_ctx,
                user="请综合以上搜索结果回答学生原始问题。如果资料不足以回答，请明确告知学生而不是编造内容。",
            )
        elif history_ctx:
            messages = [
                SystemMessage(content=system_prompt + f"\n\n{history_ctx}"),
                HumanMessage(content="请综合以上搜索结果回答学生原始问题。如果资料不足以回答，请明确告知学生而不是编造内容。"),
            ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="请综合以上搜索结果回答学生原始问题。如果资料不足以回答，请明确告知学生而不是编造内容。"),
            ]

        resp = await self._llm_retry(messages)
        return resp.content if hasattr(resp, "content") else str(resp)

    @staticmethod
    def _get_doc_list() -> list[str]:
        try:
            return get_doc_names()
        except Exception:
            return []

    @staticmethod
    def _build_tool_list() -> str:
        """Build a formatted tool whitelist for injection into the Planner prompt."""
        try:
            from ...tools import get_tool_registry
            tools = get_tool_registry()
            if not tools:
                return "- **rag_search**: 搜索本地已上传的教材资料"
            return "\n".join(
                f"- **{name}**: {info['description']}"
                for name, info in tools.items()
            )
        except Exception:
            return "- **rag_search**: 搜索本地已上传的教材资料"

    async def _parse_and_validate(self, text: str, original_prompt: str) -> list[dict]:
        """
        Parse Planner LLM output with Pydantic validation + retry.

        Flow:
          1. Parse JSON (regex → json.loads)
          2. Validate with Pydantic PlanOutput schema
          3. On failure → send error back to LLM for correction (1 retry)
          4. On retry failure → return partially valid sub_questions (not empty)

        This replaces the old _parse_json() that silently returned {} on any failure.
        """
        # ── Step 1: Extract JSON ────────────────────────────────
        parsed = self._extract_json(text)
        if parsed is None:
            retry_text = await self._retry_with_feedback(
                text, original_prompt, "JSON 格式错误：无法解析为合法 JSON。请重新输出。"
            )
            if retry_text is None:
                logger.warning("Planner: JSON parse failed after retry, returning []")
                return []
            parsed = self._extract_json(retry_text)
            if parsed is None:
                logger.warning("Planner: retry text also not valid JSON, returning []")
                return []

        # ── Step 2: Pydantic validation ─────────────────────────
        try:
            plan = PlanOutput(**parsed)
            valid = plan.filter_valid()
            if valid:
                logger.info("Planner: %d valid sub-questions (Pydantic)", len(valid))
            else:
                logger.warning("Planner: 0 valid sub-questions after Pydantic filter")
            return valid
        except (ValidationError, TypeError) as ve:
            error_summary = self._format_validation_errors(ve) if isinstance(ve, ValidationError) else str(ve)
            logger.warning(
                "Planner: Pydantic validation failed: %s",
                error_summary[:200],
            )

            # ── Step 3: Retry with validation feedback ─────────
            retry_text = await self._retry_with_feedback(
                text, original_prompt,
                f"JSON 格式正确但内容不符合要求：\n{error_summary}\n请修正后重新输出。",
            )
            if retry_text is None:
                salvage = self._salvage_partial(parsed)
                logger.warning("Planner: validation retry failed, salvaged %d items", len(salvage))
                return salvage

            retry_parsed = self._extract_json(retry_text)
            if retry_parsed is None:
                return self._salvage_partial(parsed)

            try:
                plan = PlanOutput(**retry_parsed)
                valid = plan.filter_valid()
                logger.info("Planner: retry succeeded, %d valid sub-questions", len(valid))
                return valid
            except (ValidationError, TypeError) as e:
                logger.warning("Planner: retry validation also failed: %s", e)
                return self._salvage_partial(parsed)

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract JSON from LLM output using regex, with nested-brace awareness."""
        # Try regex first (fast path)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: brace counting for nested structures
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    async def _retry_with_feedback(
        self, failed_text: str, original_prompt: str, error_msg: str,
    ) -> str | None:
        """Send parsing/validation error back to LLM for correction."""
        for attempt in range(MAX_PLAN_PARSE_RETRIES + 1):
            try:
                logger.info(
                    "Planner: retry attempt %d/%d — %s",
                    attempt + 1, MAX_PLAN_PARSE_RETRIES + 1, error_msg[:80],
                )
                resp = await self._llm_retry(
                    [
                        SystemMessage(content=original_prompt),
                        HumanMessage(content=f"你之前的输出：\n{failed_text[:2000]}\n\n{error_msg}"),
                    ],
                    max_retries=0,  # no nested retry — one shot per attempt
                )
                return resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                logger.warning("Planner: retry LLM call failed: %s", e)
                if attempt >= MAX_PLAN_PARSE_RETRIES:
                    return None

        return None

    @staticmethod
    def _format_validation_errors(ve: ValidationError) -> str:
        """Format Pydantic ValidationError into human-readable Chinese feedback."""
        lines = []
        for err in ve.errors()[:10]:  # cap at 10 errors
            loc = " → ".join(str(x) for x in err["loc"])
            msg = err["msg"]
            lines.append(f"  - 字段 [{loc}]: {msg}")
        return "\n".join(lines)

    @staticmethod
    def _salvage_partial(parsed: dict | None) -> list[dict]:
        """
        Extract whatever valid sub_questions we can from partial data.

        When Pydantic validation fails, we don't want to throw away everything —
        some sub_questions may be correctly structured.
        """
        if not parsed:
            return []
        raw_list = parsed.get("sub_questions", [])
        if not isinstance(raw_list, list):
            return []
        salvage = []
        for i, sq in enumerate(raw_list):
            if not isinstance(sq, dict):
                continue
            # Build a minimal valid sub_question with safe defaults
            salvage.append({
                "id": sq.get("id", i + 1),
                "question": str(sq.get("question", "")),
                "keywords": sq.get("keywords", []) if isinstance(sq.get("keywords"), list) else [],
                "target_doc": str(sq.get("target_doc", "")),
                "tool": sq.get("tool", "rag_search") if sq.get("tool") in {"rag_search", "web_search", "mineru_ocr"} else "rag_search",
                "depends_on": sq.get("depends_on", []) if isinstance(sq.get("depends_on"), list) else [],
            })
        return salvage
