# PromptBuilder — structured message construction.
#
# Takes system prompt + typed context + user input and builds
# LangChain message sequences. Context is injected as a clearly
# marked structured block, NOT string-concatenated to the system prompt.
#
# Usage:
#   messages = PromptBuilder.build(
#       system=ROUTER_SYSTEM,
#       context=router_context,     # RouterContext dataclass
#       user="请分析以上问题的难度和类型。",
#   )
#   # → [SystemMessage(system + context_block), HumanMessage(user)]

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from .contexts import BaseContext

CONTEXT_HEADER = "## 上下文信息"


class PromptBuilder:
    """Build structured prompt messages from typed contexts.

    Context is rendered as a separate, clearly marked block within the
    SystemMessage, not appended as a raw string to the system prompt.
    """

    @staticmethod
    def build(
        system: str,
        context: BaseContext | None = None,
        user: str = "",
    ) -> list:
        """Build [SystemMessage, HumanMessage] list.

        Args:
            system: The agent's system prompt (ROUTER_SYSTEM, PLAN_PROMPT, etc.).
            context: Typed context dataclass (RouterContext, PlannerContext, ...).
            user: The human message / instruction.

        Returns:
            List of LangChain messages ready for llm.ainvoke().
        """
        # Render context block
        context_block = ""
        if context is not None:
            rendered = context.to_prompt()
            if rendered.strip():
                context_block = f"\n\n{CONTEXT_HEADER}\n{rendered}"

        # Assemble system message: system prompt → context block (clearly separated)
        system_content = system + context_block

        messages = [SystemMessage(content=system_content)]
        if user.strip():
            messages.append(HumanMessage(content=user))

        return messages

    @staticmethod
    def build_system_only(
        system: str,
        context: BaseContext | None = None,
    ) -> SystemMessage:
        """Build just the SystemMessage (when HumanMessage is added separately)."""
        context_block = ""
        if context is not None:
            rendered = context.to_prompt()
            if rendered.strip():
                context_block = f"\n\n{CONTEXT_HEADER}\n{rendered}"

        return SystemMessage(content=system + context_block)
