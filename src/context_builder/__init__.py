# Context Builder — GSSC (Gather-Select-Structure-Compress) pipeline
# + ContextRouter + PromptBuilder.
#
# Usage:
#   from src.context_builder import GSSCPipeline, ContextRouter, PromptBuilder
#
#   # GSSC pipeline: full gather→select→structure→compress
#   pipeline = GSSCPipeline(memory_manager, tool_registry)
#   prompt = await pipeline.run(question, session_id, ...)
#
#   # ContextRouter: per-agent typed contexts
#   router_ctx = ContextRouter().build_router(question, memory_ctx)
#
#   # PromptBuilder: structured message construction (no string concat)
#   messages = PromptBuilder.build(system=ROUTER_SYSTEM, context=router_ctx, user=q)

from .schema import Fragment, ScoredFragment, StructuredPrompt, TEMPLATE
from .gather import Gatherer
from .select import Selector
from .structure import Structurer
from .compress import Compressor
from .pipeline import GSSCPipeline
from .contexts import (
    BaseContext,
    RewriterContext,
    RouterContext,
    SolverContext,
    PlannerContext,
    ReflectorContext,
)
from .router import ContextRouter
from .builder import PromptBuilder

__all__ = [
    # GSSC pipeline
    "GSSCPipeline",
    "Fragment",
    "ScoredFragment",
    "StructuredPrompt",
    "TEMPLATE",
    "Gatherer",
    "Selector",
    "Structurer",
    "Compressor",
    # Typed contexts
    "BaseContext",
    "RewriterContext",
    "RouterContext",
    "SolverContext",
    "PlannerContext",
    "ReflectorContext",
    # Routing + building
    "ContextRouter",
    "PromptBuilder",
]
