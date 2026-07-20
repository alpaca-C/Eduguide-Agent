# AppContext — centralized dependency injection container
#
# Replaces scattered module-level globals (_vector_store, _knowledge_graph,
# _agent, _store, etc.) with a single initialized dataclass.
#
# Usage:
#   from src.context import AppContext, init_context, get_context
#
#   # At app startup:
#   ctx = init_context()
#
#   # In routers / agents:
#   ctx = get_context()
#   agent = ReflectionAgent(ctx.config)
#   results = ctx.vector_store.search(query)

from __future__ import annotations

import os as _os
_os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# PaddlePaddle 3.x ONEDNN bug workaround on Windows
_os.environ.setdefault("FLAGS_use_onednn", "0")

import logging
from dataclasses import dataclass, field
from typing import Optional

from .config import Configuration
from .knowledge.graph import KnowledgeGraph
from .memory.store import MemoryStore
from .memory.short_term import ShortTermMemory
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .memory.manager import MemoryManager
from .cache import ExactMatchCache
from .context_builder import GSSCPipeline
from .tools import get_tool_registry
from .skills.rag_retrieval import RAGRetrievalSkill
from .skills.problem_solve import ProblemSolveSkill
from .supervisor import Supervisor

logger = logging.getLogger(__name__)

# Module-level singleton — initialized once at app startup
_ctx: Optional[AppContext] = None


@dataclass
class AppContext:
    """Centralized container for all application-level dependencies.

    Each field represents a shared service that was previously a module-level
    global. This single container enables:
    - Clean test isolation (instantiate a fresh AppContext per test)
    - Clear ownership (one place to see all shared state)
    - Gradual migration to FastAPI Depends (get_context() is already a callable)
    """

    config: Configuration
    memory_store: MemoryStore
    knowledge_graph: KnowledgeGraph

    # Unified memory manager — short_term + episodic + semantic
    memory_manager: MemoryManager = field(default=None, repr=False)

    # Episodic memory (also accessible via memory_manager.episodic)
    episodic_memory: EpisodicMemory = field(default=None, repr=False)

    # Standalone cache (not part of memory — performance acceleration only)
    exact_cache: ExactMatchCache = field(default=None, repr=False)

    # GSSC context builder pipeline (Gather → Select → Structure → Compress)
    gssc_pipeline: GSSCPipeline = field(default=None, repr=False)

    # RAG retrieval skill (manages fast/full tier switching)
    rag_skill: RAGRetrievalSkill = field(default=None, repr=False)

    # Supervisor — thin dispatch layer above skills
    supervisor: Supervisor = field(default=None, repr=False)

    # These are initialized lazily to avoid heavy imports at module load time
    chapter_agent: object = field(default=None, repr=False)
    vector_store: object = field(default=None, repr=False)

    # In-memory caches shared across routers
    chapters_cache: dict = field(default_factory=dict)
    uploaded_files: dict = field(default_factory=dict)


def init_context(
    config: Configuration | None = None,
    memory_db_path: str = "",
) -> AppContext:
    """Initialize the application context (called once at startup).

    Args:
        config: Optional pre-built Configuration. If None, loads from env.
        memory_db_path: Optional path for the memory store database.

    Returns:
        The initialized AppContext (also stored as module-level singleton).
    """
    global _ctx

    if config is None:
        config = Configuration.from_env()

    store = MemoryStore(db_path=memory_db_path or config.memory_db_path or "")

    # Path-based imports to avoid circular dependencies at module load
    from .memory import set_memory_store
    set_memory_store(store)

    kg = KnowledgeGraph()

    # Lazy init — heavy objects created on first access via properties
    ctx = AppContext(
        config=config,
        memory_store=store,
        knowledge_graph=kg,
    )

    # Eager-init light objects
    from .agents.chapterizer import ChapterizerAgent
    ctx.chapter_agent = ChapterizerAgent(config)

    from .agents.qa import DocumentVectorStore
    ctx.vector_store = DocumentVectorStore()

    # Wire RAG tool to vector store + knowledge graph (backward compat)
    # Also pass MemoryManager for unified recall path
    from .tools.rag_search import init_rag_tool
    init_rag_tool(vs=ctx.vector_store, kg=ctx.knowledge_graph)

    # Build standalone exact-match cache (not part of memory)
    ctx.exact_cache = ExactMatchCache(store)

    # Build EpisodicMemory — cross-session experience recording + semantic recall
    ctx.episodic_memory = EpisodicMemory()

    # Build unified MemoryManager — short_term + episodic + semantic
    ctx.memory_manager = MemoryManager(
        short_term=ShortTermMemory(store),
        episodic=ctx.episodic_memory,
        semantic=SemanticMemory(ctx.vector_store, ctx.knowledge_graph),
    )
    logger.info("MemoryManager: short_term + episodic + semantic layers ready")

    # Build GSSC context pipeline
    ctx.gssc_pipeline = GSSCPipeline(
        memory_manager=ctx.memory_manager,
        tool_registry=get_tool_registry(),
        token_budget=config.context_token_budget,
        hard_limit=config.context_hard_limit,
        relevance_weight=config.context_relevance_weight,
        recency_weight=config.context_recency_weight,
        min_score=config.context_min_score,
    )
    logger.info("GSSCPipeline: Gather→Select→Structure→Compress ready")

    # Build RAG retrieval skill (manages fast/full tier switching)
    ctx.rag_skill = RAGRetrievalSkill()
    logger.info("RAGRetrievalSkill: Dense+CE (default) / Dense+Sparse+Graph+CE (full) ready")

    # Build QASystem → wrap in ProblemSolveSkill → build Supervisor
    # Wrapped in try-except so network issues during ChatOpenAI init
    # don't block the app from starting (health check still works).
    try:
        from .agents.qa.orchestrator import QASystem
        qa_system = QASystem(config, gssc_pipeline=ctx.gssc_pipeline, rag_skill=ctx.rag_skill)
        problem_solve = ProblemSolveSkill(qa_system)
        ctx.supervisor = Supervisor(ctx.memory_manager, {"problem_solve": problem_solve})
        logger.info("Supervisor: ready (1 skill registered: problem_solve)")
    except Exception as e:
        logger.error("Supervisor init failed (QA will be unavailable): %s", e)
        ctx.supervisor = None

    # Update RAG tool with MemoryManager (enables unified recall path)
    init_rag_tool(memory_manager=ctx.memory_manager)

    # Initialize harness hooks (logging, permissions, rate limits)
    from .harness import init_hooks
    init_hooks()

    _ctx = ctx
    logger.info("AppContext initialized successfully")
    return ctx


def get_context() -> AppContext:
    """Get the current application context.

    Raises RuntimeError if init_context() hasn't been called yet.
    """
    if _ctx is None:
        raise RuntimeError(
            "AppContext not initialized. Call init_context() at application startup."
        )
    return _ctx


def reset_context():
    """Reset the context (for testing)."""
    global _ctx
    _ctx = None
