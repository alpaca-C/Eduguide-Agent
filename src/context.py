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

import logging
from dataclasses import dataclass, field
from typing import Optional

from .config import Configuration
from .knowledge.graph import KnowledgeGraph
from .memory.store import MemoryStore

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

    # Wire RAG tool to vector store + knowledge graph
    from .tools.rag_search import init_rag_tool
    init_rag_tool(ctx.vector_store, ctx.knowledge_graph)

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
