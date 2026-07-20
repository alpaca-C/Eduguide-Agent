# MemoryContext — unified recall result for MemoryManager
#
#   MemoryManager.recall(question, session_id) → MemoryContext
#
# The MemoryContext aggregates results from the three memory layers:
#   short_term  — conversation history (compressed for prompt injection)
#   episodic    — relevant past experiences (semantic vector recall)
#   semantic    — document knowledge (available doc names, on-demand search)

from __future__ import annotations

from dataclasses import dataclass, field


# ── Semantic ──────────────────────────────────────────────────────────────

@dataclass
class SemanticResult:
    """Result from a semantic memory search (vector store + knowledge graph).

    Populated on-demand via SemanticMemory.search() — not pre-computed
    during recall(), to avoid wasteful searches for trivial questions.
    """
    query: str = ""
    dense: list[dict] = field(default_factory=list)     # ChromaDB results
    sparse: list[dict] = field(default_factory=list)     # FTS5 BM25 results
    graph: list[dict] = field(default_factory=list)      # KG concepts with neighbors
    fused: list[dict] = field(default_factory=list)     # RRF fused ranking

    @property
    def total_hits(self) -> int:
        return len(self.dense) + len(self.sparse) + len(self.graph)

    @property
    def is_empty(self) -> bool:
        return self.total_hits == 0


# ── MemoryContext (unified) ───────────────────────────────────────────────

@dataclass
class MemoryContext:
    """Aggregated memory recalled for a specific question.

    Returned by MemoryManager.recall(). The QA orchestrator injects this
    into Router, Planner, Solver, and Reflector prompts.

    Three memory layers:
      short_term — conversation history (already compressed)
      episodic   — relevant past experiences (semantic vector recall)
      semantic   — available document names (heavy search on-demand)
    """
    question: str = ""
    session_id: str | None = None

    # ── Short-term: conversation history ──────────────────────────────
    chat_history: list[dict] = field(default_factory=list)   # raw messages
    history_context: str = ""                                 # compressed for prompts

    # ── Episodic: relevant past experiences ───────────────────────────
    episodes: list = field(default_factory=list)   # list[Episode] from EpisodicMemory.recall()

    # ── Semantic: available document names (lightweight) ──────────────
    available_docs: list[str] = field(default_factory=list)
