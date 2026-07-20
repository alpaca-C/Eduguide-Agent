# Supervisor — thin orchestration layer above skills.
#
# The Supervisor doesn't make QA decisions. It only:
#   1. Recalls memory via MemoryManager
#   2. Feeds context to the right skill
#   3. Returns the result
#
# Currently only one skill: problem_solve (full QA pipeline).
# Future: more skills (simple_rag, web_research, ...) with LLM routing.

from .supervisor import Supervisor, SupervisorOutput

__all__ = ["Supervisor", "SupervisorOutput"]
