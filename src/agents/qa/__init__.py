# QA Agent Module — 5 sub-agents + orchestrator
#
# Architecture:
#   QuestionRouter → DirectSolver (moderate) / Planner→Executor→Reflector (complex)
#
# Sub-agents:
#   router.py      — QuestionRouter: difficulty classification
#   solver.py      — DirectSolver: think→act→synthesize for moderate questions
#   planner.py     — Planner: decompose + summarize complex questions
#   executor.py    — Executor: concurrent tool execution
#   reflector.py   — Reflector: structured review with search suggestions
#
# Orchestrator:
#   orchestrator.py — 3-tier routing logic + QA entry point

from .orchestrator import QASystem, answer_question, get_agent
from .router import QuestionRouter
from .solver import DirectSolver
from .planner import Planner
from .executor import Executor
from .reflector import Reflector
from .query_rewriter import QueryRewriter
from .vector_store import DocumentVectorStore, CrossLingualEmbeddingFunction
