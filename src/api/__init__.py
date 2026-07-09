"""Document QA System — FastAPI application factory.

Routers are split by domain:
  router_health    — GET  /api/health
  router_files     — POST /api/files/upload  | GET /api/files/list  | DELETE /api/files/{name}
  router_chapters  — POST /api/chapters/detect | POST /api/chapters/save | GET /api/chapters/{name}
  router_knowledge — POST /api/knowledge/process | DELETE /api/knowledge/clear | GET /api/knowledge/stats | GET /api/knowledge/documents
  router_chat      — POST /api/chat
  router_sessions  — GET  /api/sessions | GET /api/sessions/{id} | DELETE /api/sessions/{id}
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── App factory ────────────────────────────────────────────────────────
app = FastAPI(title="Document QA System", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ───────────────────────────────────────────────────
from .router_health import router as health_router
from .router_files import router as files_router
from .router_chapters import router as chapters_router
from .router_knowledge import router as knowledge_router
from .router_chat import router as chat_router
from .router_sessions import router as sessions_router

app.include_router(health_router)
app.include_router(files_router)
app.include_router(chapters_router)
app.include_router(knowledge_router)
app.include_router(chat_router)
app.include_router(sessions_router)

# ── Optional: Monitoring router ─────────────────────────────────────────
try:
    from src.monitoring.reporter import router as monitoring_router
    app.include_router(monitoring_router)
except Exception as e:
    logger.debug("Monitoring router not available, skipping: %s", e)

# ── Static files (frontend) ─────────────────────────────────────────────
from starlette.staticfiles import StaticFiles as _StaticFiles
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dist_dir = PROJECT_ROOT / "frontend" / "dist"
frontend_dir = PROJECT_ROOT / "frontend"
_static_dir = dist_dir if dist_dir.exists() else frontend_dir

if _static_dir.exists():
    # Mount with html=True for SPA fallback. Cache for 1 hour on assets
    # (they have content hashes) but not on index.html.
    app.mount(
        "/",
        _StaticFiles(directory=str(_static_dir), html=True),
        name="frontend",
    )
