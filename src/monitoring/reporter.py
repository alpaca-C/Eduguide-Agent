# Optional FastAPI router to expose token usage stats via API

from __future__ import annotations

from fastapi import APIRouter

from . import get_stats, get_recent

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/tokens/stats")
async def token_stats(since_hours: float = 0.0):
    """Get aggregated token usage statistics.

    Args:
        since_hours: Only include calls from the last N hours. 0 = all time.
    """
    import time
    since_ts = time.time() - since_hours * 3600 if since_hours > 0 else 0.0
    return get_stats(since_ts)


@router.get("/tokens/recent")
async def token_recent(limit: int = 50):
    """Get most recent token usage records."""
    return {"records": get_recent(limit)}
