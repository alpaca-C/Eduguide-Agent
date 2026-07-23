"""Health check + monitoring stats endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/monitoring/stats")
async def monitoring_stats():
    """Get aggregated QA request stats and processing stats."""
    result = {"request": None, "processing": None}
    try:
        from src.monitoring.usage_store import get_recorder_store
        store = get_recorder_store()
        result["request"] = store.request_aggregate()
        result["processing"] = store.processing_aggregate()
    except Exception as e:
        result["error"] = str(e)
    return result
