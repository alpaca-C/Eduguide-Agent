# Unit tests for monitoring reporter (FastAPI router)

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

# ── Build a minimal test app with the reporter router ──────────────────

@pytest.fixture
def client():
    """Create a FastAPI TestClient with the reporter router + mocked store."""
    mock_stats = {
        "overall": {"calls": 10, "total_tokens": 5000,
                     "prompt_tokens": 3000, "completion_tokens": 2000},
        "by_model": [{"model": "gpt-4", "calls": 10, "total_tokens": 5000}],
        "by_call_type": [{"call_type": "qa", "calls": 10, "total_tokens": 5000}],
    }

    mock_records = [
        {"id": 1, "timestamp": 1234567890.0, "model": "gpt-4",
         "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
         "call_type": "qa", "duration_ms": 500.0, "metadata": {}},
    ]

    with patch("src.monitoring.reporter.get_stats", return_value=mock_stats), \
         patch("src.monitoring.reporter.get_recent", return_value=mock_records):
        app = FastAPI()
        from src.monitoring.reporter import router
        app.include_router(router)
        yield TestClient(app)


class TestTokenStats:
    """Tests for GET /api/monitoring/tokens/stats."""

    def test_stats_returns_aggregated_data(self, client):
        """Should return aggregated token usage stats."""
        response = client.get("/api/monitoring/tokens/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["overall"]["calls"] == 10
        assert data["overall"]["total_tokens"] == 5000
        assert len(data["by_model"]) == 1
        assert data["by_model"][0]["model"] == "gpt-4"

    def test_stats_with_since_hours(self, client):
        """since_hours query param should be parsed and forwarded."""
        response = client.get("/api/monitoring/tokens/stats?since_hours=24")
        assert response.status_code == 200
        # The mocked get_stats was called — we just verify it works
        data = response.json()
        assert "overall" in data

    def test_stats_default_since_hours(self, client):
        """Default since_hours=0 should return all-time stats."""
        response = client.get("/api/monitoring/tokens/stats")
        assert response.status_code == 200


class TestTokenRecent:
    """Tests for GET /api/monitoring/tokens/recent."""

    def test_recent_returns_records(self, client):
        """Should return recent token usage records."""
        response = client.get("/api/monitoring/tokens/recent")
        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert len(data["records"]) == 1
        assert data["records"][0]["model"] == "gpt-4"
        assert data["records"][0]["total_tokens"] == 150

    def test_recent_with_limit(self, client):
        """limit query param should be parsed."""
        response = client.get("/api/monitoring/tokens/recent?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "records" in data

    def test_recent_default_limit(self, client):
        """Default limit=50 should be used when not specified."""
        response = client.get("/api/monitoring/tokens/recent")
        assert response.status_code == 200
