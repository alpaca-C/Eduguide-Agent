"""Unit tests for src.harness.rate_limit — sliding-window rate limiter.

Pure in-memory logic: no IO, no async, no external deps. Every line is testable.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.harness.rate_limit import RateLimiter, RateLimitError


# ═══════════════════════════════════════════════════════════════════════
# RateLimitError
# ═══════════════════════════════════════════════════════════════════════

class TestRateLimitError:
    def test_contains_tool_name(self):
        err = RateLimitError("web_search", 10, 60.0)
        assert "web_search" in str(err)
        assert "10" in str(err)
        assert "60" in str(err)

    def test_attributes(self):
        err = RateLimitError("ocr", 5, 30.0)
        assert err.tool_name == "ocr"
        assert err.max_calls == 5
        assert err.window_s == 30.0


# ═══════════════════════════════════════════════════════════════════════
# RateLimiter
# ═══════════════════════════════════════════════════════════════════════

class TestRateLimiter:
    """Core sliding-window logic."""

    def test_no_limit_always_allowed(self):
        limiter = RateLimiter()
        for _ in range(100):
            limiter.check("unlimited_tool")  # no exception

    def test_under_limit_allowed(self):
        limiter = RateLimiter({"web_search": (5, 60.0)})
        for _ in range(5):
            limiter.check("web_search")  # all OK

    def test_exceeds_limit_raises(self):
        limiter = RateLimiter({"web_search": (3, 60.0)})
        for _ in range(3):
            limiter.check("web_search")
        with pytest.raises(RateLimitError) as exc:
            limiter.check("web_search")
        assert exc.value.tool_name == "web_search"

    def test_window_slides_old_entries_expire(self):
        """Old timestamps outside the window should be discarded."""
        limiter = RateLimiter({"api": (2, 1.0)})  # 2 calls per second

        # Call twice at t=0 — fills the window
        limiter.check("api")
        limiter.check("api")

        # Third call at t=0 should be rejected
        with pytest.raises(RateLimitError):
            limiter.check("api")

        # Manually expire old entries by manipulating _windows
        now = time.time()
        limiter._windows["api"] = [now - 2.0, now - 1.5]  # both > 1s ago

        # Now a call should succeed (both old entries are outside the 1s window)
        limiter.check("api")

    def test_multiple_tools_independent(self):
        limiter = RateLimiter({
            "web_search": (2, 60.0),
            "ocr": (10, 60.0),
        })

        # Fill web_search quota
        limiter.check("web_search")
        limiter.check("web_search")
        with pytest.raises(RateLimitError):
            limiter.check("web_search")

        # ocr should still work independently
        for _ in range(10):
            limiter.check("ocr")

    def test_empty_limits_all_unlimited(self, limiter=None):
        if limiter is None:
            limiter = RateLimiter({})
        limiter.check("anything")

    def test_remaining_unlimited_returns_none(self):
        limiter = RateLimiter()
        assert limiter.remaining("not_configured") is None

    def test_remaining_counts_correctly(self):
        limiter = RateLimiter({"search": (5, 60.0)})
        assert limiter.remaining("search") == 5
        limiter.check("search")
        assert limiter.remaining("search") == 4

    def test_remaining_respects_window(self):
        limiter = RateLimiter({"search": (3, 1.0)})
        limiter.check("search")
        limiter.check("search")
        assert limiter.remaining("search") == 1

        # Expire old entries
        now = time.time()
        limiter._windows["search"] = [now - 2.0, now - 1.5]
        # Both outside 1s window → remaining should be 3
        assert limiter.remaining("search") == 3

    def test_remaining_never_negative(self):
        limiter = RateLimiter({"search": (1, 60.0)})
        limiter.check("search")
        # Manually add extra to corrupt state
        limiter._windows["search"].append(time.time())
        limiter._windows["search"].append(time.time())
        # remaining should be max(0, limit - count) → 0, not -2
        assert limiter.remaining("search") >= 0
