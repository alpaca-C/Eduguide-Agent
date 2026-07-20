"""Sliding-window rate limiter for tool calls."""

from __future__ import annotations

import time
from collections import defaultdict


class RateLimitError(Exception):
    """Raised when a tool exceeds its rate limit."""
    def __init__(self, tool_name: str, max_calls: int, window_s: float):
        self.tool_name = tool_name
        self.max_calls = max_calls
        self.window_s = window_s
        super().__init__(
            f"Rate limit exceeded: {tool_name} ({max_calls} calls / {window_s}s)"
        )


class RateLimiter:
    """Sliding-window rate limiter per tool name.

    Usage:
        limiter = RateLimiter({"web_search": (10, 60), "mineru_ocr": (5, 60)})
        limiter.check("web_search")       # OK
        limiter.check("web_search") x 10  # 11th → raises RateLimitError
    """

    def __init__(self, limits: dict[str, tuple[int, float]] | None = None):
        """
        Args:
            limits: {tool_name: (max_calls, window_seconds)}
                    Tools not in this dict are not rate-limited.
        """
        self._limits: dict[str, tuple[int, float]] = dict(limits or {})
        self._windows: dict[str, list[float]] = defaultdict(list)

    def check(self, tool_name: str) -> None:
        """Check if calling `tool_name` would exceed its rate limit.

        Raises RateLimitError if limit exceeded. Otherwise records the call.
        """
        if tool_name not in self._limits:
            return  # No limit configured → always allowed

        max_calls, window_s = self._limits[tool_name]
        now = time.time()
        window_start = now - window_s

        # Slide the window: discard timestamps older than window_s
        timestamps = self._windows[tool_name]
        self._windows[tool_name] = [t for t in timestamps if t > window_start]

        if len(self._windows[tool_name]) >= max_calls:
            raise RateLimitError(tool_name, max_calls, window_s)

        self._windows[tool_name].append(now)

    def remaining(self, tool_name: str) -> int | None:
        """Return remaining calls in current window, or None if unlimited."""
        if tool_name not in self._limits:
            return None
        max_calls, window_s = self._limits[tool_name]
        now = time.time()
        self._windows[tool_name] = [
            t for t in self._windows[tool_name] if t > now - window_s
        ]
        return max(0, max_calls - len(self._windows[tool_name]))
