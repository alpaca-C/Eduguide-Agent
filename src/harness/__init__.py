"""Harness layer — unified hook system for tool calls and LLM calls.

Intercepts before/after tool execution and LLM invocation to provide:
  - Structured logging with per-request tracing
  - Tool permission checks (which Agent can call which tool)
  - Rate limiting (sliding-window per tool)
  - Latency tracking

Design: hooks are registered on a singleton HookManager. Tool functions are
auto-wrapped at registration time. LLM calls are intercepted in BaseAgent.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .rate_limit import RateLimiter, RateLimitError

logger = logging.getLogger(__name__)

# ── Per-request tracing ──────────────────────────────────────────

# Set by the API layer at the start of each request, read by hooks.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


# Per-agent tracing — set by orchestrator before dispatching to an agent
_agent_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_name", default="unknown"
)


# ── Per-request counters (flushed to RecorderStore at request end) ──

@dataclass
class RequestCounters:
    """Accumulated counters for one request. Reset per request."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    start_time: float = field(default_factory=time.time)


_request_counters: contextvars.ContextVar[RequestCounters] = contextvars.ContextVar(
    "request_counters", default=RequestCounters()
)


def get_request_id() -> str:
    """Get the current request ID. Returns empty string if not set."""
    return _request_id.get()


def set_request_id(rid: str = "") -> str:
    """Set the request ID for the current async context. Generates one if empty."""
    if not rid:
        rid = str(uuid.uuid4())[:8]
    _request_id.set(rid)
    return rid


# ── Hook context dataclasses ─────────────────────────────────────


@dataclass
class HookContext:
    """Base context passed to every hook."""
    request_id: str
    agent_name: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolHookContext(HookContext):
    """Context for tool-call hooks."""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)


@dataclass
class LLMHookContext(HookContext):
    """Context for LLM-call hooks."""
    model: str = ""
    message_count: int = 0


# ── Hook types ───────────────────────────────────────────────────

BeforeToolHook = Callable[[ToolHookContext], Awaitable[None]]
AfterToolHook = Callable[[ToolHookContext, Any], Awaitable[None]]
BeforeLLMHook = Callable[[LLMHookContext], Awaitable[None]]
AfterLLMHook = Callable[[LLMHookContext, Any], Awaitable[None]]


# ── HookManager ──────────────────────────────────────────────────


class HookManager:
    """Singleton registry for before/after hooks on tool calls and LLM calls."""

    def __init__(self):
        self._before_tool: list[BeforeToolHook] = []
        self._after_tool: list[AfterToolHook] = []
        self._before_llm: list[BeforeLLMHook] = []
        self._after_llm: list[AfterLLMHook] = []
        self._rate_limiter: RateLimiter = RateLimiter()
        # Tool permission map: tool_name → set of allowed agent names
        # Empty set or missing key → all agents allowed
        self._tool_permissions: dict[str, set[str]] = {}

    # ── Hook registration ────────────────────────────────────────

    def on_before_tool(self, fn: BeforeToolHook):
        self._before_tool.append(fn)

    def on_after_tool(self, fn: AfterToolHook):
        self._after_tool.append(fn)

    def on_before_llm(self, fn: BeforeLLMHook):
        self._before_llm.append(fn)

    def on_after_llm(self, fn: AfterLLMHook):
        self._after_llm.append(fn)

    # ── Permission & rate limit config ───────────────────────────

    def set_rate_limits(self, limits: dict[str, tuple[int, float]]):
        """Configure per-tool rate limits: {tool_name: (max_calls, window_seconds)}."""
        self._rate_limiter = RateLimiter(limits)

    def set_tool_permissions(self, permissions: dict[str, set[str]]):
        """Configure per-tool permissions: {tool_name: {agent_name, ...}}.
        Empty set → all agents allowed. Missing key → all agents allowed.
        Non-empty set → only the listed agents can call this tool.
        """
        self._tool_permissions = dict(permissions)

    # ── Request lifecycle ──────────────────────────────────────────

    def begin_request(self, request_id: str = ""):
        """Start a new request context — resets counters."""
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        _request_id.set(request_id)
        _request_counters.set(RequestCounters())

    def finish_request(
        self, *, session_id: str = "", question: str = "",
        route: str = "", rounds: int = 0,
    ) -> dict | None:
        """Flush accumulated counters to RecorderStore. Returns the record dict or None."""
        counters = _request_counters.get()
        rid = _request_id.get()
        if not rid or counters.llm_calls == 0:
            return None

        try:
            from src.monitoring.usage_store import (
                get_recorder_store, RequestRecord,
            )
            record = RequestRecord(
                request_id=rid,
                session_id=session_id,
                question=question,
                route=route,
                rounds=rounds,
                llm_calls=counters.llm_calls,
                prompt_tokens=counters.prompt_tokens,
                completion_tokens=counters.completion_tokens,
                tool_calls=counters.tool_calls,
                total_latency_ms=(time.time() - counters.start_time) * 1000,
            )
            store = get_recorder_store()
            store.insert_request(record)
            total_tokens = counters.prompt_tokens + counters.completion_tokens
            latency_ms = round(record.total_latency_ms)
            # Print to console (always visible regardless of log level)
            print(
                f"[QA] {rid} | route={route} rounds={rounds} | "
                f"LLM: {counters.llm_calls} calls, "
                f"tokens={total_tokens} (in={counters.prompt_tokens} out={counters.completion_tokens}) | "
                f"tools: {counters.tool_calls} calls | "
                f"latency: {latency_ms}ms"
            )
            logger.info(
                "[STATS] req=%s route=%s rounds=%d llm_calls=%d "
                "tokens(in=%d out=%d total=%d) tool_calls=%d latency=%.0fms",
                rid, route, rounds, counters.llm_calls,
                counters.prompt_tokens, counters.completion_tokens,
                total_tokens, counters.tool_calls, latency_ms,
            )
            return {
                "request_id": rid, "route": route, "rounds": rounds,
                "llm_calls": counters.llm_calls,
                "prompt_tokens": counters.prompt_tokens,
                "completion_tokens": counters.completion_tokens,
                "tool_calls": counters.tool_calls,
                "latency_ms": round(record.total_latency_ms),
            }
        except Exception as e:
            logger.warning("[STATS] flush failed: %s", e)
            return None

    # ── Fire hooks (called by the wrapper layer) ─────────────────

    async def fire_before_tool(self, ctx: ToolHookContext) -> None:
        """Run all before-tool hooks. May raise RateLimitError or PermissionError."""
        # Built-in: rate limit check
        try:
            self._rate_limiter.check(ctx.tool_name)
        except RateLimitError as e:
            logger.warning(
                "[HOOK] req=%s | RATE_LIMITED tool=%s (%s calls/s)",
                ctx.request_id, ctx.tool_name,
                self._rate_limiter._limits.get(ctx.tool_name, ("?", "?"))[0],
            )
            raise

        # Built-in: permission check
        # Empty set → all agents allowed. Non-empty → only listed agents allowed.
        allowed = self._tool_permissions.get(ctx.tool_name)
        if allowed is not None and allowed and ctx.agent_name not in allowed:
            msg = (
                f"Agent '{ctx.agent_name}' is not allowed to call tool "
                f"'{ctx.tool_name}'. Allowed: {allowed or 'none'}"
            )
            logger.warning("[HOOK] req=%s | PERMISSION_DENIED %s", ctx.request_id, msg)
            raise PermissionError(msg)

        # User-registered hooks
        for hook in self._before_tool:
            try:
                await hook(ctx)
            except Exception:
                logger.debug("[HOOK] before_tool hook failed", exc_info=True)

    async def fire_after_tool(self, ctx: ToolHookContext, result: Any) -> None:
        """Run all after-tool hooks. Hooks receive the tool result."""
        for hook in self._after_tool:
            try:
                await hook(ctx, result)
            except Exception:
                logger.debug("[HOOK] after_tool hook failed", exc_info=True)

    async def fire_before_llm(self, ctx: LLMHookContext) -> None:
        """Run all before-LLM hooks."""
        for hook in self._before_llm:
            try:
                await hook(ctx)
            except Exception:
                logger.debug("[HOOK] before_llm hook failed", exc_info=True)

    async def fire_after_llm(self, ctx: LLMHookContext, response: Any) -> None:
        """Run all after-LLM hooks. Hooks receive the LLM response."""
        for hook in self._after_llm:
            try:
                await hook(ctx, response)
            except Exception:
                logger.debug("[HOOK] after_llm hook failed", exc_info=True)


# ── Singleton ────────────────────────────────────────────────────

_hook_manager: HookManager | None = None


def get_hook_manager() -> HookManager:
    """Get the global HookManager singleton."""
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager()
    return _hook_manager


# ── Default hooks ────────────────────────────────────────────────


async def _log_before_tool(ctx: ToolHookContext):
    """Default hook: log every tool call start."""
    logger.info(
        "[HOOK] req=%s | agent=%s | TOOL_START tool=%s args=%s",
        ctx.request_id, ctx.agent_name, ctx.tool_name,
        str(ctx.tool_args)[:200],
    )


async def _log_after_tool(ctx: ToolHookContext, result: Any):
    """Default hook: log tool call result with latency + accumulate counter."""
    # Accumulate
    counters = _request_counters.get()
    counters.tool_calls += 1

    latency = time.time() - ctx.timestamp
    status = "OK"
    detail = ""
    if hasattr(result, "is_error") and result.is_error:
        status = f"ERROR({result.error.value if hasattr(result.error, 'value') else result.error})"
        detail = getattr(result, "error_detail", "")[:100]
    elif hasattr(result, "content"):
        detail = str(getattr(result, "content", ""))[:100]

    logger.info(
        "[HOOK] req=%s | agent=%s | TOOL_END tool=%s latency=%.2fs status=%s %s",
        ctx.request_id, ctx.agent_name, ctx.tool_name, latency, status, detail,
    )


async def _log_before_llm(ctx: LLMHookContext):
    """Default hook: log LLM call start + accumulate call counter."""
    counters = _request_counters.get()
    counters.llm_calls += 1

    logger.info(
        "[HOOK] req=%s | agent=%s | LLM_START model=%s msgs=%d",
        ctx.request_id, ctx.agent_name, ctx.model, ctx.message_count,
    )


async def _log_after_llm(ctx: LLMHookContext, response: Any):
    """Default hook: log LLM call result with token usage + accumulate tokens."""
    latency = time.time() - ctx.timestamp
    resp_len = 0
    tokens_in = 0
    tokens_out = 0
    if hasattr(response, "content"):
        resp_len = len(str(response.content))
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("token_usage", {}) or {}
        tokens_in = usage.get("prompt_tokens", 0) or 0
        tokens_out = usage.get("completion_tokens", 0) or 0

    # Accumulate
    counters = _request_counters.get()
    counters.prompt_tokens += tokens_in
    counters.completion_tokens += tokens_out

    logger.info(
        "[HOOK] req=%s | agent=%s | LLM_END model=%s latency=%.2fs "
        "tokens_in=%s tokens_out=%s resp_len=%d",
        ctx.request_id, ctx.agent_name, ctx.model, latency,
        tokens_in, tokens_out, resp_len,
    )


def init_hooks():
    """Initialize the hook system with default hooks.

    Called once at app startup from init_context().
    Registers:
      - Structured logging for tool calls and LLM calls
      - Tool permission map (DirectSolver → rag_search only)
      - Rate limits (web_search: 10/min, mineru_ocr: 5/min)
    """
    manager = get_hook_manager()

    # ── Structured logging ──
    manager.on_before_tool(_log_before_tool)
    manager.on_after_tool(_log_after_tool)
    manager.on_before_llm(_log_before_llm)
    manager.on_after_llm(_log_after_llm)

    # ── Tool permissions ──
    manager.set_tool_permissions({
        # DirectSolver only uses rag_search — enforce it
        "rag_search": set(),           # all agents allowed
        "web_search": {"Executor"},    # only Executor (complex path)
        "mineru_ocr": set(),           # all agents allowed (mainly parser, not agent-called)
    })

    # ── Rate limits ──
    manager.set_rate_limits({
        "web_search": (10, 60),       # 10 calls per 60 seconds
        "mineru_ocr": (5, 60),        # 5 calls per 60 seconds
        # rag_search is not limited (local, no API cost)
    })

    logger.info("Harness hooks initialized (logging + permissions + rate limits)")
