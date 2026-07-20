"""Tools module for the QA Reflection Agent.

Tool functions are automatically wrapped with harness hooks at registration
time — every tool call goes through before/after hooks for logging, permission
checks, and rate limiting. Callers (Executor, DirectSolver) need no changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class ToolErrorType(Enum):
    """Structured error classification for tool execution results."""
    NOT_CONFIGURED = "not_configured"   # API key / dependency missing
    NETWORK = "network"                 # HTTP / connection error
    TIMEOUT = "timeout"                 # Operation timed out
    EMPTY_RESULT = "empty_result"       # Search returned nothing
    RATE_LIMITED = "rate_limited"       # API rate limit hit
    INTERNAL = "internal"               # Unexpected internal error


@dataclass
class ToolResult:
    """Result from executing a tool.

    When `error` is set, `content` contains a human-readable error message.
    The `is_error` and `is_empty` properties provide convenient health checks.
    """
    tool_name: str
    query: str
    content: str
    error: ToolErrorType | None = None
    error_detail: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def is_empty(self) -> bool:
        return self.error == ToolErrorType.EMPTY_RESULT


ToolFunc = Callable[..., Awaitable[ToolResult]]

# Registry of available tools
_tool_registry: dict[str, dict] = {}


def _wrap_with_hooks(name: str, func: ToolFunc) -> ToolFunc:
    """Wrap a tool function to go through harness hooks.

    On every call: fire before-hooks → execute → fire after-hooks.
    If before-hooks raise RateLimitError → return ToolResult(RATE_LIMITED).
    If before-hooks raise PermissionError → return ToolResult(INTERNAL).
    """
    async def wrapped(*args, **kwargs):
        try:
            from src.harness import (
                get_hook_manager, ToolHookContext,
                get_request_id, _agent_name,
            )
        except ImportError:
            # Harness not initialized — call directly
            return await func(*args, **kwargs)

        manager = get_hook_manager()
        if manager is None:
            return await func(*args, **kwargs)

        agent_name = _agent_name.get()
        ctx = ToolHookContext(
            request_id=get_request_id(),
            agent_name=agent_name,
            tool_name=name,
            tool_args={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
        )

        # Before hooks (permission, rate limit, logging)
        try:
            await manager.fire_before_tool(ctx)
        except PermissionError:
            return ToolResult(
                tool_name=name,
                query=str(kwargs.get("query", "")),
                content=f"（工具 {name} 无调用权限：{agent_name}）",
                error=ToolErrorType.INTERNAL,
                error_detail=f"Permission denied for agent '{agent_name}' on tool '{name}'",
            )
        except Exception:
            # RateLimitError or other — treat as rate-limited
            return ToolResult(
                tool_name=name,
                query=str(kwargs.get("query", "")),
                content=f"（工具 {name} 调用频率超限，请稍后重试）",
                error=ToolErrorType.RATE_LIMITED,
                error_detail="Rate limit exceeded",
            )

        # Execute
        t0 = time.time()
        result = await func(*args, **kwargs)
        ctx.timestamp = t0  # Reset to call start time for latency calc

        # After hooks (logging, stats)
        await manager.fire_after_tool(ctx, result)
        return result

    return wrapped


def register_tool(name: str, description: str, func: ToolFunc):
    """Register a tool for the agent to use. Wraps with harness hooks."""
    _tool_registry[name] = {
        "name": name,
        "description": description,
        "func": _wrap_with_hooks(name, func),
    }


def get_tool_registry() -> dict:
    """Get all registered tools."""
    return dict(_tool_registry)


def get_tools_for_llm() -> list[dict]:
    """Get tools in OpenAI function-calling format for LangChain bind_tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询字符串",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
        for name, info in _tool_registry.items()
    ]
