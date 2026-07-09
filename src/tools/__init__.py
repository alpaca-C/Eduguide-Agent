"""Tools module for the QA Reflection Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


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


def register_tool(name: str, description: str, func: ToolFunc):
    """Register a tool for the agent to use."""
    _tool_registry[name] = {"name": name, "description": description, "func": func}


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
