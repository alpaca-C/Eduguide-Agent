"""Web search tool -- search the internet via Tavily API."""

from __future__ import annotations

import logging
import os

from . import ToolResult, ToolErrorType, register_tool

logger = logging.getLogger(__name__)


async def web_search(query: str, max_results: int = 5) -> ToolResult:
    """Search the web using Tavily Search API."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return ToolResult(
            tool_name="web_search",
            query=query,
            content="（网络搜索未配置 TAVILY_API_KEY）",
            error=ToolErrorType.NOT_CONFIGURED,
            error_detail="TAVILY_API_KEY environment variable is not set.",
        )

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )
    except ImportError:
        return ToolResult(
            tool_name="web_search",
            query=query,
            content="（tavily-python 未安装，请运行: pip install tavily-python）",
            error=ToolErrorType.NOT_CONFIGURED,
            error_detail="tavily-python package is not installed.",
        )
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return ToolResult(
            tool_name="web_search",
            query=query,
            content=f"（网络搜索出错: {e}）",
            error=ToolErrorType.NETWORK,
            error_detail=str(e),
        )

    parts = ["=== 网络搜索结果 ==="]

    answer = response.get("answer", "")
    if answer:
        parts.append(f"**摘要**: {answer}")

    results = response.get("results", [])
    for i, r in enumerate(results):
        title = r.get("title", "无标题")
        content = r.get("content", "")
        url = r.get("url", "")
        parts.append(f"\n[{i+1}] **{title}**")
        parts.append(f"    {content[:300]}")
        if url:
            parts.append(f"    来源: {url}")

    if len(parts) <= 1:
        return ToolResult(
            tool_name="web_search",
            query=query,
            content="（未找到网络搜索结果）",
            error=ToolErrorType.EMPTY_RESULT,
            error_detail="Tavily search returned no results.",
            metadata={"results_count": 0, "has_answer": False},
        )

    return ToolResult(
        tool_name="web_search",
        query=query,
        content="\n".join(parts),
        metadata={"results_count": len(results), "has_answer": bool(answer)},
    )


# Register the tool
register_tool(
    name="web_search",
    description="搜索互联网获取最新信息。当用户问题需要实时信息、新闻、或本地资料中不包含的知识时使用。",
    func=web_search,
)
