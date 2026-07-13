# Unit tests for web_search tool (Tavily API)

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools import ToolResult, ToolErrorType


# We import the function but patch its dependencies in tests
from src.tools.web_search import web_search


class TestWebSearchNotConfigured:
    """Tests for missing or misconfigured API key."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_not_configured(self, monkeypatch):
        """When TAVILY_API_KEY is not set, return NOT_CONFIGURED."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        result = await web_search("test query")

        assert isinstance(result, ToolResult)
        assert result.error == ToolErrorType.NOT_CONFIGURED
        assert result.tool_name == "web_search"
        assert result.query == "test query"
        assert "未配置" in result.content

    @pytest.mark.asyncio
    async def test_empty_api_key_returns_not_configured(self, monkeypatch):
        """When TAVILY_API_KEY is empty string, return NOT_CONFIGURED."""
        monkeypatch.setenv("TAVILY_API_KEY", "")

        result = await web_search("test query")

        assert result.error == ToolErrorType.NOT_CONFIGURED
        assert result.tool_name == "web_search"

    # Note: the ImportError branch (tavily not installed) is not tested here
    # because tavily IS installed in the venv. This path is covered when
    # deploying to environments without tavily-python.


class TestWebSearchSuccess:
    """Tests for successful web search."""

    @pytest.mark.asyncio
    async def test_search_success_with_answer(self, monkeypatch):
        """Successful search with answer and results should format correctly."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "This is the AI-generated summary.",
            "results": [
                {
                    "title": "Result One",
                    "content": "Content of result one.",
                    "url": "https://example.com/1",
                },
                {
                    "title": "Result Two",
                    "content": "Content of result two.",
                    "url": "https://example.com/2",
                },
            ],
        }

        with patch("tavily.TavilyClient", return_value=mock_client):
            result = await web_search("test query")

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.tool_name == "web_search"
        assert result.query == "test query"

        # Content should contain the answer and results
        assert "摘要" in result.content
        assert "This is the AI-generated summary" in result.content
        assert "Result One" in result.content
        assert "Result Two" in result.content
        assert "https://example.com/1" in result.content

        # Metadata should be set
        assert result.metadata["results_count"] == 2
        assert result.metadata["has_answer"] is True

    @pytest.mark.asyncio
    async def test_search_success_without_answer(self, monkeypatch):
        """Search results without an answer field should still work."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "",
            "results": [
                {"title": "Single Result", "content": "Content here.", "url": ""},
            ],
        }

        with patch("tavily.TavilyClient", return_value=mock_client):
            result = await web_search("another query")

        assert result.error is None
        assert result.metadata["results_count"] == 1
        assert result.metadata["has_answer"] is False
        assert "Single Result" in result.content

    @pytest.mark.asyncio
    async def test_search_result_content_truncation(self, monkeypatch):
        """Long content should be truncated to 300 chars."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        long_content = "X" * 500

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "",
            "results": [
                {"title": "Long", "content": long_content, "url": "https://x.com"},
            ],
        }

        with patch("tavily.TavilyClient", return_value=mock_client):
            result = await web_search("query")

        # The content in the result should contain truncated version (max 300)
        assert long_content[:300] in result.content
        assert long_content not in result.content  # Full 500-char string not present

    @pytest.mark.asyncio
    async def test_search_passes_max_results(self, monkeypatch):
        """max_results parameter should be passed to TavilyClient.search."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "",
            "results": [],
        }

        with patch("tavily.TavilyClient", return_value=mock_client):
            await web_search("query", max_results=10)

        # Verify max_results was forwarded
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["max_results"] == 10


class TestWebSearchErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_search_api_exception(self, monkeypatch):
        """Tavily API exception should return NETWORK error."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Connection refused")

        with patch("tavily.TavilyClient", return_value=mock_client):
            result = await web_search("query")

        assert result.error == ToolErrorType.NETWORK
        assert result.tool_name == "web_search"
        assert "出错" in result.content
        assert "Connection refused" in result.error_detail

    @pytest.mark.asyncio
    async def test_search_empty_results(self, monkeypatch):
        """Empty results list with no answer should return EMPTY_RESULT."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "",
            "results": [],
        }

        with patch("tavily.TavilyClient", return_value=mock_client):
            result = await web_search("obscure query")

        assert result.error == ToolErrorType.EMPTY_RESULT
        assert result.tool_name == "web_search"
        assert "未找到" in result.content

    @pytest.mark.asyncio
    async def test_search_result_with_missing_title(self, monkeypatch):
        """Results missing title field should use fallback text."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "answer": "Summary.",
            "results": [
                {"content": "No title here.", "url": "https://example.com"},
            ],
        }

        with patch("tavily.TavilyClient", return_value=mock_client):
            result = await web_search("query")

        assert result.error is None
        assert "无标题" in result.content
