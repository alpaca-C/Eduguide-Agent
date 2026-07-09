# Token Monitoring Module -- zero-intrusion LLM token tracking
#
# Activation:
#   1. Set env var MONITORING_ENABLED=true (auto-activates on import)
#   2. Use the wrapper script: python run_with_monitor.py
#   3. Or import manually: import src.monitoring
#
# All token usage is logged to storage/token_usage.db

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .store import TokenRecord, get_token_store

logger = logging.getLogger("token-monitor")

# ---------------------------------------------------------------------------
# LangChain Callback Handler -- intercepts all LLM calls automatically
# ---------------------------------------------------------------------------


class TokenTrackerCallback(BaseCallbackHandler):
    """LangChain callback that records token usage from every LLM call.

    Automatically fires on_llm_end for every ChatOpenAI invocation made
    through LangChain''s standard interface (invoke, ainvoke, etc.).
    """

    def __init__(self):
        super().__init__()
        self._call_starts: dict[str, float] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._call_starts[run_id] = time.time()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        start_time = self._call_starts.pop(run_id, time.time())
        duration_ms = (time.time() - start_time) * 1000

        # Extract token usage from LLMResult
        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}

        # Fallback: try generations[0].generation_info
        if not token_usage:
            for gen_list in response.generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    if gen_info:
                        maybe_usage = gen_info.get("token_usage") or gen_info.get("usage_metadata")
                        if maybe_usage:
                            token_usage = maybe_usage
                            break
                if token_usage:
                    break

        # Normalize token usage keys (different providers use different names)
        prompt_tokens = (
            token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or token_usage.get("prompt_token_count")
            or 0
        )
        completion_tokens = (
            token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or token_usage.get("candidates_token_count")
            or 0
        )
        total_tokens = (
            token_usage.get("total_tokens")
            or token_usage.get("total_token_count")
            or (prompt_tokens + completion_tokens)
        )

        # Determine model name
        model = response.llm_output.get("model_name", "") if response.llm_output else ""
        if not model:
            for gen_list in response.generations:
                for gen in gen_list:
                    gen_info = getattr(gen, "generation_info", {}) or {}
                    model = gen_info.get("model_name", "")
                    if model:
                        break
                if model:
                    break

        # Infer call type from tags/metadata
        call_type = "llm_call"
        if tags:
            for tag in tags:
                if tag in ("qa", "extractor", "chapterizer", "reviewer", "classify"):
                    call_type = tag
                    break

        # Build the record
        record = TokenRecord(
            timestamp=time.time(),
            model=model or "unknown",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            call_type=call_type,
            duration_ms=round(duration_ms, 2),
            metadata={
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else "",
            },
        )

        if total_tokens > 0:
            store = get_token_store()
            store.insert(record)
            logger.info(
                "Token used: %d (prompt=%d, completion=%d) | model=%s | %.0fms | %s",
                total_tokens, prompt_tokens, completion_tokens,
                model or "?", duration_ms, call_type,
            )


# ---------------------------------------------------------------------------
# Monkey-patch ChatOpenAI to register the callback
# ---------------------------------------------------------------------------

_patched = False


def _patch_chat_openai():
    """Monkey-patch langchain_openai.ChatOpenAI to inject our callback.

    This ensures ALL ChatOpenAI instances (including existing ones created
    before this module was imported) get the token tracker callback.
    """
    global _patched
    if _patched:
        return

    try:
        from langchain_openai import ChatOpenAI

        original_init = ChatOpenAI.__init__
        _tracker = TokenTrackerCallback()

        def patched_init(self, *args, **kwargs):
            callbacks = kwargs.get("callbacks", None)
            if callbacks is None:
                kwargs["callbacks"] = [_tracker]
            elif isinstance(callbacks, list):
                if not any(isinstance(cb, TokenTrackerCallback) for cb in callbacks):
                    kwargs["callbacks"] = callbacks + [_tracker]
            original_init(self, *args, **kwargs)

        ChatOpenAI.__init__ = patched_init
        _patched = True
        logger.info("TokenTracker: patched ChatOpenAI.__init__")
    except ImportError:
        logger.debug("langchain_openai not available, skipping ChatOpenAI patch")


def _register_global_callback():
    """Register the token tracker as a global LangChain callback.

    This catches LLM calls that go through LangChain''s chain/graph infrastructure
    even if they weren''t created via the patched ChatOpenAI.
    """
    try:
        from langchain_core.callbacks.manager import (
            _get_global_callback_manager,
            CallbackManager,
        )

        manager = _get_global_callback_manager()
        _tracker = TokenTrackerCallback()
        if manager is None:
            CallbackManager.configure(inheritable_callbacks=[_tracker])
        else:
            manager.add_handler(_tracker)

        logger.info("TokenTracker: registered global callback handler")
    except Exception as e:
        logger.debug("Failed to register global callback: %s", e)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def _should_activate() -> bool:
    """Check if monitoring should be enabled."""
    return os.environ.get("MONITORING_ENABLED", "").lower() in ("true", "1", "yes", "on")


def _activate():
    """Activate token monitoring."""
    if not _should_activate():
        return
    _patch_chat_openai()
    _register_global_callback()
    logger.info("TokenMonitor activated -- usage logged to %s", get_token_store().db_path)


# Activate on import if MONITORING_ENABLED is set
_activate()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_stats(since_ts: float = 0.0) -> dict:
    """Get aggregate token usage statistics."""
    return get_token_store().stats(since_ts)


def get_recent(limit: int = 50) -> list[dict]:
    """Get most recent token usage records."""
    return get_token_store().recent(limit)


def enable():
    """Explicitly enable monitoring at runtime."""
    os.environ["MONITORING_ENABLED"] = "true"
    _activate()


def disable():
    """Disable monitoring at runtime (won''t undo existing patches though)."""
    os.environ["MONITORING_ENABLED"] = "false"
