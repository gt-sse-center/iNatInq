"""Unit tests for the FastAPI lifespan context manager.

The lifespan manages a background ``asyncio.Task`` that periodically
invalidates the semantic cache, plus graceful shutdown (cancel task,
close cache client).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.app import lifespan
from fastapi import FastAPI


@pytest.fixture
def mock_cache() -> MagicMock:
    """Return a mock CacheClient with the methods the lifespan uses."""
    cache = MagicMock()
    cache.config = MagicMock()
    cache.config.invalidation_interval_seconds = 60
    cache.invalidate = AsyncMock()
    cache.close = AsyncMock()
    return cache


# ── Startup ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lifespan_starts_invalidation_task(mock_cache: MagicMock) -> None:
    """Background invalidation task is created when the cache is enabled.

    Why important: Ensures periodic cache invalidation is actually
    scheduled at startup so stale entries are automatically evicted.
    """
    app = FastAPI()

    with (
        patch("api.app.get_semantic_cache", return_value=mock_cache),
        patch("api.app.asyncio.create_task", wraps=asyncio.create_task) as mock_create_task,
    ):
        async with lifespan(app):
            mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_skips_when_cache_disabled() -> None:
    """No background task is created when the cache is disabled (None).

    Why important: When the cache feature is turned off, the lifespan
    must not attempt to create or manage any task.
    """
    app = FastAPI()

    with (
        patch("api.app.get_semantic_cache", return_value=None),
        patch("api.app.asyncio.create_task") as mock_create_task,
    ):
        async with lifespan(app):
            mock_create_task.assert_not_called()


# ── Shutdown ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lifespan_cancels_task_on_shutdown(mock_cache: MagicMock) -> None:
    """The invalidation task is cancelled when the app shuts down.

    Why important: A lingering background task after shutdown could
    cause ``RuntimeError`` or resource leaks.
    """
    app = FastAPI()

    async def _hang_forever(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
        await asyncio.sleep(999_999)

    with (
        patch("api.app.get_semantic_cache", return_value=mock_cache),
        patch("api.app._cache_invalidation_loop", side_effect=_hang_forever),
    ):
        async with lifespan(app):
            # Grab the running task created by lifespan
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            assert len(tasks) == 1
            task = tasks[0]
            assert not task.done()

    # After exiting the context, the task must have been cancelled
    assert task.cancelled()


@pytest.mark.asyncio
async def test_lifespan_closes_cache_on_shutdown(mock_cache: MagicMock) -> None:
    """The cache client is closed during shutdown.

    Why important: The in-memory Qdrant client holds resources that
    must be released to avoid warnings or leaks.
    """
    app = FastAPI()

    async def _hang_forever(*_args: object, **_kwargs: object) -> None:  # noqa: RUF029
        await asyncio.sleep(999_999)

    with (
        patch("api.app.get_semantic_cache", return_value=mock_cache),
        patch("api.app._cache_invalidation_loop", side_effect=_hang_forever),
    ):
        async with lifespan(app):
            pass

    mock_cache.close.assert_awaited_once()
