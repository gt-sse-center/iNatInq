"""Unit tests for the FastAPI lifespan context manager.

The lifespan manages graceful shutdown of the semantic cache client.
Periodic invalidation is handled inline by ``CacheClient.lookup()``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from api.app import lifespan
from fastapi import FastAPI


@pytest.fixture
def mock_cache() -> MagicMock:
    """Return a mock CacheClient with the methods the lifespan uses."""
    cache = MagicMock()
    cache.close = AsyncMock()
    return cache


@pytest.mark.asyncio
async def test_lifespan_closes_cache_on_shutdown(mock_cache: MagicMock) -> None:
    """The cache client is closed during shutdown.

    Why important: The in-memory Qdrant client holds resources that
    must be released to avoid warnings or leaks.
    """
    app = FastAPI()

    with patch("api.app.get_semantic_cache", return_value=mock_cache):
        async with lifespan(app):
            pass

    mock_cache.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_skips_close_when_cache_disabled() -> None:
    """No close is attempted when the cache is disabled (None).

    Why important: When the cache feature is turned off, the lifespan
    must not attempt to close a non-existent client.
    """
    app = FastAPI()

    with patch("api.app.get_semantic_cache", return_value=None):
        async with lifespan(app):
            pass
    # No assertion needed — just verifying no exception was raised
