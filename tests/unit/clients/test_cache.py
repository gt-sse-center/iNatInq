"""Unit tests for clients.cache module.

This file tests the CacheClient class which provides an in-memory
semantic cache backed by Qdrant.

# Test Coverage

The tests cover:
  - Client Initialization: Timeout propagation, AsyncQdrantClient construction args, initialized flag
  - Singleton Pattern: Identity across multiple instantiations, re-initialization guard
  - Thread Safety: Concurrent creation from multiple threads yields a single instance
  - Close: Delegation to the underlying AsyncQdrantClient

# Test Structure

Tests use pytest class-based organization with a per-test fixture that resets
the singleton state so each test starts with a fresh CacheClient.

# Running Tests

Run with: uv run pytest tests/unit/clients/test_cache.py -v
"""

# pylint: disable=redefined-outer-name
# Pytest fixtures intentionally redefine fixture names - this is expected behavior

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.cache import CacheClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset CacheClient singleton state before and after every test."""
    CacheClient._instance = None  # pylint: disable=protected-access
    yield
    inst = CacheClient._instance  # pylint: disable=protected-access
    if inst is not None and hasattr(inst, "initialized"):
        del inst.initialized
    CacheClient._instance = None  # pylint: disable=protected-access


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Patch AsyncQdrantClient so no real in-memory Qdrant server is created."""
    with patch("clients.cache.AsyncQdrantClient") as patched:
        mock_instance = AsyncMock()
        patched.return_value = mock_instance
        yield patched


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestCacheClientInit:
    """Test suite for CacheClient initialization."""

    def test_creates_client_with_timeout(self, mock_qdrant_client: MagicMock) -> None:
        """Verify timeout is stored and AsyncQdrantClient receives the correct args.

        **Why this test is important:**
          - The cache must forward the timeout to the underlying Qdrant client
          - The Qdrant client must be opened in-memory (location=":memory:")

        **What it tests:**
          - CacheClient.timeout matches the provided value
          - AsyncQdrantClient is called once with location=":memory:" and the timeout
        """
        cache = CacheClient(timeout=30)

        assert cache.timeout == 30
        mock_qdrant_client.assert_called_once_with(location=":memory:", timeout=30)

    def test_sets_initialized_flag(self, mock_qdrant_client: MagicMock) -> None:
        """Confirm the initialized sentinel is set after construction.

        **Why this test is important:**
          - The initialized flag gates re-initialization; if it is not set,
            every call to __init__ would overwrite the client

        **What it tests:**
          - cache.initialized is True after first construction
        """
        cache = CacheClient(timeout=10)

        assert cache.initialized is True


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestCacheClientSingleton:
    """Test suite for CacheClient singleton behaviour."""

    def test_same_instance_returned(self, mock_qdrant_client: MagicMock) -> None:
        """Two constructions must return the exact same object.

        **Why this test is important:**
          - The singleton guarantee is the core contract of CacheClient

        **What it tests:**
          - `a is b` identity check across two CacheClient() calls
        """
        a = CacheClient(timeout=10)
        b = CacheClient(timeout=10)

        assert a is b

    def test_second_init_ignored(self, mock_qdrant_client: MagicMock) -> None:
        """A second instantiation with different args must not re-initialize.

        **Why this test is important:**
          - Prevents accidental reconfiguration of the shared cache
          - AsyncQdrantClient should be constructed exactly once

        **What it tests:**
          - timeout remains the value from the first call
          - AsyncQdrantClient constructor is called only once
        """
        first = CacheClient(timeout=10)
        second = CacheClient(timeout=99)

        assert second.timeout == 10
        assert first is second
        mock_qdrant_client.assert_called_once()

    def test_id_matches(self, mock_qdrant_client: MagicMock) -> None:
        """id() of two constructions must be equal.

        **Why this test is important:**
          - Complements the `is` check with an explicit id comparison

        **What it tests:**
          - id(a) == id(b) across two CacheClient() calls
        """
        a = CacheClient(timeout=5)
        b = CacheClient(timeout=5)

        assert id(a) == id(b)


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestCacheClientThreadSafety:
    """Test suite for CacheClient thread safety."""

    def test_concurrent_creation_returns_same_instance(self, mock_qdrant_client: MagicMock) -> None:
        """Threads racing to create the singleton must all get the same object.

        **Why this test is important:**
          - The double-checked locking in __new__ and __init__ must hold under
            actual thread contention
          - AsyncQdrantClient must be constructed exactly once even with races

        **What it tests:**
          - 10 threads each call CacheClient(timeout=10)
          - All returned instances are identical (`is` check)
          - AsyncQdrantClient constructor is invoked exactly once
        """
        results: list[CacheClient] = []
        barrier = threading.Barrier(10)

        def create():
            barrier.wait()
            return CacheClient(timeout=10)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(create) for _ in range(10)]
            results.extend(future.result() for future in as_completed(futures))

        first = results[0]
        assert all(r is first for r in results)
        mock_qdrant_client.assert_called_once()


# =============================================================================
# Close Tests
# =============================================================================


class TestCacheClientClose:
    """Test suite for CacheClient.close()."""

    @pytest.mark.asyncio
    async def test_close_delegates_to_inner_client(self, mock_qdrant_client: MagicMock) -> None:
        """close() must await the underlying AsyncQdrantClient.close().

        **Why this test is important:**
          - Failing to close the inner client would leak resources

        **What it tests:**
          - cache.client.close is awaited exactly once
        """
        cache = CacheClient(timeout=10)
        await cache.close()

        cache.client.close.assert_awaited_once()  # pylint: disable=no-member
