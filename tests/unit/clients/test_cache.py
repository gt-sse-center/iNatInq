"""Unit tests for clients.cache module.

This file tests the CacheClient class which provides an in-memory
semantic cache backed by Qdrant.

# Test Coverage

The tests cover:
  - Initialization: Collection creation, healthy flag, graceful failure
  - Cache get: Hit, miss (empty), miss (below threshold), failure fallback
  - Cache put: Storage, FIFO eviction when max_items exceeded
  - Close: Delegation to the underlying AsyncQdrantClient
  - Serialization: Round-trip of SearchResults through JSON payload

# Running Tests

Run with: uv run pytest tests/unit/clients/test_cache.py -v
"""

# pylint: disable=redefined-outer-name

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.cache import CacheClient, _deserialize_results, _serialize_results
from config import CacheConfig
from core.models import SearchResultItem, SearchResults


# =============================================================================
# Fixtures
# =============================================================================


def _make_config(max_items: int = 100, threshold: float = 0.9, timeout: float = 5.0) -> CacheConfig:
    return CacheConfig(max_items=max_items, cache_hit_score_threshold=threshold, timeout=timeout)


def _make_results(n: int = 2) -> SearchResults:
    """Create a small SearchResults for testing."""
    items = [
        SearchResultItem(point_id=f"pt-{i}", score=0.9 - i * 0.1, payload={"key": f"val-{i}"})
        for i in range(n)
    ]
    return SearchResults(items=items, total=n)


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Patch AsyncQdrantClient so no real Qdrant is created."""
    with patch("clients.cache.AsyncQdrantClient") as patched:
        instance = AsyncMock()
        patched.return_value = instance
        yield instance


# =============================================================================
# Initialization Tests
# =============================================================================


class TestCacheClientInit:
    """Test suite for CacheClient initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_collection(self, mock_qdrant_client: AsyncMock) -> None:
        """initialize() creates the cache collection and marks client healthy."""
        client = CacheClient(config=_make_config(), vector_size=512)
        await client.initialize()

        mock_qdrant_client.create_collection.assert_awaited_once()
        assert client._healthy is True

    @pytest.mark.asyncio
    async def test_initialize_failure_marks_unhealthy(self, mock_qdrant_client: AsyncMock) -> None:
        """If collection creation fails, the client stays unhealthy (no-op mode)."""
        mock_qdrant_client.create_collection.side_effect = RuntimeError("boom")
        client = CacheClient(config=_make_config(), vector_size=512)
        await client.initialize()

        assert client._healthy is False


# =============================================================================
# Cache Get Tests
# =============================================================================


class TestCacheClientGet:
    """Test suite for CacheClient.get()."""

    @pytest.mark.asyncio
    async def test_get_returns_none_when_unhealthy(self) -> None:
        """get() is a no-op when the cache failed to initialize."""
        with patch("clients.cache.AsyncQdrantClient"):
            client = CacheClient(config=_make_config(), vector_size=512)
            result = await client.get(query_vector=[0.1, 0.2], collection="docs")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_empty_cache(self, mock_qdrant_client: AsyncMock) -> None:
        """get() returns None when no points match."""
        mock_qdrant_client.query_points.return_value = MagicMock(points=[])
        client = CacheClient(config=_make_config(), vector_size=512)
        await client.initialize()

        result = await client.get(query_vector=[0.1, 0.2], collection="docs")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_results_on_cache_hit(self, mock_qdrant_client: AsyncMock) -> None:
        """get() returns cached SearchResults when score >= threshold."""
        expected = _make_results()
        point = MagicMock()
        point.score = 0.95
        point.payload = {"results_json": _serialize_results(expected), "collection": "docs"}
        mock_qdrant_client.query_points.return_value = MagicMock(points=[point])

        client = CacheClient(config=_make_config(threshold=0.9), vector_size=512)
        await client.initialize()

        result = await client.get(query_vector=[0.1, 0.2], collection="docs")
        assert result is not None
        assert result.total == expected.total
        assert len(result.items) == len(expected.items)
        assert result.items[0].point_id == expected.items[0].point_id

    @pytest.mark.asyncio
    async def test_get_returns_none_below_threshold(self, mock_qdrant_client: AsyncMock) -> None:
        """get() returns None when the best score is below threshold."""
        point = MagicMock()
        point.score = 0.5
        point.payload = {"results_json": _serialize_results(_make_results()), "collection": "docs"}
        mock_qdrant_client.query_points.return_value = MagicMock(points=[point])

        client = CacheClient(config=_make_config(threshold=0.9), vector_size=512)
        await client.initialize()

        result = await client.get(query_vector=[0.1, 0.2], collection="docs")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_catches_exceptions(self, mock_qdrant_client: AsyncMock) -> None:
        """get() returns None on internal failure instead of propagating."""
        mock_qdrant_client.query_points.side_effect = RuntimeError("network")
        client = CacheClient(config=_make_config(), vector_size=512)
        await client.initialize()

        result = await client.get(query_vector=[0.1, 0.2], collection="docs")
        assert result is None


# =============================================================================
# Cache Put Tests
# =============================================================================


class TestCacheClientPut:
    """Test suite for CacheClient.put()."""

    @pytest.mark.asyncio
    async def test_put_skips_when_unhealthy(self) -> None:
        """put() is a no-op when the cache is unhealthy."""
        with patch("clients.cache.AsyncQdrantClient"):
            client = CacheClient(config=_make_config(), vector_size=512)
            await client.put(query_vector=[0.1], results=_make_results(), collection="docs")

    @pytest.mark.asyncio
    async def test_put_upserts_point(self, mock_qdrant_client: AsyncMock) -> None:
        """put() upserts a point with the serialized results payload."""
        mock_qdrant_client.count.return_value = MagicMock(count=0)
        client = CacheClient(config=_make_config(max_items=10), vector_size=512)
        await client.initialize()

        await client.put(query_vector=[0.1, 0.2], results=_make_results(), collection="docs")

        mock_qdrant_client.upsert.assert_awaited_once()
        call_kwargs = mock_qdrant_client.upsert.call_args
        points = call_kwargs.kwargs.get("points") or call_kwargs[1].get("points")
        assert len(points) == 1
        assert "results_json" in points[0].payload
        assert points[0].payload["collection"] == "docs"

    @pytest.mark.asyncio
    async def test_put_evicts_when_at_capacity(self, mock_qdrant_client: AsyncMock) -> None:
        """put() evicts oldest entries when count >= max_items."""
        mock_qdrant_client.count.return_value = MagicMock(count=5)
        old_point = MagicMock()
        old_point.id = "oldest-id"
        mock_qdrant_client.scroll.return_value = ([old_point], None)

        client = CacheClient(config=_make_config(max_items=5), vector_size=512)
        await client.initialize()

        await client.put(query_vector=[0.1], results=_make_results(), collection="docs")

        mock_qdrant_client.scroll.assert_awaited_once()
        mock_qdrant_client.delete.assert_awaited_once()
        mock_qdrant_client.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_put_catches_exceptions(self, mock_qdrant_client: AsyncMock) -> None:
        """put() logs and swallows internal failures."""
        mock_qdrant_client.count.side_effect = RuntimeError("disk")
        client = CacheClient(config=_make_config(), vector_size=512)
        await client.initialize()

        await client.put(query_vector=[0.1], results=_make_results(), collection="docs")


# =============================================================================
# Close Tests
# =============================================================================


class TestCacheClientClose:
    """Test suite for CacheClient.close()."""

    @pytest.mark.asyncio
    async def test_close_delegates_to_inner_client(self, mock_qdrant_client: AsyncMock) -> None:
        """close() awaits the underlying AsyncQdrantClient.close()."""
        client = CacheClient(config=_make_config(), vector_size=512)
        await client.close()

        mock_qdrant_client.close.assert_awaited_once()


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Test round-trip serialization of SearchResults."""

    def test_round_trip(self) -> None:
        """Serialize then deserialize produces identical SearchResults."""
        original = _make_results(3)
        payload = {"results_json": _serialize_results(original)}
        restored = _deserialize_results(payload)

        assert restored.total == original.total
        assert len(restored.items) == len(original.items)
        for orig, rest in zip(original.items, restored.items):
            assert rest.point_id == orig.point_id
            assert rest.score == orig.score
            assert rest.payload == orig.payload
