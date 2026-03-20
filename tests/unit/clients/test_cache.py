"""Unit tests for clients.cache module.

This file tests the CacheClient class which provides an in-memory
semantic cache backed by Qdrant.

Tests use a real in-memory Qdrant instance (no mocking) since the
in-memory backend is lightweight and fast.

# Test Coverage

The tests cover:
  - Lookup miss on empty cache
  - Store then lookup hit (exact vector)
  - Lookup miss below similarity threshold
  - Limit truncation on lookup
  - Eviction at max capacity
  - Invalidation clears cache
  - Per-collection isolation
  - Close does not raise
  - get_semantic_cache singleton behavior

# Running Tests

Run with: uv run pytest tests/unit/clients/test_cache.py -v
"""

from unittest.mock import patch

import pytest

from clients.cache import CacheClient, get_semantic_cache
from config import SemanticCacheConfig
from core.models import SearchResultItem, SearchResults


# =============================================================================
# Helpers
# =============================================================================


def _make_config(
    threshold: float = 0.95,
    max_entries: int = 1000,
    timeout: int = 5,
    invalidation_interval: int = 3600,
) -> SemanticCacheConfig:
    return SemanticCacheConfig(
        enabled=True,
        similarity_threshold=threshold,
        max_entries_per_collection=max_entries,
        timeout_s=timeout,
        invalidation_interval_seconds=invalidation_interval,
    )


def _make_results(n: int = 3) -> SearchResults:
    """Create a small SearchResults for testing."""
    items = [
        SearchResultItem(point_id=f"pt-{i}", score=0.9 - i * 0.05, payload={"key": f"val-{i}"})
        for i in range(n)
    ]
    return SearchResults(items=items, total=n)


def _make_vector(dim: int = 128, seed: float = 1.0) -> list[float]:
    """Create a deterministic vector with varying direction based on seed.

    Uses sin() to ensure different seeds produce vectors pointing in
    genuinely different directions (not just different magnitudes).
    """
    import math

    return [math.sin(seed * (i + 1)) for i in range(dim)]


# =============================================================================
# Tests
# =============================================================================


class TestCacheClientLookup:
    """Tests for CacheClient.lookup()."""

    @pytest.mark.asyncio
    async def test_lookup_miss_on_empty_cache(self) -> None:
        """lookup returns None when no cache collection exists."""
        client = CacheClient(config=_make_config())
        try:
            result = await client.lookup(
                collection="documents",
                query_vector=_make_vector(),
                limit=10,
            )
            assert result is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_lookup_miss_below_threshold(self) -> None:
        """Store result, lookup with dissimilar vector, get None."""
        client = CacheClient(config=_make_config(threshold=0.99))
        try:
            results = _make_results()
            await client.store(
                collection="docs",
                query_vector=_make_vector(seed=1.0),
                query_text="sunset",
                results=results,
                limit=10,
            )

            # Use a very different vector to get a low similarity score
            dissimilar = _make_vector(seed=-1.0)
            hit = await client.lookup(collection="docs", query_vector=dissimilar, limit=10)
            assert hit is None
        finally:
            await client.close()


class TestCacheClientStore:
    """Tests for CacheClient.store() and round-trip with lookup()."""

    @pytest.mark.asyncio
    async def test_store_then_lookup_hit(self) -> None:
        """Store result, lookup with same vector, get exact match."""
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            results = _make_results(2)
            vec = _make_vector()

            await client.store(
                collection="docs",
                query_vector=vec,
                query_text="test query",
                results=results,
                limit=10,
            )

            hit = await client.lookup(collection="docs", query_vector=vec, limit=10)
            assert hit is not None
            assert hit.total == results.total
            assert len(hit.items) == len(results.items)
            assert hit.items[0].point_id == results.items[0].point_id
            assert hit.items[0].score == results.items[0].score
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_limit_truncation(self) -> None:
        """Store 10 results, lookup with limit=5, get 5 items."""
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            results = _make_results(10)
            vec = _make_vector()

            await client.store(
                collection="docs",
                query_vector=vec,
                query_text="query",
                results=results,
                limit=10,
            )

            hit = await client.lookup(collection="docs", query_vector=vec, limit=5)
            assert hit is not None
            assert len(hit.items) == 5
            assert hit.total == 10  # Original total preserved
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_eviction_at_max_capacity(self) -> None:
        """Fill to max, store one more, verify count maintained."""
        max_entries = 3
        client = CacheClient(config=_make_config(max_entries=max_entries))
        try:
            results = _make_results(1)
            for i in range(max_entries + 2):
                vec = _make_vector(seed=float(i + 1))
                await client.store(
                    collection="docs",
                    query_vector=vec,
                    query_text=f"query-{i}",
                    results=results,
                    limit=10,
                )

            count = await client.size("docs")
            assert count <= max_entries
        finally:
            await client.close()


class TestCacheClientInvalidate:
    """Tests for CacheClient.invalidate()."""

    @pytest.mark.asyncio
    async def test_invalidation_clears_cache(self) -> None:
        """Store entries, invalidate, lookup returns None."""
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            vec = _make_vector()
            await client.store(
                collection="docs",
                query_vector=vec,
                query_text="test",
                results=_make_results(),
                limit=10,
            )
            assert await client.size("docs") > 0

            await client.invalidate(collection="docs")

            assert await client.size("docs") == 0
            hit = await client.lookup(collection="docs", query_vector=vec, limit=10)
            assert hit is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_invalidate_all_collections(self) -> None:
        """Invalidate without collection clears all known caches."""
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            vec = _make_vector()
            for col in ("alpha", "beta"):
                await client.store(
                    collection=col,
                    query_vector=vec,
                    query_text="test",
                    results=_make_results(),
                    limit=10,
                )

            await client.invalidate()

            assert await client.size("alpha") == 0
            assert await client.size("beta") == 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_lookup_triggers_periodic_invalidation(self) -> None:
        """lookup() invalidates and returns None when the interval has elapsed."""
        client = CacheClient(config=_make_config(threshold=0.90, invalidation_interval=1))
        try:
            vec = _make_vector()
            await client.store(
                collection="docs",
                query_vector=vec,
                query_text="test",
                results=_make_results(),
                limit=10,
            )
            assert await client.size("docs") > 0

            # Simulate the interval having elapsed
            client._last_invalidation -= 2  # push 2 seconds into the past

            hit = await client.lookup(collection="docs", query_vector=vec, limit=10)
            assert hit is None
            assert await client.size("docs") == 0
        finally:
            await client.close()


class TestCacheClientIsolation:
    """Tests for per-collection isolation."""

    @pytest.mark.asyncio
    async def test_per_collection_isolation(self) -> None:
        """Store in collection A, lookup in collection B, get None."""
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            vec = _make_vector()
            await client.store(
                collection="alpha",
                query_vector=vec,
                query_text="test",
                results=_make_results(),
                limit=10,
            )

            hit = await client.lookup(collection="beta", query_vector=vec, limit=10)
            assert hit is None

            # But same vector in alpha should hit
            hit_alpha = await client.lookup(collection="alpha", query_vector=vec, limit=10)
            assert hit_alpha is not None
        finally:
            await client.close()


class TestCacheClientClose:
    """Tests for CacheClient.close()."""

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self) -> None:
        """Verify close doesn't raise."""
        client = CacheClient(config=_make_config())
        await client.close()  # Should not raise


class TestGetSemanticCache:
    """Tests for the module-level get_semantic_cache() singleton."""

    def test_get_semantic_cache_returns_none_when_disabled(self) -> None:
        """Disabled config returns None."""
        import clients.cache as cache_module

        # Reset singleton state
        cache_module._cache_initialized = False
        cache_module._cache_instance = None
        try:
            with patch.dict(
                "os.environ",
                {"SEMANTIC_CACHE_ENABLED": "false"},
                clear=False,
            ):
                result = get_semantic_cache()
                assert result is None
        finally:
            cache_module._cache_initialized = False
            cache_module._cache_instance = None

    def test_get_semantic_cache_returns_singleton(self) -> None:
        """Repeated calls return same instance."""
        import clients.cache as cache_module

        cache_module._cache_initialized = False
        cache_module._cache_instance = None
        try:
            with patch.dict(
                "os.environ",
                {"SEMANTIC_CACHE_ENABLED": "true"},
                clear=False,
            ):
                first = get_semantic_cache()
                second = get_semantic_cache()
                assert first is not None
                assert first is second
        finally:
            cache_module._cache_initialized = False
            cache_module._cache_instance = None
