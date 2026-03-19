"""Integration tests for CacheClient.

These tests verify the semantic cache works correctly with a real in-memory
Qdrant instance. Unlike unit tests that test individual methods in isolation,
these exercise multi-step workflows, realistic data shapes, and edge cases
that emerge only when all components work together.

No Docker container is required — the CacheClient uses Qdrant's in-memory
backend (``location=":memory:"``).

## Test Categories

1. Happy Path — Store and retrieve with realistic data
2. Multi-Operation Workflows — Sequential operations, re-population after invalidation
3. Per-Collection Isolation — Cross-collection boundary checks
4. Capacity and Eviction — Behavior at and beyond max_entries_per_collection
5. Serialization Round-Trip — Complex payload shapes survive cache storage

## Running Tests

```bash
# Run cache integration tests only
pytest tests/integration/clients/test_cache.py -v

# Run with specific test class
pytest tests/integration/clients/test_cache.py::TestHappyPath -v
```
"""

import pytest

from clients.cache import CacheClient
from config import SemanticCacheConfig
from core.models import SearchResultItem, SearchResults


# =============================================================================
# Helpers
# =============================================================================


def _make_config(
    threshold: float = 0.95,
    max_entries: int = 100,
    timeout: int = 5,
) -> SemanticCacheConfig:
    return SemanticCacheConfig(
        enabled=True,
        similarity_threshold=threshold,
        max_entries_per_collection=max_entries,
        timeout_s=timeout,
    )


def _make_vector(dim: int = 128, seed: float = 1.0) -> list[float]:
    """Create a deterministic vector with varying direction based on seed.

    Uses alternating signs seeded by the seed value to ensure different seeds
    produce vectors that point in genuinely different directions (not just
    different magnitudes), so cosine similarity between different seeds is low.
    """
    import math

    return [math.sin(seed * (i + 1)) for i in range(dim)]


def _make_image_results(n: int = 5) -> SearchResults:
    """Create realistic image search results with full payload."""
    items = [
        SearchResultItem(
            point_id=f"img-{i:04d}",
            score=round(0.95 - i * 0.02, 4),
            payload={
                "s3_key": f"images/photo-{i:04d}.jpg",
                "s3_uri": f"s3://pipeline/images/photo-{i:04d}.jpg",
                "format": "jpeg",
                "width": 1920,
                "height": 1080,
                "thumbnail_key": f"thumbnails/photo-{i:04d}.jpg",
            },
        )
        for i in range(n)
    ]
    return SearchResults(items=items, total=n)


# =============================================================================
# Happy Path
# =============================================================================


@pytest.mark.integration
class TestHappyPath:
    """Basic store → lookup → verify workflows."""

    @pytest.mark.asyncio
    async def test_store_and_lookup_with_image_payloads(self) -> None:
        """Store realistic image search results and verify full round-trip.

        **Why this test is important:**
          - Validates that complex payloads (nested dicts with mixed types)
            survive JSON serialization and deserialization.
          - Exercises the full CacheClient pipeline: store → lookup → deserialize.

        **What it tests:**
          - All SearchResultItem fields are preserved (point_id, score, payload).
          - Nested payload keys (s3_key, width, height, etc.) are intact.
          - Total count is preserved.
        """
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            results = _make_image_results(5)
            vec = _make_vector()

            await client.store(
                collection="photos",
                query_vector=vec,
                query_text="sunset over ocean",
                results=results,
                limit=10,
            )

            hit = await client.lookup(collection="photos", query_vector=vec, limit=10)
            assert hit is not None
            assert hit.total == 5
            assert len(hit.items) == 5

            # Verify payload fidelity
            for i, item in enumerate(hit.items):
                assert item.point_id == f"img-{i:04d}"
                assert item.payload["s3_key"] == f"images/photo-{i:04d}.jpg"
                assert item.payload["format"] == "jpeg"
                assert item.payload["width"] == 1920
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_size_reflects_stored_entries(self) -> None:
        """size() returns the correct count after multiple stores.

        **Why this test is important:**
          - Validates that size() accurately reflects cache state.
          - Ensures distinct vectors create separate cache entries.

        **What it tests:**
          - size() returns 0 before any stores.
          - size() increments with each store.
        """
        client = CacheClient(config=_make_config())
        try:
            assert await client.size("docs") == 0

            for i in range(3):
                await client.store(
                    collection="docs",
                    query_vector=_make_vector(seed=float(i + 1)),
                    query_text=f"query-{i}",
                    results=_make_image_results(1),
                    limit=10,
                )

            assert await client.size("docs") == 3
        finally:
            await client.close()


# =============================================================================
# Multi-Operation Workflows
# =============================================================================


@pytest.mark.integration
class TestMultiOperationWorkflows:
    """Tests that exercise sequences of cache operations."""

    @pytest.mark.asyncio
    async def test_invalidate_then_repopulate(self) -> None:
        """Cache can be invalidated and then repopulated from scratch.

        **Why this test is important:**
          - Validates that invalidation fully cleans up internal state.
          - Ensures the cache can be rebuilt after invalidation.

        **What it tests:**
          - After invalidation, lookup returns None.
          - After re-store, lookup finds the new entry.
          - size() reflects the new state.
        """
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            vec = _make_vector()
            original = _make_image_results(2)

            # Populate
            await client.store(
                collection="docs",
                query_vector=vec,
                query_text="original",
                results=original,
                limit=10,
            )
            assert await client.size("docs") == 1

            # Invalidate
            await client.invalidate(collection="docs")
            assert await client.size("docs") == 0
            assert await client.lookup(collection="docs", query_vector=vec, limit=10) is None

            # Repopulate
            updated = _make_image_results(3)
            await client.store(
                collection="docs",
                query_vector=vec,
                query_text="updated",
                results=updated,
                limit=10,
            )

            hit = await client.lookup(collection="docs", query_vector=vec, limit=10)
            assert hit is not None
            assert hit.total == 3
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_multiple_queries_same_collection(self) -> None:
        """Multiple different queries can coexist in the same collection.

        **Why this test is important:**
          - Validates that the cache can store multiple query→result mappings.
          - Each stored query returns its own cached results.

        **What it tests:**
          - Two different vectors stored in same collection.
          - Each vector's lookup returns its own results, not the other's.
        """
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            vec_a = _make_vector(seed=1.0)
            vec_b = _make_vector(seed=100.0)  # Very different vector
            results_a = SearchResults(
                items=[SearchResultItem(point_id="a-1", score=0.9, payload={"tag": "alpha"})],
                total=1,
            )
            results_b = SearchResults(
                items=[SearchResultItem(point_id="b-1", score=0.8, payload={"tag": "beta"})],
                total=1,
            )

            await client.store(
                collection="multi",
                query_vector=vec_a,
                query_text="query alpha",
                results=results_a,
                limit=10,
            )
            await client.store(
                collection="multi",
                query_vector=vec_b,
                query_text="query beta",
                results=results_b,
                limit=10,
            )

            hit_a = await client.lookup(collection="multi", query_vector=vec_a, limit=10)
            hit_b = await client.lookup(collection="multi", query_vector=vec_b, limit=10)

            assert hit_a is not None
            assert hit_a.items[0].payload["tag"] == "alpha"
            assert hit_b is not None
            assert hit_b.items[0].payload["tag"] == "beta"
        finally:
            await client.close()


# =============================================================================
# Per-Collection Isolation
# =============================================================================


@pytest.mark.integration
class TestPerCollectionIsolation:
    """Tests that cache collections are fully isolated from each other."""

    @pytest.mark.asyncio
    async def test_invalidate_one_collection_preserves_others(self) -> None:
        """Invalidating collection A does not affect collection B.

        **Why this test is important:**
          - Per-collection isolation is a core design requirement.
          - Selective invalidation must not cause data loss in other collections.

        **What it tests:**
          - After invalidating "alpha", "beta" cache remains intact.
          - size() and lookup() reflect correct state for each.
        """
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            vec = _make_vector()
            for col in ("alpha", "beta"):
                await client.store(
                    collection=col,
                    query_vector=vec,
                    query_text="test",
                    results=_make_image_results(1),
                    limit=10,
                )

            await client.invalidate(collection="alpha")

            assert await client.size("alpha") == 0
            assert await client.size("beta") == 1

            hit = await client.lookup(collection="beta", query_vector=vec, limit=10)
            assert hit is not None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_size_is_per_collection(self) -> None:
        """size() reports counts independently per collection.

        **Why this test is important:**
          - Ensures entries stored in collection A are not counted in collection B.

        **What it tests:**
          - Storing 3 entries in "alpha" and 1 in "beta" shows correct counts.
        """
        client = CacheClient(config=_make_config())
        try:
            for i in range(3):
                await client.store(
                    collection="alpha",
                    query_vector=_make_vector(seed=float(i + 1)),
                    query_text=f"alpha-{i}",
                    results=_make_image_results(1),
                    limit=10,
                )
            await client.store(
                collection="beta",
                query_vector=_make_vector(seed=99.0),
                query_text="beta-0",
                results=_make_image_results(1),
                limit=10,
            )

            assert await client.size("alpha") == 3
            assert await client.size("beta") == 1
        finally:
            await client.close()


# =============================================================================
# Capacity and Eviction
# =============================================================================


@pytest.mark.integration
class TestCapacityAndEviction:
    """Tests for eviction behavior at capacity limits."""

    @pytest.mark.asyncio
    async def test_eviction_maintains_max_entries(self) -> None:
        """Cache never exceeds max_entries_per_collection after stores.

        **Why this test is important:**
          - Memory-bounded cache must enforce its capacity limit.
          - Eviction must actually remove entries, not just skip the store.

        **What it tests:**
          - After storing more entries than max, size stays at or below max.
        """
        max_entries = 5
        client = CacheClient(config=_make_config(max_entries=max_entries))
        try:
            for i in range(max_entries + 10):
                await client.store(
                    collection="bounded",
                    query_vector=_make_vector(seed=float(i + 1)),
                    query_text=f"q-{i}",
                    results=_make_image_results(1),
                    limit=10,
                )

            count = await client.size("bounded")
            assert count <= max_entries
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_eviction_is_per_collection(self) -> None:
        """Eviction in one collection does not affect another.

        **Why this test is important:**
          - Each collection has its own capacity limit.
          - Overflowing collection A must not evict from collection B.

        **What it tests:**
          - Fill collection A past capacity (max=2), B stays at 1.
        """
        client = CacheClient(config=_make_config(max_entries=2))
        try:
            # Store 1 in beta
            await client.store(
                collection="beta",
                query_vector=_make_vector(seed=50.0),
                query_text="beta-0",
                results=_make_image_results(1),
                limit=10,
            )

            # Overflow alpha
            for i in range(5):
                await client.store(
                    collection="alpha",
                    query_vector=_make_vector(seed=float(i + 1)),
                    query_text=f"alpha-{i}",
                    results=_make_image_results(1),
                    limit=10,
                )

            assert await client.size("alpha") <= 2
            assert await client.size("beta") == 1
        finally:
            await client.close()


# =============================================================================
# Serialization Round-Trip
# =============================================================================


@pytest.mark.integration
class TestSerializationRoundTrip:
    """Tests that complex data survives the store→lookup JSON round-trip."""

    @pytest.mark.asyncio
    async def test_payload_with_none_values(self) -> None:
        """Payload fields with None values survive round-trip.

        **Why this test is important:**
          - Real image results may have optional fields set to None
            (e.g. thumbnail_key, width, height for incomplete metadata).

        **What it tests:**
          - None values in payload are preserved after deserialization.
        """
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            results = SearchResults(
                items=[
                    SearchResultItem(
                        point_id="img-001",
                        score=0.92,
                        payload={
                            "s3_key": "images/test.jpg",
                            "format": "jpeg",
                            "width": None,
                            "height": None,
                            "thumbnail_key": None,
                        },
                    )
                ],
                total=1,
            )
            vec = _make_vector()

            await client.store(
                collection="nones",
                query_vector=vec,
                query_text="test none values",
                results=results,
                limit=10,
            )

            hit = await client.lookup(collection="nones", query_vector=vec, limit=10)
            assert hit is not None
            assert hit.items[0].payload["width"] is None
            assert hit.items[0].payload["thumbnail_key"] is None
            assert hit.items[0].payload["s3_key"] == "images/test.jpg"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_empty_results_round_trip(self) -> None:
        """Empty SearchResults (total=0, items=[]) survives round-trip.

        **Why this test is important:**
          - A query that returns no results should still be cacheable
            to avoid repeating the same empty search.

        **What it tests:**
          - SearchResults with 0 items can be stored and retrieved.
        """
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            results = SearchResults(items=[], total=0)
            vec = _make_vector()

            await client.store(
                collection="empty",
                query_vector=vec,
                query_text="query with no results",
                results=results,
                limit=10,
            )

            hit = await client.lookup(collection="empty", query_vector=vec, limit=10)
            assert hit is not None
            assert hit.total == 0
            assert len(hit.items) == 0
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_large_result_set_round_trip(self) -> None:
        """A large result set (50 items) survives round-trip.

        **Why this test is important:**
          - Real queries may return up to 100 results.
          - JSON serialization must handle larger payloads.

        **What it tests:**
          - 50 items stored and retrieved correctly.
          - All item data is preserved.
        """
        client = CacheClient(config=_make_config(threshold=0.90))
        try:
            results = _make_image_results(50)
            vec = _make_vector()

            await client.store(
                collection="large",
                query_vector=vec,
                query_text="large result set",
                results=results,
                limit=50,
            )

            hit = await client.lookup(collection="large", query_vector=vec, limit=50)
            assert hit is not None
            assert hit.total == 50
            assert len(hit.items) == 50
            assert hit.items[49].point_id == "img-0049"
        finally:
            await client.close()
