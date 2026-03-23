"""Unit tests for semantic cache Prometheus metrics.

Verifies that ``CacheClient`` emits the correct metrics for cache-internal
operations:
- Store duration (histogram)
- Cache size (gauge)
- Evictions and invalidations (counters)

Hit/miss counters and lookup duration are recorded at the service layer
(``ImageSearchService``); see ``tests/unit/services/test_search_service_metrics.py``.
"""

import pytest
from prometheus_client import REGISTRY

from clients.cache import CacheClient
from config import SemanticCacheConfig
from core.models import SearchResultItem, SearchResults


def _counter_value(metric_name: str, labels: dict[str, str]) -> float:
    """Return the value of a counter with the given labels, or 0.0 if absent."""
    return REGISTRY.get_sample_value(metric_name, labels) or 0.0


def _gauge_value(metric_name: str, labels: dict[str, str]) -> float:
    """Return the value of a gauge with the given labels, or 0.0 if absent."""
    return REGISTRY.get_sample_value(metric_name, labels) or 0.0


def _histogram_count(metric_name: str, labels: dict[str, str]) -> float:
    """Return the `_count` sample for a histogram label-set, or 0.0 if absent."""
    return REGISTRY.get_sample_value(f"{metric_name}_count", labels) or 0.0


@pytest.fixture
def cache_config() -> SemanticCacheConfig:
    """Create cache config with small capacity for testing."""
    return SemanticCacheConfig(
        enabled=True,
        similarity_threshold=0.95,
        max_entries_per_collection=2,  # Small capacity to test eviction
        invalidation_interval_seconds=999999,  # Very long interval (effectively disabled)
        timeout_s=5,
    )


@pytest.fixture
async def cache(cache_config: SemanticCacheConfig) -> CacheClient:
    """Create in-memory cache client for testing."""
    client = CacheClient(cache_config)
    yield client
    await client.close()


@pytest.fixture
def sample_results() -> SearchResults:
    """Sample search results for caching."""
    return SearchResults(
        items=[
            SearchResultItem(point_id="1", score=0.9, payload={"text": "cat"}),
            SearchResultItem(point_id="2", score=0.8, payload={"text": "dog"}),
        ],
        total=2,
    )


class TestCacheStoreDuration:
    """inatinq_cache_store_duration_seconds histogram records store time."""

    @pytest.mark.asyncio
    async def test_store_records_duration(self, cache: CacheClient, sample_results: SearchResults) -> None:
        """Store duration histogram count increments on each store."""
        collection = "wildlife"
        query_vector = [0.1, 0.1, 0.1]

        labels = {"collection": collection}
        before_count = _histogram_count("inatinq_cache_store_duration_seconds", labels)
        before_sum = REGISTRY.get_sample_value("inatinq_cache_store_duration_seconds_sum", labels) or 0.0

        # Store result
        await cache.store(collection, query_vector, "bird", sample_results, limit=10)

        # Verify histogram recorded
        after_count = _histogram_count("inatinq_cache_store_duration_seconds", labels)
        after_sum = REGISTRY.get_sample_value("inatinq_cache_store_duration_seconds_sum", labels) or 0.0

        assert after_count == before_count + 1.0
        assert after_sum > before_sum, "Duration sum should increase"


class TestCacheSizeGauge:
    """inatinq_cache_size gauge reflects current cache entry count."""

    @pytest.mark.asyncio
    async def test_cache_size_gauge_updated(self, cache: CacheClient, sample_results: SearchResults) -> None:
        """Cache size gauge updates after store and invalidate operations."""
        collection = "test_collection"
        labels = {"collection": collection}

        # Initial size should be 0
        initial_size = _gauge_value("inatinq_cache_size", labels)
        assert initial_size == 0.0, "Initial cache size should be 0"

        # Store first entry
        await cache.store(collection, [0.1, 0.2, 0.3], "query1", sample_results, limit=10)
        size_after_first = _gauge_value("inatinq_cache_size", labels)
        assert size_after_first == 1.0, "Size should be 1 after first store"

        # Store second entry
        await cache.store(collection, [0.4, 0.5, 0.6], "query2", sample_results, limit=10)
        size_after_second = _gauge_value("inatinq_cache_size", labels)
        assert size_after_second == 2.0, "Size should be 2 after second store"

        # Invalidate and check size drops to 0
        await cache.invalidate(collection)
        size_after_invalidate = _gauge_value("inatinq_cache_size", labels)
        assert size_after_invalidate == 0.0, "Size should be 0 after invalidation"


class TestCacheEvictionMetric:
    """inatinq_cache_evictions_total counter increments when capacity is exceeded."""

    @pytest.mark.asyncio
    async def test_eviction_increments_eviction_counter(
        self, cache: CacheClient, sample_results: SearchResults
    ) -> None:
        """Eviction counter increments when max capacity is exceeded."""
        collection = "limited"
        labels = {"collection": collection}

        before = _counter_value("inatinq_cache_evictions_total", labels)

        # Fill cache to capacity (max_entries_per_collection=2)
        await cache.store(collection, [0.1, 0.2, 0.3], "q1", sample_results, limit=10)
        await cache.store(collection, [0.4, 0.5, 0.6], "q2", sample_results, limit=10)

        # Store third entry, triggering eviction
        await cache.store(collection, [0.7, 0.8, 0.9], "q3", sample_results, limit=10)

        # Verify eviction counter incremented
        after = _counter_value("inatinq_cache_evictions_total", labels)
        assert after == before + 1.0, f"Eviction counter should increment by 1, got {after - before}"


class TestCacheInvalidationMetric:
    """inatinq_cache_invalidations_total counter increments on invalidate() calls."""

    @pytest.mark.asyncio
    async def test_invalidation_increments_invalidation_counter(
        self, cache: CacheClient, sample_results: SearchResults
    ) -> None:
        """Invalidation counter increments on each invalidate call."""
        collection = "temp"
        labels = {}  # No collection label for invalidations_total

        # Store some data
        await cache.store(collection, [0.1, 0.2, 0.3], "query", sample_results, limit=10)

        before = _counter_value("inatinq_cache_invalidations_total", labels)

        # Invalidate
        await cache.invalidate(collection)

        # Verify invalidation counter incremented
        after = _counter_value("inatinq_cache_invalidations_total", labels)
        assert after == before + 1.0, f"Invalidation counter should increment by 1, got {after - before}"
