"""E2E tests for Prometheus metrics observability.

Verifies that metrics are correctly emitted during real operations by scraping
/metrics before and after operations and asserting specific counters increased.

The pipeline uses an in-memory semantic cache that can cause searches to skip
Qdrant entirely on repeated similar queries.  Tests that assert Qdrant metrics
must bust the cache first so the search actually reaches the vector database.
"""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.helpers import api_url
from tests.e2e.metrics_helper import get_metric_value, metric_delta, scrape_metrics


@pytest.fixture(autouse=True)
def _bust_semantic_cache():
    """Bust the semantic cache before each metric test.

    Ensures searches reach Qdrant instead of returning cached results,
    so Qdrant-related metrics actually increment.
    """
    try:
        httpx.delete(api_url("/cache"), timeout=5.0)
    except httpx.HTTPStatusError:
        pass  # Cache may not be enabled; that's fine


@pytest.mark.e2e
def test_search_embedding_metrics(ingested_collection: str):
    """Test that search embedding metrics are emitted correctly.

    After a search request, verify the embedding duration histogram counter
    increased by exactly 1 for the CLIP provider.
    """
    # Use metric_delta context manager to measure before/after
    with metric_delta(
        "http://localhost:8000",
        "inatinq_search_embedding_duration_seconds_count",
        {"provider": "clip"},
    ) as delta:
        # Perform a search request
        response = httpx.get(
            api_url("/search/images"),
            params={
                "q": "test query for metrics",
                "collection": ingested_collection,
                "limit": 3,
            },
            timeout=10.0,
        )
        assert response.status_code == 200

    # Embedding duration counter should have increased by exactly 1
    assert delta.value == 1.0


@pytest.mark.e2e
def test_search_vector_query_metrics(ingested_collection: str):
    """Test that vector query duration metrics are emitted correctly."""
    with metric_delta(
        "http://localhost:8000",
        "inatinq_search_vector_query_duration_seconds_count",
        {"provider": "qdrant", "collection": ingested_collection},
    ) as delta:
        response = httpx.get(
            api_url("/search/images"),
            params={
                "q": "another test query",
                "collection": ingested_collection,
                "limit": 3,
            },
            timeout=10.0,
        )
        assert response.status_code == 200

    # Vector query duration counter should have increased by 1
    assert delta.value == 1.0


@pytest.mark.e2e
def test_search_result_count_metrics(ingested_collection: str):
    """Test that result count metrics are emitted correctly."""
    with metric_delta(
        "http://localhost:8000",
        "inatinq_search_result_count_count",
        {"collection": ingested_collection},
    ) as delta:
        response = httpx.get(
            api_url("/search/images"),
            params={
                "q": "metrics test",
                "collection": ingested_collection,
                "limit": 5,
            },
            timeout=10.0,
        )
        assert response.status_code == 200

    # Result count counter should have increased
    assert delta.value == 1.0


@pytest.mark.e2e
def test_client_request_metrics_clip(ingested_collection: str):
    """Test that CLIP client request metrics are emitted correctly."""
    with metric_delta(
        "http://localhost:8000",
        "inatinq_client_request_duration_seconds_count",
        {"client": "clip", "operation": "embed_text", "status": "success"},
    ) as delta:
        response = httpx.get(
            api_url("/search/images"),
            params={
                "q": "test",
                "collection": ingested_collection,
                "limit": 1,
            },
            timeout=10.0,
        )
        assert response.status_code == 200

    # CLIP embed client request counter should have increased
    assert delta.value >= 1.0  # May be >1 if retry logic fires


@pytest.mark.e2e
def test_client_request_metrics_qdrant(ingested_collection: str):
    """Test that Qdrant client request metrics are emitted correctly."""
    with metric_delta(
        "http://localhost:8000",
        "inatinq_client_request_duration_seconds_count",
        {"client": "qdrant", "operation": "search_async", "status": "success"},
    ) as delta:
        response = httpx.get(
            api_url("/search/images"),
            params={
                "q": "test",
                "collection": ingested_collection,
                "limit": 1,
            },
            timeout=10.0,
        )
        assert response.status_code == 200

    # Qdrant search client request counter should have increased
    assert delta.value >= 1.0


@pytest.mark.e2e
def test_circuit_breaker_state_metrics(ingested_collection: str):
    """Test that circuit breaker state metrics show breakers are closed.

    After a successful search (which exercises CLIP and Qdrant), circuit
    breaker state metrics should exist and show value 0 (closed).
    """
    # Perform a search to ensure breakers are initialized
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "test",
            "collection": ingested_collection,
            "limit": 1,
        },
        timeout=10.0,
    )
    assert response.status_code == 200

    # Scrape metrics
    metrics = scrape_metrics("http://localhost:8000")

    # Circuit breaker state for CLIP should be 0 (closed)
    clip_breaker_state = get_metric_value(metrics, "inatinq_circuit_breaker_state", {"breaker": "clip"})
    assert clip_breaker_state is not None
    assert clip_breaker_state == 0.0

    # Circuit breaker state for Qdrant should be 0 (closed)
    qdrant_breaker_state = get_metric_value(metrics, "inatinq_circuit_breaker_state", {"breaker": "qdrant"})
    assert qdrant_breaker_state is not None
    assert qdrant_breaker_state == 0.0


@pytest.mark.e2e
def test_http_request_metrics(docker_compose_stack: None):
    """Test that HTTP request duration metrics from instrumentator exist.

    The instrumentator automatically emits http_request_duration_seconds
    histogram for all HTTP requests. Verify at least one sample exists.
    """
    # Make a simple request to /healthz
    response = httpx.get(api_url("/healthz"), timeout=5.0)
    assert response.status_code == 200

    # Scrape metrics
    metrics = scrape_metrics("http://localhost:8000")

    # Should have http_request_duration_seconds_count samples
    assert "http_request_duration_seconds_count" in metrics
    assert len(metrics["http_request_duration_seconds_count"]) > 0

    # At least one sample should be for /healthz GET
    healthz_samples = [
        s
        for s in metrics["http_request_duration_seconds_count"]
        if s.labels.get("handler") == "/healthz" and s.labels.get("method") == "GET"
    ]
    assert len(healthz_samples) > 0


@pytest.mark.e2e
def test_all_search_metrics_together(ingested_collection: str):
    """Integration test: verify all search-related metrics increase together.

    This tests that a single search operation emits all expected metric types.
    """
    # Scrape metrics before
    before_metrics = scrape_metrics("http://localhost:8000")

    # Perform search
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "comprehensive metrics test",
            "collection": ingested_collection,
            "limit": 5,
        },
        timeout=10.0,
    )
    assert response.status_code == 200

    # Scrape metrics after
    after_metrics = scrape_metrics("http://localhost:8000")

    # Helper to compute delta for a metric
    def compute_delta(metric_name: str, labels: dict[str, str]) -> float:
        before = get_metric_value(before_metrics, metric_name, labels) or 0.0
        after = get_metric_value(after_metrics, metric_name, labels) or 0.0
        return after - before

    # Verify all key search metrics increased
    assert compute_delta("inatinq_search_embedding_duration_seconds_count", {"provider": "clip"}) == 1.0
    assert (
        compute_delta(
            "inatinq_search_vector_query_duration_seconds_count",
            {"provider": "qdrant", "collection": ingested_collection},
        )
        == 1.0
    )
    assert compute_delta("inatinq_search_result_count_count", {"collection": ingested_collection}) == 1.0

    # Client request metrics should have increased
    assert (
        compute_delta(
            "inatinq_client_request_duration_seconds_count",
            {"client": "clip", "operation": "embed_text", "status": "success"},
        )
        >= 1.0
    )
    assert (
        compute_delta(
            "inatinq_client_request_duration_seconds_count",
            {"client": "qdrant", "operation": "search_async", "status": "success"},
        )
        >= 1.0
    )
