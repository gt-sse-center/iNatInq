"""E2E tests for image search.

Tests semantic image search against the ingested collection from the
ingested_collection fixture. Verifies response schema, scores, and error handling.
"""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.helpers import api_url


@pytest.mark.e2e
def test_search_happy_path(ingested_collection: str):
    """Test happy-path image search with valid query.

    Searches the ingested collection (5 synthetic images) with a text query
    and verifies results come back with correct schema and valid scores.
    """
    # Perform search with text query
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "nature",
            "collection": ingested_collection,
            "limit": 5,
        },
        timeout=30.0,  # First search initializes CLIP/Qdrant clients - may be slow
    )

    # Should return 200 OK
    assert response.status_code == 200

    # Parse response JSON
    data = response.json()

    # Verify response schema
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0  # Should find at least 1 match

    # Verify each result has correct schema
    for result in data["results"]:
        assert "s3_key" in result
        assert "score" in result
        assert isinstance(result["s3_key"], str)
        assert isinstance(result["score"], (int, float))

        # Scores should be normalized between 0.0 and 1.0
        assert 0.0 <= result["score"] <= 1.0


@pytest.mark.e2e
def test_search_returns_all_results(ingested_collection: str):
    """Test that search returns all 5 ingested images when limit is high enough."""
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "colorful gradient pattern",
            "collection": ingested_collection,
            "limit": 10,  # Higher than the 5 images we ingested
        },
        timeout=10.0,
    )

    assert response.status_code == 200
    data = response.json()

    # Should return all 5 vectors
    assert len(data["results"]) == 5


@pytest.mark.e2e
def test_search_respects_limit(ingested_collection: str):
    """Test that search respects the limit parameter."""
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "test",
            "collection": ingested_collection,
            "limit": 2,
        },
        timeout=10.0,
    )

    assert response.status_code == 200
    data = response.json()

    # Should return at most 2 results
    assert len(data["results"]) <= 2


@pytest.mark.e2e
def test_search_with_empty_query(docker_compose_stack: None):
    """Test that search rejects empty query string."""
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "",
            "collection": "test_collection",
            "limit": 5,
        },
        timeout=5.0,
    )

    # Should reject with 400 Bad Request or 422 Validation Error
    assert response.status_code in {400, 422}


@pytest.mark.e2e
def test_search_with_missing_query(docker_compose_stack: None):
    """Test that search requires query parameter."""
    response = httpx.get(
        api_url("/search/images"),
        params={
            "collection": "test_collection",
            "limit": 5,
        },
        timeout=5.0,
    )

    # Should reject with 400 or 422
    assert response.status_code in {400, 422}


@pytest.mark.e2e
def test_search_with_nonexistent_collection(docker_compose_stack: None):
    """Test that search handles nonexistent collection gracefully."""
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "nature",
            "collection": "nonexistent_collection_xyz",
            "limit": 5,
        },
        timeout=10.0,
    )

    # UpstreamError from Qdrant client maps to HTTP 502 via ExceptionHandlerMiddleware
    assert response.status_code in {404, 500, 502}

    # Response should be JSON with error information
    data = response.json()
    assert "error" in data or "message" in data or "detail" in data


@pytest.mark.e2e
def test_search_initializes_circuit_breakers(ingested_collection: str):
    """Test that search request initializes CLIP and Qdrant circuit breakers.

    This is important for Task 5 and Task 7 - metrics and resilience tests
    need circuit breaker metrics to be available, which only happens after
    the first search request initializes the clients lazily.
    """
    # Make a search request to initialize circuit breakers
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "test query to initialize circuit breakers",
            "collection": ingested_collection,
            "limit": 1,
        },
        timeout=30.0,
    )

    assert response.status_code == 200

    # Now fetch /metrics and verify circuit breaker metrics exist
    metrics_response = httpx.get(api_url("/metrics"), timeout=5.0)
    assert metrics_response.status_code == 200

    metrics_text = metrics_response.text

    # Circuit breaker metrics should now be present
    # (they're only created after first service call)
    assert "inatinq_circuit_breaker_state" in metrics_text
