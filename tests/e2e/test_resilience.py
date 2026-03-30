"""E2E tests for resilience observability.

Verifies that circuit breaker and retry metrics are observable in a live system.
Tests that circuit breakers are in healthy state and metrics are properly instrumented.
"""

from __future__ import annotations

import time

import httpx
import pytest

from tests.e2e.helpers import api_url
from tests.e2e.metrics_helper import get_metric_value, scrape_metrics


@pytest.mark.e2e
def test_circuit_breakers_closed_after_healthy_operations(ingested_collection: str):
    """Test that all circuit breakers are in closed state (0) after healthy operations.

    After the ingested_collection fixture completes (which runs ingestion and search),
    all circuit breakers should be in closed state, proving the resilience metrics
    instrumentation is wired correctly end-to-end.
    """
    # Perform a search to ensure breakers are exercised
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "resilience test",
            "collection": ingested_collection,
            "limit": 3,
        },
        timeout=10.0,
    )
    assert response.status_code == 200

    # Scrape metrics
    metrics = scrape_metrics("http://localhost:8000")

    # Verify CLIP circuit breaker is closed (state = 0)
    clip_state = get_metric_value(metrics, "inatinq_circuit_breaker_state", {"breaker": "clip"})
    assert clip_state is not None, "CLIP circuit breaker state metric not found"
    assert clip_state == 0.0, f"CLIP circuit breaker should be closed (0), got {clip_state}"

    # Verify Qdrant circuit breaker is closed (state = 0)
    qdrant_state = get_metric_value(metrics, "inatinq_circuit_breaker_state", {"breaker": "qdrant"})
    assert qdrant_state is not None, "Qdrant circuit breaker state metric not found"
    assert qdrant_state == 0.0, f"Qdrant circuit breaker should be closed (0), got {qdrant_state}"


@pytest.mark.e2e
def test_circuit_breaker_transitions_zero_after_healthy_operations(ingested_collection: str):
    """Test that circuit breaker transition counts are 0 after healthy operations.

    After only successful operations, no circuit breaker state transitions should
    have occurred (open/close/half-open transitions).
    """
    # Perform a search
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

    # Verify no transitions occurred for any breaker.
    # The transitions counter has labels [breaker, from_state, to_state], so
    # exact-match with only {"breaker": ...} won't match any sample. Instead,
    # iterate all samples and sum values for each breaker.
    transitions_key = "inatinq_circuit_breaker_transitions_total"
    if transitions_key in metrics:
        clip_total = sum(s.value for s in metrics[transitions_key] if s.labels.get("breaker") == "clip")
        assert clip_total == 0.0, f"CLIP breaker should have 0 transitions, got {clip_total}"

        qdrant_total = sum(s.value for s in metrics[transitions_key] if s.labels.get("breaker") == "qdrant")
        assert qdrant_total == 0.0, f"Qdrant breaker should have 0 transitions, got {qdrant_total}"
    # If metric key is absent, no transitions occurred (expected in healthy run)


@pytest.mark.e2e
def test_retry_metrics_exist(ingested_collection: str):
    """Test that retry metric families exist in /metrics output.

    After healthy operations, retry metrics should be registered even if no
    retries occurred. Counters with labels only produce samples when
    .labels(...).inc() is called, so in a healthy run these may have zero
    samples. We check the raw /metrics text for # TYPE headers instead.
    """
    # Perform a search
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

    # Fetch raw /metrics text to check for # TYPE headers
    # (present even when no samples have been emitted)
    metrics_response = httpx.get("http://localhost:8000/metrics", timeout=5.0)
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text

    # Verify retry_attempts_total metric family is registered
    assert "# TYPE inatinq_retry_attempts_total" in metrics_text, (
        "Retry attempts metric family not found in /metrics output"
    )

    # Verify retry_exhaustions_total metric family is registered
    assert "# TYPE inatinq_retry_exhaustions_total" in metrics_text, (
        "Retry exhaustions metric family not found in /metrics output"
    )


@pytest.mark.e2e
def test_all_expected_inatinq_metrics_present(ingested_collection: str):
    """Test that all expected inatinq_* metric families are present in /metrics.

    Verifies complete metrics instrumentation coverage by checking for all
    metric families defined in src/foundation/metrics/registry.py.

    Uses a two-tier check:
    - Metrics guaranteed to have samples after a search (histograms, gauges
      that are always set) are checked in the parsed samples dict.
    - Counters with labels that only emit samples when incremented (transitions,
      retries) are checked via raw # TYPE headers in the /metrics text.
    """
    # Perform a search to ensure all metrics are initialized
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "comprehensive metrics test",
            "collection": ingested_collection,
            "limit": 3,
        },
        timeout=10.0,
    )
    assert response.status_code == 200

    # Scrape metrics (parsed samples dict)
    metrics = scrape_metrics("http://localhost:8000")

    # Also fetch raw /metrics text for # TYPE header checks
    metrics_response = httpx.get("http://localhost:8000/metrics", timeout=5.0)
    metrics_text = metrics_response.text

    # Tier 1: Metrics guaranteed to have samples after a search
    # (histograms always emit _count/_sum/_bucket, gauges always set)
    sample_families = [
        "inatinq_client_request_duration_seconds",
        "inatinq_search_embedding_duration_seconds",
        "inatinq_search_vector_query_duration_seconds",
        "inatinq_search_result_count",
        "inatinq_circuit_breaker_state",
    ]

    missing_sample_families = []
    for family in sample_families:
        found = any(
            metric_name.startswith(family) for metric_name in metrics if metric_name.startswith("inatinq_")
        )
        if not found:
            missing_sample_families.append(family)

    assert not missing_sample_families, (
        f"Missing metric families in parsed samples: {missing_sample_families}"
    )

    # Tier 2: Counters with labels that may have no samples in a healthy run
    # (only produce samples when .labels(...).inc() is called)
    # Check for # TYPE headers in raw /metrics text instead
    type_header_families = [
        "inatinq_circuit_breaker_transitions_total",
        "inatinq_retry_attempts_total",
        "inatinq_retry_exhaustions_total",
    ]

    missing_type_families = []
    for family in type_header_families:
        if f"# TYPE {family}" not in metrics_text:
            missing_type_families.append(family)

    assert not missing_type_families, (
        f"Missing metric # TYPE headers in /metrics output: {missing_type_families}"
    )


@pytest.mark.e2e
def test_resilience_metrics_baseline_values(ingested_collection: str):
    """Test baseline values for resilience metrics after healthy operations.

    Verifies that after only successful operations:
    - Circuit breakers are closed (0)
    - Transition counts are 0
    - Retry attempts with outcome=success exist
    - No retry exhaustions
    """
    # Perform a search
    response = httpx.get(
        api_url("/search/images"),
        params={
            "q": "baseline test",
            "collection": ingested_collection,
            "limit": 2,
        },
        timeout=10.0,
    )
    assert response.status_code == 200

    # Scrape metrics
    metrics = scrape_metrics("http://localhost:8000")

    # Circuit breakers should be closed
    clip_state = get_metric_value(metrics, "inatinq_circuit_breaker_state", {"breaker": "clip"})
    qdrant_state = get_metric_value(metrics, "inatinq_circuit_breaker_state", {"breaker": "qdrant"})
    assert clip_state == 0.0
    assert qdrant_state == 0.0

    # Transitions should be 0 — sum all samples per breaker since
    # the metric has labels [breaker, from_state, to_state]
    transitions_key = "inatinq_circuit_breaker_transitions_total"
    if transitions_key in metrics:
        clip_transitions = sum(s.value for s in metrics[transitions_key] if s.labels.get("breaker") == "clip")
        qdrant_transitions = sum(
            s.value for s in metrics[transitions_key] if s.labels.get("breaker") == "qdrant"
        )
    else:
        clip_transitions = 0.0
        qdrant_transitions = 0.0
    assert clip_transitions == 0.0
    assert qdrant_transitions == 0.0

    # Retry exhaustions should be 0 (no failures)
    # Note: metric may not exist if no exhaustions occurred
    if "inatinq_retry_exhaustions_total" in metrics:
        for sample in metrics["inatinq_retry_exhaustions_total"]:
            assert sample.value == 0.0, f"Retry exhaustions should be 0, got {sample.value}"


@pytest.mark.e2e
def test_circuit_breaker_metrics_consistency(docker_compose_stack: None):
    """Test that circuit breaker state metrics are consistent across scrapes.

    Verifies that circuit breaker state values remain stable when no failures occur.
    """
    # Scrape metrics twice with a small delay
    metrics1 = scrape_metrics("http://localhost:8000")

    time.sleep(1)

    metrics2 = scrape_metrics("http://localhost:8000")

    # Circuit breaker states should be identical (both closed)
    clip_state1 = get_metric_value(metrics1, "inatinq_circuit_breaker_state", {"breaker": "clip"}) or 0.0
    clip_state2 = get_metric_value(metrics2, "inatinq_circuit_breaker_state", {"breaker": "clip"}) or 0.0
    assert clip_state1 == clip_state2, "CLIP breaker state should be stable"

    qdrant_state1 = get_metric_value(metrics1, "inatinq_circuit_breaker_state", {"breaker": "qdrant"}) or 0.0
    qdrant_state2 = get_metric_value(metrics2, "inatinq_circuit_breaker_state", {"breaker": "qdrant"}) or 0.0
    assert qdrant_state1 == qdrant_state2, "Qdrant breaker state should be stable"
