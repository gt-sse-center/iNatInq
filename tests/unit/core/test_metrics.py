"""Unit tests for core.metrics module.

This file tests the Prometheus metrics registry which provides a centralized
definition of all custom metrics for the iNatInq codebase.

# Test Coverage

The tests cover:
  - Metric object definitions: Importability, correct types, registration in REGISTRY
  - Bucket profiles: Correct values for FAST, SLOW, and RESULT_COUNT buckets
  - Duplicate detection: No metric name collisions
  - Registry integration: Metrics appear in prometheus_client.REGISTRY

# Test Structure

Tests use pytest class-based organization with descriptive docstrings.
Each test documents "Why important" and "What it tests".

# Running Tests

Run with: uv run pytest tests/unit/core/test_metrics.py
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import aiobreaker
import prometheus_client
import pybreaker
import requests.exceptions
from foundation.exceptions import UpstreamError


class TestMetricDefinitions:
    """Test suite for metric object definitions and bucket profiles."""

    def test_metric_objects_are_importable(self) -> None:
        """Test that all 13 metric objects can be imported from foundation.metrics.

        **Why this test is important:**
          - Verifies the module structure is correct
          - Ensures all metric names are defined
          - Critical for downstream consumers

        **What it tests:**
          - All 13 metric objects exist and are importable
          - Bucket profile tuples are importable
          - classify_error function is importable
        """
        # Arrange & Act
        from core.metrics import (
            CLIENT_ERRORS_TOTAL,
            CLIENT_REQUEST_DURATION,
            CLIENT_REQUEST_TOTAL,
            CIRCUIT_BREAKER_STATE,
            CIRCUIT_BREAKER_TRANSITIONS,
            FAST_BUCKETS,
            INGESTION_BATCH_DURATION,
            INGESTION_CHECKPOINT_SAVES,
            INGESTION_DOCS_PROCESSED,
            RESULT_COUNT_BUCKETS,
            RETRY_ATTEMPTS_TOTAL,
            RETRY_EXHAUSTIONS_TOTAL,
            SEARCH_EMBEDDING_DURATION,
            SEARCH_RESULT_COUNT,
            SEARCH_VECTOR_QUERY_DURATION,
            SLOW_BUCKETS,
            classify_error,
        )

        # Assert - if imports succeed, test passes
        assert CLIENT_REQUEST_DURATION is not None
        assert CLIENT_REQUEST_TOTAL is not None
        assert CLIENT_ERRORS_TOTAL is not None
        assert CIRCUIT_BREAKER_STATE is not None
        assert CIRCUIT_BREAKER_TRANSITIONS is not None
        assert RETRY_ATTEMPTS_TOTAL is not None
        assert RETRY_EXHAUSTIONS_TOTAL is not None
        assert SEARCH_EMBEDDING_DURATION is not None
        assert SEARCH_VECTOR_QUERY_DURATION is not None
        assert SEARCH_RESULT_COUNT is not None
        assert INGESTION_DOCS_PROCESSED is not None
        assert INGESTION_BATCH_DURATION is not None
        assert INGESTION_CHECKPOINT_SAVES is not None
        assert FAST_BUCKETS is not None
        assert SLOW_BUCKETS is not None
        assert RESULT_COUNT_BUCKETS is not None
        assert classify_error is not None

    def test_metric_label_names(self) -> None:
        """Test that each metric has the correct label names.

        **Why this test is important:**
          - Typos in label names silently produce incorrect metric cardinality
          - Label names are part of the metric contract consumed by dashboards
          - Prevents drift from the plan's metric inventory specification

        **What it tests:**
          - Each metric's labelnames match the plan's specification exactly
        """
        # Arrange
        from core.metrics import (
            CLIENT_ERRORS_TOTAL,
            CLIENT_REQUEST_DURATION,
            CLIENT_REQUEST_TOTAL,
            CIRCUIT_BREAKER_STATE,
            CIRCUIT_BREAKER_TRANSITIONS,
            INGESTION_BATCH_DURATION,
            INGESTION_CHECKPOINT_SAVES,
            INGESTION_DOCS_PROCESSED,
            RETRY_ATTEMPTS_TOTAL,
            RETRY_EXHAUSTIONS_TOTAL,
            SEARCH_EMBEDDING_DURATION,
            SEARCH_RESULT_COUNT,
            SEARCH_VECTOR_QUERY_DURATION,
        )

        # Assert - verify label names match plan specification
        assert CLIENT_REQUEST_DURATION._labelnames == ("client", "operation", "status")  # pyright: ignore[reportPrivateUsage]
        assert CLIENT_REQUEST_TOTAL._labelnames == ("client", "operation", "status")  # pyright: ignore[reportPrivateUsage]
        assert CLIENT_ERRORS_TOTAL._labelnames == ("client", "operation", "error_type")  # pyright: ignore[reportPrivateUsage]
        assert CIRCUIT_BREAKER_STATE._labelnames == ("breaker",)  # pyright: ignore[reportPrivateUsage]
        assert CIRCUIT_BREAKER_TRANSITIONS._labelnames == ("breaker", "from_state", "to_state")  # pyright: ignore[reportPrivateUsage]
        assert RETRY_ATTEMPTS_TOTAL._labelnames == ("client", "operation", "outcome")  # pyright: ignore[reportPrivateUsage]
        assert RETRY_EXHAUSTIONS_TOTAL._labelnames == ("client", "operation")  # pyright: ignore[reportPrivateUsage]
        assert SEARCH_EMBEDDING_DURATION._labelnames == ("provider",)  # pyright: ignore[reportPrivateUsage]
        assert SEARCH_VECTOR_QUERY_DURATION._labelnames == ("provider", "collection")  # pyright: ignore[reportPrivateUsage]
        assert SEARCH_RESULT_COUNT._labelnames == ("collection",)  # pyright: ignore[reportPrivateUsage]
        assert INGESTION_DOCS_PROCESSED._labelnames == ("status", "pipeline")  # pyright: ignore[reportPrivateUsage]
        assert INGESTION_BATCH_DURATION._labelnames == ("pipeline",)  # pyright: ignore[reportPrivateUsage]
        assert INGESTION_CHECKPOINT_SAVES._labelnames == ("pipeline",)  # pyright: ignore[reportPrivateUsage]

    def test_histogram_bucket_profiles(self) -> None:
        """Test that the three bucket profiles have correct values.

        **Why this test is important:**
          - Bucket boundaries determine histogram precision
          - Incorrect buckets reduce metric usefulness
          - These profiles are used by 5 histogram metrics

        **What it tests:**
          - FAST_BUCKETS matches expected millisecond-scale values
          - SLOW_BUCKETS matches expected second-scale values
          - RESULT_COUNT_BUCKETS matches expected count ranges
        """
        # Arrange
        from core.metrics import FAST_BUCKETS, RESULT_COUNT_BUCKETS, SLOW_BUCKETS

        # Expected values from plan
        expected_fast = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        expected_slow = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
        expected_result_count = (0, 1, 5, 10, 25, 50, 100, 250, 500, 1000)

        # Assert
        assert expected_fast == FAST_BUCKETS, f"FAST_BUCKETS mismatch: {FAST_BUCKETS}"
        assert expected_slow == SLOW_BUCKETS, f"SLOW_BUCKETS mismatch: {SLOW_BUCKETS}"
        assert expected_result_count == RESULT_COUNT_BUCKETS, (
            f"RESULT_COUNT_BUCKETS mismatch: {RESULT_COUNT_BUCKETS}"
        )

    def test_default_registry_contains_metrics(self) -> None:
        """Test that all custom metrics appear in prometheus_client.REGISTRY.

        **Why this test is important:**
          - Metrics must register to be exported on /metrics endpoint
          - Verifies module import has the expected side effect
          - Ensures Instrumentator can discover custom metrics

        **What it tests:**
          - All 13 metrics with inatinq_ prefix are in REGISTRY
          - Filters out default collectors (python_info, python_gc_*)
        """
        # Arrange
        import core.metrics  # noqa: F401 - import to trigger registration

        # Act - collect metric family names from REGISTRY (not sample names)
        inatinq_names = set()
        for metric_family in prometheus_client.REGISTRY.collect():
            if metric_family.name.startswith("inatinq_"):
                inatinq_names.add(metric_family.name)

        # Expected metric names (Counter family names appear without _total suffix)
        expected_names = {
            "inatinq_client_request_duration_seconds",
            "inatinq_client_request",  # Counter strips _total
            "inatinq_client_errors",  # Counter strips _total
            "inatinq_circuit_breaker_state",
            "inatinq_circuit_breaker_transitions",  # Counter strips _total
            "inatinq_retry_attempts",  # Counter strips _total
            "inatinq_retry_exhaustions",  # Counter strips _total
            "inatinq_search_embedding_duration_seconds",
            "inatinq_search_vector_query_duration_seconds",
            "inatinq_search_result_count",
            "inatinq_ingestion_documents_processed",  # Counter strips _total
            "inatinq_ingestion_batch_duration_seconds",
            "inatinq_ingestion_checkpoint_saves",  # Counter strips _total
        }

        # Assert
        assert inatinq_names == expected_names, (
            f"REGISTRY mismatch. Missing: {expected_names - inatinq_names}, Extra: {inatinq_names - expected_names}"
        )


class TestClassifyError:
    """Test suite for the classify_error exception classification helper."""

    def test_classify_error_timeout(self) -> None:
        """Test that timeout exceptions are classified as 'timeout'.

        **Why this test is important:**
          - Timeout errors need distinct handling from connection errors
          - Enables operators to identify slow vs broken services
          - Critical for alert tuning (timeouts vs failures)

        **What it tests:**
          - builtin TimeoutError returns "timeout"
          - asyncio.TimeoutError returns "timeout" (same class in Python 3.11+)
          - requests.exceptions.Timeout returns "timeout" (requests library)
        """
        # Arrange
        from core.metrics import classify_error

        # Act & Assert
        assert classify_error(TimeoutError()) == "timeout"
        assert classify_error(TimeoutError()) == "timeout"  # asyncio.TimeoutError is TimeoutError in 3.11+
        assert classify_error(requests.exceptions.Timeout()) == "timeout"

    def test_classify_error_connection(self) -> None:
        """Test that connection exceptions are classified as 'connection'.

        **Why this test is important:**
          - Connection errors indicate network issues distinct from timeouts
          - Helps distinguish client-side vs server-side problems
          - Essential for network troubleshooting

        **What it tests:**
          - builtin ConnectionError returns "connection"
          - builtin OSError returns "connection"
          - requests.exceptions.ConnectionError returns "connection"
        """
        # Arrange
        from core.metrics import classify_error

        # Act & Assert
        assert classify_error(ConnectionError()) == "connection"
        assert classify_error(OSError()) == "connection"
        assert classify_error(requests.exceptions.ConnectionError()) == "connection"

    def test_classify_error_upstream(self) -> None:
        """Test that UpstreamError is classified as 'upstream'.

        **Why this test is important:**
          - UpstreamError is the application's wrapper for external service failures
          - Distinct from network errors (service responded but with error)
          - Enables tracking service-level vs infrastructure-level issues

        **What it tests:**
          - UpstreamError returns "upstream"
        """
        # Arrange
        from core.metrics import classify_error

        # Act & Assert
        assert classify_error(UpstreamError("service unavailable")) == "upstream"

    def test_classify_error_circuit_open(self) -> None:
        """Test that circuit breaker errors are classified as 'circuit_open'.

        **Why this test is important:**
          - Circuit breaker open state indicates repeated failures
          - Helps operators understand when fail-fast is triggered
          - Critical for capacity planning and incident response

        **What it tests:**
          - pybreaker.CircuitBreakerError returns "circuit_open"
          - aiobreaker.CircuitBreakerError returns "circuit_open"
          - Handles different constructor signatures (pybreaker vs aiobreaker)
        """
        # Arrange
        from core.metrics import classify_error

        # Create mock breaker for pybreaker (takes breaker instance)
        mock_breaker = MagicMock()

        # Act & Assert
        assert classify_error(pybreaker.CircuitBreakerError(mock_breaker)) == "circuit_open"
        # aiobreaker requires (message: str, reopen_time: datetime)
        assert (
            classify_error(aiobreaker.CircuitBreakerError("breaker open", datetime.now(tz=UTC)))
            == "circuit_open"
        )

    def test_classify_error_unknown_fallback(self) -> None:
        """Test that unrecognized exceptions are classified as 'unknown'.

        **Why this test is important:**
          - Ensures bounded cardinality for label values
          - Prevents metrics explosion from exotic exception types
          - Provides safe default for unexpected errors

        **What it tests:**
          - ValueError (arbitrary exception) returns "unknown"
          - Acts as catch-all for unmapped exception types
        """
        # Arrange
        from core.metrics import classify_error

        # Act & Assert
        assert classify_error(ValueError("some error")) == "unknown"
        assert classify_error(RuntimeError("unexpected")) == "unknown"

    def test_classify_error_timeout_not_misclassified_as_connection(self) -> None:
        """Test that TimeoutError is not misclassified due to OSError inheritance.

        **Why this test is important:**
          - CRITICAL regression guard for isinstance ordering bug
          - TimeoutError inherits from OSError in Python 3.11+
          - If OSError check comes before TimeoutError, timeouts get wrong label
          - This test fails if isinstance checks are reordered incorrectly

        **What it tests:**
          - TimeoutError explicitly returns "timeout", not "connection"
          - Guards against OSError check matching TimeoutError first
        """
        # Arrange
        from core.metrics import classify_error

        # Act
        result = classify_error(TimeoutError())

        # Assert - MUST be "timeout", not "connection"
        assert result == "timeout", (
            "TimeoutError must not be misclassified as 'connection' due to OSError inheritance"
        )
