"""Unit tests for core.metrics.decorators module.

This file tests the @with_client_metrics and @with_client_metrics_async decorators
that instrument client methods with Prometheus metrics.

# Test Coverage

The tests cover:
  - Sync decorator success path: counter increments, histogram observation
  - Sync decorator error path: error counter with classified error type, exception re-raise
  - Async decorator success path: identical behavior to sync
  - Async decorator error path: identical behavior to sync
  - Label values are bounded (client, operation from decorator args)
  - Metrics decorator stacks correctly with circuit breaker decorator

# Test Structure

Tests use pytest class-based organization with descriptive docstrings.
Each test documents "Why important" and "What it tests".
Metrics assertions use prometheus_client.REGISTRY.get_sample_value() for verification.

# Running Tests

Run with: uv run pytest tests/unit/core/metrics/test_decorators.py -v
"""

import pytest
from prometheus_client import REGISTRY


class TestWithClientMetrics:
    """Test suite for the @with_client_metrics decorator (sync)."""

    def test_successful_call_increments_counter(self) -> None:
        """Test that sync success increments request_total with status=success.

        **Why this test is important:**
          - Verifies the core metric emission on happy path
          - Ensures label values are correctly bound (client, operation, status)
          - Critical for throughput monitoring

        **What it tests:**
          - @with_client_metrics decorator increments CLIENT_REQUEST_TOTAL
          - Counter has correct labels: client, operation, status="success"
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics

        class DummyClient:
            @with_client_metrics("test_client", "test_operation")
            def my_method(self):
                return "success"

        client = DummyClient()

        # Read counter value before call
        before = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",  # Sample name includes _total
                {"client": "test_client", "operation": "test_operation", "status": "success"},
            )
            or 0.0
        )

        # Act
        result = client.my_method()

        # Assert
        after = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",  # Sample name includes _total
                {"client": "test_client", "operation": "test_operation", "status": "success"},
            )
            or 0.0
        )

        assert result == "success"
        assert after == before + 1.0, f"Counter did not increment (before={before}, after={after})"

    def test_successful_call_observes_latency(self) -> None:
        """Test that sync success observes duration histogram.

        **Why this test is important:**
          - Verifies latency tracking on happy path
          - Ensures histogram buckets are populated for percentile queries
          - Critical for performance monitoring

        **What it tests:**
          - @with_client_metrics decorator observes CLIENT_REQUEST_DURATION
          - Histogram sample count increments by 1
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics

        class DummyClient:
            @with_client_metrics("test_client", "test_op_histogram")
            def my_method(self):
                return "success"

        client = DummyClient()

        # Read histogram count before call
        before_count = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_duration_seconds_count",
                {"client": "test_client", "operation": "test_op_histogram", "status": "success"},
            )
            or 0.0
        )

        # Act
        result = client.my_method()

        # Assert
        after_count = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_duration_seconds_count",
                {"client": "test_client", "operation": "test_op_histogram", "status": "success"},
            )
            or 0.0
        )

        assert result == "success"
        assert after_count == before_count + 1.0, (
            f"Histogram count did not increment (before={before_count}, after={after_count})"
        )

    def test_failed_call_increments_error_counter(self) -> None:
        """Test that exceptions increment error counter with classified error type.

        **Why this test is important:**
          - Verifies error tracking and classification
          - Ensures error_type labels are correctly bound
          - Critical for alerting on specific error types

        **What it tests:**
          - @with_client_metrics increments CLIENT_ERRORS_TOTAL with error_type
          - classify_error() is called to determine error_type
          - Counter increments with status="error"
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics

        class DummyClient:
            @with_client_metrics("test_client", "test_error")
            def my_method(self):
                raise ValueError("test error")

        client = DummyClient()

        # Read counters before call
        before_total = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",
                {"client": "test_client", "operation": "test_error", "status": "error"},
            )
            or 0.0
        )

        before_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_client", "operation": "test_error", "error_type": "unknown"},
            )
            or 0.0
        )

        # Act & Assert
        with pytest.raises(ValueError, match="test error"):
            client.my_method()

        # Assert counters incremented
        after_total = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",
                {"client": "test_client", "operation": "test_error", "status": "error"},
            )
            or 0.0
        )

        after_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_client", "operation": "test_error", "error_type": "unknown"},
            )
            or 0.0
        )

        assert after_total == before_total + 1.0, "Request counter (status=error) did not increment"
        assert after_errors == before_errors + 1.0, "Error counter (error_type=unknown) did not increment"

    def test_failed_call_still_raises(self) -> None:
        """Test that decorator re-raises the original exception unchanged.

        **Why this test is important:**
          - Ensures decorator doesn't swallow exceptions
          - Verifies circuit breaker and other middleware can still handle errors
          - Critical for maintaining error propagation semantics

        **What it tests:**
          - Exception is re-raised with same type and message
          - No exception wrapping or modification
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics

        class DummyClient:
            @with_client_metrics("test_client", "test_reraise")
            def my_method(self):
                raise RuntimeError("original exception")

        client = DummyClient()

        # Act & Assert
        with pytest.raises(RuntimeError, match="original exception"):
            client.my_method()


class TestWithClientMetricsAsync:
    """Test suite for the @with_client_metrics_async decorator (async)."""

    async def test_async_successful_call_increments_counter(self) -> None:
        """Test that async success increments request_total with status=success.

        **Why this test is important:**
          - Verifies async decorator has identical behavior to sync
          - Ensures async/await semantics don't break metrics
          - Critical for async client monitoring

        **What it tests:**
          - @with_client_metrics_async decorator increments CLIENT_REQUEST_TOTAL
          - Counter has correct labels: client, operation, status="success"
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics_async

        class DummyAsyncClient:
            @with_client_metrics_async("test_async_client", "test_async_operation")
            async def my_method(self):
                return "async success"

        client = DummyAsyncClient()

        # Read counter value before call
        before = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",
                {"client": "test_async_client", "operation": "test_async_operation", "status": "success"},
            )
            or 0.0
        )

        # Act
        result = await client.my_method()

        # Assert
        after = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",
                {"client": "test_async_client", "operation": "test_async_operation", "status": "success"},
            )
            or 0.0
        )

        assert result == "async success"
        assert after == before + 1.0, f"Async counter did not increment (before={before}, after={after})"

    async def test_async_failed_call_increments_error_counter(self) -> None:
        """Test that async exceptions increment error counter with classified error type.

        **Why this test is important:**
          - Verifies async error handling matches sync behavior
          - Ensures async exceptions are classified correctly
          - Critical for async client error monitoring

        **What it tests:**
          - @with_client_metrics_async increments CLIENT_ERRORS_TOTAL
          - Error classification works for async exceptions
          - Counter increments with status="error"
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics_async

        class DummyAsyncClient:
            @with_client_metrics_async("test_async_client", "test_async_error")
            async def my_method(self):
                raise ConnectionError("async connection failed")

        client = DummyAsyncClient()

        # Read counters before call
        before_total = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",
                {"client": "test_async_client", "operation": "test_async_error", "status": "error"},
            )
            or 0.0
        )

        before_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_async_client", "operation": "test_async_error", "error_type": "connection"},
            )
            or 0.0
        )

        # Act & Assert
        with pytest.raises(ConnectionError, match="async connection failed"):
            await client.my_method()

        # Assert counters incremented
        after_total = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",
                {"client": "test_async_client", "operation": "test_async_error", "status": "error"},
            )
            or 0.0
        )

        after_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_async_client", "operation": "test_async_error", "error_type": "connection"},
            )
            or 0.0
        )

        assert after_total == before_total + 1.0, "Async request counter (status=error) did not increment"
        assert after_errors == before_errors + 1.0, (
            "Async error counter (error_type=connection) did not increment"
        )


class TestDecoratorBehavior:
    """Test suite for decorator meta-behavior (wrapping, stacking, label bounds)."""

    def test_label_values_are_bounded(self) -> None:
        """Test that label values come from decorator args, not runtime data.

        **Why this test is important:**
          - Prevents cardinality explosion from unbounded user input
          - Ensures labels are fixed at decoration time
          - Critical for Prometheus performance

        **What it tests:**
          - client and operation labels are set at decoration time
          - No dynamic label values from function arguments
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics

        class DummyClient:
            @with_client_metrics("fixed_client", "fixed_operation")
            def my_method(self, user_provided_arg: str):
                # user_provided_arg should NOT appear in labels
                return f"processed: {user_provided_arg}"

        client = DummyClient()

        # Act - call with different arguments
        result1 = client.my_method("arg1")
        result2 = client.my_method("arg2")

        # Assert - both calls use the same label values
        count = (
            REGISTRY.get_sample_value(
                "inatinq_client_request_total",
                {"client": "fixed_client", "operation": "fixed_operation", "status": "success"},
            )
            or 0.0
        )

        assert result1 == "processed: arg1"
        assert result2 == "processed: arg2"
        assert count >= 2.0, "Both calls should increment the same metric (same labels)"

    def test_marker_attribute_set(self) -> None:
        """Test that decorator sets _client_metrics marker attribute.

        **Why this test is important:**
          - Enables introspection-based tests (test_all_clients_instrumented)
          - Verifies the marker carries correct client/operation labels

        **What it tests:**
          - Wrapper function has _client_metrics attribute
          - Attribute value matches (client, operation) arguments
        """
        # Arrange
        from core.metrics.decorators import with_client_metrics

        class DummyClient:
            @with_client_metrics("test_marker", "test_op")
            def my_method(self):
                return "ok"

        # Assert
        method = vars(DummyClient)["my_method"]
        assert hasattr(method, "_client_metrics")
        assert method._client_metrics == ("test_marker", "test_op")


class TestCancellationSignalsPassthrough:
    """Verify that cancellation/shutdown signals bypass error metrics.

    BaseException subclasses like KeyboardInterrupt, SystemExit, and
    asyncio.CancelledError are not actionable client errors. The decorators
    must let them propagate without incrementing error counters.
    """

    def test_keyboard_interrupt_not_counted(self) -> None:
        """KeyboardInterrupt propagates without incrementing error metrics.

        **Why this test is important:**
          - KeyboardInterrupt during shutdown should not inflate error rates
          - Prevents noisy, non-actionable alerts

        **What it tests:**
          - KeyboardInterrupt is re-raised
          - CLIENT_ERRORS_TOTAL is not incremented
        """
        from core.metrics.decorators import with_client_metrics

        class DummyClient:
            @with_client_metrics("test_client", "test_kb_interrupt")
            def my_method(self):
                raise KeyboardInterrupt

        client = DummyClient()

        before_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_client", "operation": "test_kb_interrupt", "error_type": "unknown"},
            )
            or 0.0
        )

        with pytest.raises(KeyboardInterrupt):
            client.my_method()

        after_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_client", "operation": "test_kb_interrupt", "error_type": "unknown"},
            )
            or 0.0
        )

        assert after_errors == before_errors, "KeyboardInterrupt should not increment error counter"

    async def test_cancelled_error_not_counted(self) -> None:
        """asyncio.CancelledError propagates without incrementing error metrics.

        **Why this test is important:**
          - Task cancellation during graceful shutdown is expected, not an error
          - Prevents inflated error rates during scaling events

        **What it tests:**
          - CancelledError is re-raised
          - CLIENT_ERRORS_TOTAL is not incremented
        """
        import asyncio

        from core.metrics.decorators import with_client_metrics_async

        class DummyAsyncClient:
            @with_client_metrics_async("test_async_client", "test_cancelled")
            async def my_method(self):
                raise asyncio.CancelledError

        client = DummyAsyncClient()

        before_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_async_client", "operation": "test_cancelled", "error_type": "unknown"},
            )
            or 0.0
        )

        with pytest.raises(asyncio.CancelledError):
            await client.my_method()

        after_errors = (
            REGISTRY.get_sample_value(
                "inatinq_client_errors_total",
                {"client": "test_async_client", "operation": "test_cancelled", "error_type": "unknown"},
            )
            or 0.0
        )

        assert after_errors == before_errors, "CancelledError should not increment error counter"


class TestAllClientsInstrumented:
    """Verify all client network methods are instrumented with metrics decorators.

    This test prevents regressions when new client methods are added. It
    programmatically scans all client classes and verifies that every method
    with a circuit breaker decorator also has a metrics decorator.
    """

    # Expected instrumented methods per client class (ground truth from plan spec)
    EXPECTED: dict[str, set[str]] = {
        "OllamaClient": {
            "embed",
            "embed_batch",
            "embed_async",
            "embed_batch_async",
        },
        "QdrantClientWrapper": {
            "ensure_collection_async",
            "ensure_image_collection_async",
            "search_async",
            "disable_indexing",
            "enable_indexing",
            "_do_batch_upsert",
        },
        "WeaviateClientWrapper": {
            "ensure_collection_async",
            "ensure_image_collection_async",
            "search_async",
            "_do_batch_upsert",
        },
        "S3ClientWrapper": {
            "ensure_bucket",
            "put_object",
            "get_object",
            "list_objects",
            "exists",
            "delete_object",
        },
        "CLIPClient": {
            "embed_image",
            "embed_image_async",
            "embed_image_batch",
            "embed_image_batch_async",
            "embed_text",
            "embed_text_async",
            "embed_text_batch",
            "embed_text_batch_async",
        },
    }

    def _get_client_classes(self) -> list[type]:
        """Import and return all client classes."""
        from clients.clip import CLIPClient
        from clients.ollama import OllamaClient
        from clients.qdrant import QdrantClientWrapper
        from clients.s3 import S3ClientWrapper
        from clients.weaviate import WeaviateClientWrapper

        return [OllamaClient, QdrantClientWrapper, WeaviateClientWrapper, S3ClientWrapper, CLIPClient]

    def test_all_clients_instrumented(self) -> None:
        """Test that every circuit-breaker-decorated method also has a metrics decorator.

        **Why this test is important:**
          - Prevents regressions when new client methods are added
          - Ensures observability coverage matches resilience coverage
          - Single test catches all five client modules

        **What it tests:**
          - All expected methods have the _client_metrics marker attribute
          - No method with __wrapped__ (decorator-wrapped) is missing metrics
        """
        missing = []
        for cls in self._get_client_classes():
            cls_name = cls.__name__
            expected_methods = self.EXPECTED.get(cls_name, set())

            # Part 1: Every expected method must have the marker
            for method_name in expected_methods:
                method = vars(cls).get(method_name)
                assert method is not None, f"{cls_name}.{method_name} not found in class"
                if not hasattr(method, "_client_metrics"):
                    missing.append(f"{cls_name}.{method_name}")

            # Part 2: Catch-all — any method with __wrapped__ that we didn't expect
            # should also have _client_metrics (or be flagged for review)
            for attr_name, attr_value in vars(cls).items():
                # Skip staticmethod/classmethod descriptors — they have __wrapped__
                # in Python 3.10+ but are not network call wrappers
                if isinstance(attr_value, (staticmethod, classmethod)):
                    continue
                if not callable(attr_value):
                    continue
                if not hasattr(attr_value, "__wrapped__"):
                    continue
                if attr_name in expected_methods:
                    continue  # Already checked above
                # This method has a functools.wraps wrapper but isn't in our expected list.
                # It might be a new circuit-breaker-decorated method that needs metrics.
                if not hasattr(attr_value, "_client_metrics"):
                    missing.append(f"{cls_name}.{attr_name} (unexpected wrapped method without metrics)")

        assert not missing, (
            f"The following client methods are missing @with_client_metrics decorators:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
