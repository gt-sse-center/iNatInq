"""Client metrics decorators for Prometheus instrumentation.

This module provides decorators that instrument client methods with three
Prometheus metrics:
- CLIENT_REQUEST_DURATION: Histogram of request latency
- CLIENT_REQUEST_TOTAL: Counter of total requests by status
- CLIENT_ERRORS_TOTAL: Counter of errors by error type

The decorators follow the same pattern as the circuit breaker decorators in
`foundation/circuit_breaker.py` — parametric decorator factory returning a
`functools.wraps`-decorated wrapper.

Usage::

    @with_client_metrics("ollama", "embed")
    @with_circuit_breaker("ollama")
    def embed(self, text: str) -> list[float]:
        # metrics decorator is outermost, measures full duration including breaker check
        return self._client.post("/api/embed", json={"text": text})

    @with_client_metrics_async("qdrant", "search")
    @with_circuit_breaker_async("qdrant")
    async def search_async(self, query_vector: list[float]) -> SearchResults:
        return await self._client.search(query_vector)
"""

import functools
import time

from core.metrics.classify import classify_error
from core.metrics.registry import (
    CLIENT_ERRORS_TOTAL,
    CLIENT_REQUEST_DURATION,
    CLIENT_REQUEST_TOTAL,
)


def with_client_metrics(client: str, operation: str):
    """Decorator to wrap synchronous client methods with Prometheus metrics.

    This decorator:
    1. Starts a timer before the method call
    2. Calls the wrapped method
    3. On success: observes duration with status="success", increments request_total
    4. On exception: observes duration with status="error", increments request_total
       and errors_total with classified error_type, then re-raises

    Args:
        client: Client name for the `client` label (e.g., "ollama", "qdrant").
        operation: Operation name for the `operation` label (e.g., "embed", "search").

    Returns:
        Decorator function that wraps methods with metrics logic.

    Example:
        ```python
        @with_client_metrics("s3", "put_object")
        @with_circuit_breaker("s3")
        def put_object(self, *, bucket: str, key: str, body: bytes) -> None:
            self._client.put_object(Bucket=bucket, Key=key, Body=body)
        ```

    Note:
        This decorator should be the outermost decorator (applied before circuit
        breaker) to capture full request duration including breaker checks.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()

            try:
                result = func(self, *args, **kwargs)

                # Success path
                duration = time.perf_counter() - start_time
                CLIENT_REQUEST_DURATION.labels(client=client, operation=operation, status="success").observe(
                    duration
                )
                CLIENT_REQUEST_TOTAL.labels(client=client, operation=operation, status="success").inc()

                return result

            except BaseException as exc:
                # Error path
                duration = time.perf_counter() - start_time
                CLIENT_REQUEST_DURATION.labels(client=client, operation=operation, status="error").observe(
                    duration
                )
                CLIENT_REQUEST_TOTAL.labels(client=client, operation=operation, status="error").inc()

                # Classify error and increment error counter
                error_type = classify_error(exc)
                CLIENT_ERRORS_TOTAL.labels(client=client, operation=operation, error_type=error_type).inc()

                # Re-raise the original exception unchanged
                raise

        # Marker for introspection tests (Task 3: test_all_clients_instrumented)
        wrapper._client_metrics = (client, operation)  # type: ignore[attr-defined]
        return wrapper

    return decorator


def with_client_metrics_async(client: str, operation: str):
    """Decorator to wrap asynchronous client methods with Prometheus metrics.

    This is the async version of `with_client_metrics`. It works with coroutine
    methods and provides identical metrics instrumentation.

    Args:
        client: Client name for the `client` label (e.g., "ollama", "qdrant").
        operation: Operation name for the `operation` label (e.g., "embed", "search").

    Returns:
        Decorator function that wraps async methods with metrics logic.

    Example:
        ```python
        @with_client_metrics_async("qdrant", "search_async")
        @with_circuit_breaker_async("qdrant")
        async def search_async(self, *, query_vector: list[float]) -> SearchResults:
            return await self._client.search(query_vector)
        ```

    Note:
        This decorator should be the outermost decorator (applied before circuit
        breaker) to capture full request duration including breaker checks.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()

            try:
                result = await func(self, *args, **kwargs)

                # Success path
                duration = time.perf_counter() - start_time
                CLIENT_REQUEST_DURATION.labels(client=client, operation=operation, status="success").observe(
                    duration
                )
                CLIENT_REQUEST_TOTAL.labels(client=client, operation=operation, status="success").inc()

                return result

            except BaseException as exc:
                # Error path
                duration = time.perf_counter() - start_time
                CLIENT_REQUEST_DURATION.labels(client=client, operation=operation, status="error").observe(
                    duration
                )
                CLIENT_REQUEST_TOTAL.labels(client=client, operation=operation, status="error").inc()

                # Classify error and increment error counter
                error_type = classify_error(exc)
                CLIENT_ERRORS_TOTAL.labels(client=client, operation=operation, error_type=error_type).inc()

                # Re-raise the original exception unchanged
                raise

        # Marker for introspection tests (Task 3: test_all_clients_instrumented)
        wrapper._client_metrics = (client, operation)  # type: ignore[attr-defined]
        return wrapper

    return decorator
