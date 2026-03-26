"""Client metrics decorators for Prometheus instrumentation.

This module provides decorators that instrument client methods with three
Prometheus metrics:

.. note::
    Uses ``from __future__ import annotations`` so that ``CoroutineType[...]``
    type annotations are not evaluated at runtime (``types.CoroutineType`` is
    not subscriptable at runtime in Python < 3.12).
- CLIENT_REQUEST_DURATION: Histogram of request latency
- CLIENT_REQUEST_TOTAL: Counter of total requests by status
- CLIENT_ERRORS_TOTAL: Counter of errors by error type

The decorators follow the same pattern as the circuit breaker decorators in
`foundation/circuit_breaker.py` — parametric decorator factory returning a
`functools.wraps`-decorated wrapper.

Usage::

    @with_client_metrics("clip", "embed")
    @with_circuit_breaker("clip")
    def embed(self, image_bytes: bytes) -> list[float]:
        # metrics decorator is outermost, measures full duration including breaker check
        return self._client.post("/embed", files={"image": image_bytes})

    @with_client_metrics_async("qdrant", "search")
    @with_circuit_breaker_async("qdrant")
    async def search_async(self, query_vector: list[float]) -> SearchResults:
        return await self._client.search(query_vector)
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import CoroutineType

from foundation.metrics.classify import classify_error
from foundation.metrics.registry import (
    CLIENT_ERRORS_TOTAL,
    CLIENT_REQUEST_DURATION,
    CLIENT_REQUEST_TOTAL,
)

P = ParamSpec("P")
R = TypeVar("R")
SelfT = TypeVar("SelfT")


def _record_metrics(client: str, operation: str, start_time: float, exc: Exception | None = None) -> None:
    duration = time.perf_counter() - start_time
    status = "error" if exc is not None else "success"
    CLIENT_REQUEST_DURATION.labels(client=client, operation=operation, status=status).observe(duration)
    CLIENT_REQUEST_TOTAL.labels(client=client, operation=operation, status=status).inc()
    if exc is not None:
        # Classify error and increment error counter
        CLIENT_ERRORS_TOTAL.labels(client=client, operation=operation, error_type=classify_error(exc)).inc()


def with_client_metrics(
    client: str, operation: str
) -> Callable[[Callable[Concatenate[SelfT, P], R]], Callable[Concatenate[SelfT, P], R]]:
    """Decorator to wrap synchronous client methods with Prometheus metrics.

    This decorator:
    1. Starts a timer before the method call
    2. Calls the wrapped method
    3. On success: observes duration with status="success", increments request_total
    4. On exception: observes duration with status="error", increments request_total
       and errors_total with classified error_type, then re-raises

    Args:
        client: Client name for the `client` label (e.g., "clip", "qdrant").
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

    def decorator(func: Callable[Concatenate[SelfT, P], R]) -> Callable[Concatenate[SelfT, P], R]:
        @functools.wraps(func)
        def wrapper(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()

            try:
                result = func(self, *args, **kwargs)

                # Success path
                _record_metrics(client, operation, start_time)

                return result

            except Exception as exc:
                # Error path — catch Exception, not BaseException, so that
                # cancellation/shutdown signals (KeyboardInterrupt, SystemExit,
                # asyncio.CancelledError, GeneratorExit) propagate without
                # being counted as client errors.
                _record_metrics(client, operation, start_time, exc)

                # Re-raise the original exception unchanged
                raise

        # Marker for introspection tests (test_all_clients_instrumented)
        wrapper._client_metrics = (client, operation)  # pyright: ignore[reportAttributeAccessIssue]
        return wrapper

    return decorator


def with_client_metrics_async(
    client: str, operation: str
) -> Callable[
    [Callable[Concatenate[SelfT, P], CoroutineType[Any, Any, R]]],
    Callable[Concatenate[SelfT, P], CoroutineType[Any, Any, R]],
]:
    """Decorator to wrap asynchronous client methods with Prometheus metrics.

    This is the async version of `with_client_metrics`. It works with coroutine
    methods and provides identical metrics instrumentation.

    Args:
        client: Client name for the `client` label (e.g., "clip", "qdrant").
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

    def decorator(
        func: Callable[Concatenate[SelfT, P], CoroutineType[Any, Any, R]],
    ) -> Callable[Concatenate[SelfT, P], CoroutineType[Any, Any, R]]:
        @functools.wraps(func)
        async def wrapper(self: SelfT, *args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()

            try:
                result = await func(self, *args, **kwargs)

                # Success path
                _record_metrics(client, operation, start_time)

                return result

            except Exception as exc:
                # Error path — catch Exception, not BaseException, so that
                # cancellation/shutdown signals (KeyboardInterrupt, SystemExit,
                # asyncio.CancelledError, GeneratorExit) propagate without
                # being counted as client errors.
                _record_metrics(client, operation, start_time, exc)

                # Re-raise the original exception unchanged
                raise

        # Marker for introspection tests (test_all_clients_instrumented)
        wrapper._client_metrics = (client, operation)  # pyright: ignore[reportAttributeAccessIssue]
        return wrapper

    return decorator
