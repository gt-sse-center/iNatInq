"""Circuit breaker utilities for resilient HTTP clients.

This module provides circuit breaker patterns for external service dependencies
using the pybreaker library. Circuit breakers prevent cascading failures by
failing fast when a service is degraded or unavailable.

## Usage

```python
from pipeline.foundation.circuit_breaker import create_circuit_breaker

# Create a circuit breaker for a service
ollama_breaker = create_circuit_breaker(
    name="ollama",
    failure_threshold=5,
    recovery_timeout=30,
    expected_exception=ConnectionError,
)

# Use with decorator
@ollama_breaker
def call_ollama_api():
    return requests.post("http://ollama:11434/api/embed", ...)

# Or use call() method
result = ollama_breaker.call(lambda: expensive_operation())
```

## Circuit Breaker States

- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Service is failing, requests fail immediately without calling service
- **HALF_OPEN**: Testing if service has recovered, allows limited requests

## Configuration Guidelines

**Ollama (Embedding Service)**:
- failure_threshold: 5 (fail after 5 consecutive errors)
- recovery_timeout: 30s (try recovery after 30 seconds)
- Rationale: Embedding is on critical path, fail fast to avoid blocking users

**Qdrant/Weaviate (Vector Database)**:
- failure_threshold: 3 (fail after 3 consecutive errors)
- recovery_timeout: 60s (longer recovery for database)
- Rationale: Database failures indicate serious issues, longer recovery time

**MinIO/S3 (Object Storage)**:
- failure_threshold: 5 (more tolerance for transient network issues)
- recovery_timeout: 120s (storage recovery takes longer)
- Rationale: S3 failures often transient, allow time for recovery
"""

from __future__ import annotations

import logging
from typing import Callable, NoReturn

import pybreaker

from foundation.exceptions import UpstreamError

logger = logging.getLogger("foundation.circuit_breaker")


class CircuitBreakerListener(pybreaker.CircuitBreakerListener):
    """Logging listener for circuit breaker state changes.

    This listener logs all circuit breaker events with structured logging
    for observability and debugging.
    """

    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state: pybreaker.CircuitBreakerState | None,
        new_state: pybreaker.CircuitBreakerState,
    ) -> None:
        """Log circuit breaker state transitions.

        Args:
            cb: The circuit breaker instance.
            old_state: Previous state.
            new_state: New state.
        """
        logger.warning(
            "Circuit breaker state changed",
            extra={
                "circuit_breaker": cb.name,
                "old_state": str(old_state),
                "new_state": str(new_state),
                "failure_count": cb.fail_counter,
            },
        )

    def before_call(
        self,
        cb: pybreaker.CircuitBreaker,
        func: Callable,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Log before calling protected function.

        Args:
            cb: The circuit breaker instance.
            func: Function being called.
            *args: Positional arguments (unused but required for signature match).
            **kwargs: Keyword arguments (unused but required for signature match).
        """
        logger.debug(
            "Circuit breaker call",
            extra={
                "circuit_breaker": cb.name,
                "state": str(cb.current_state),
                "function": func.__name__,
            },
        )

    def failure(self, cb: pybreaker.CircuitBreaker, exc: BaseException) -> None:
        """Log when a call fails.

        Args:
            cb: The circuit breaker instance.
            exc: Exception that occurred.
        """
        logger.error(
            "Circuit breaker failure",
            extra={
                "circuit_breaker": cb.name,
                "state": str(cb.current_state),
                "failure_count": cb.fail_counter,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            exc_info=True,
        )

    def success(self, cb: pybreaker.CircuitBreaker) -> None:
        """Log successful call.

        Args:
            cb: The circuit breaker instance.
        """
        logger.debug(
            "Circuit breaker success",
            extra={
                "circuit_breaker": cb.name,
                "state": str(cb.current_state),
            },
        )


def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type[Exception] | None = None,
) -> pybreaker.CircuitBreaker:
    """Create a circuit breaker for an external service dependency.

    Args:
        name: Unique name for the circuit breaker (e.g., "ollama", "qdrant").
        failure_threshold: Number of consecutive failures before opening circuit.
            Default: 5.
        recovery_timeout: Seconds to wait before attempting recovery (moving to
            half-open state). Default: 60.
        expected_exception: Exception type that should trigger the circuit breaker.
            Default: None (catches all exceptions). Note: pybreaker doesn't support
            filtering by exception type in __init__, so this parameter is ignored.

    Returns:
        Configured CircuitBreaker instance with logging listener.

    Example:
        ```python
        # Create breaker for Ollama service
        ollama_breaker = create_circuit_breaker(
            name="ollama",
            failure_threshold=5,
            recovery_timeout=30,
        )

        # Use with decorator
        @ollama_breaker
        def get_embedding(text: str) -> list[float]:
            response = requests.post("http://ollama:11434/api/embed", ...)
            return response.json()["embedding"]

        # Handle circuit open
        try:
            embedding = get_embedding("hello world")
        except pybreaker.CircuitBreakerError:
            # Circuit is open, service is unavailable
            raise UpstreamError("Ollama service unavailable")
        ```

    Note:
        The circuit breaker will automatically transition through states:
        - CLOSED → OPEN: After `failure_threshold` consecutive failures
        - OPEN → HALF_OPEN: After `recovery_timeout` seconds
        - HALF_OPEN → CLOSED: After first successful call
        - HALF_OPEN → OPEN: If call fails during recovery
    """
    return pybreaker.CircuitBreaker(
        name=name,
        fail_max=failure_threshold,
        reset_timeout=recovery_timeout,
        listeners=[CircuitBreakerListener()],
    )


def handle_circuit_breaker_error(service_name: str) -> NoReturn:
    """Handle circuit breaker open state by raising UpstreamError.

    This helper converts pybreaker.CircuitBreakerError into our application's
    UpstreamError for consistent error handling through the middleware.

    Args:
        service_name: Name of the service (for error message).

    Raises:
        UpstreamError: Always raises with message about service unavailability.

    Example:
        ```python
        try:
            result = breaker.call(lambda: expensive_operation())
        except pybreaker.CircuitBreakerError:
            handle_circuit_breaker_error("ollama")
        ```
    """
    raise UpstreamError(
        f"{service_name} service is currently unavailable. "
        "The circuit breaker is open due to repeated failures. "
        "The service will be retried automatically after the recovery timeout."
    )
