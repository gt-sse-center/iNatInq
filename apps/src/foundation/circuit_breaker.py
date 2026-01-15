"""Circuit breaker utilities for resilient HTTP clients.

This module provides circuit breaker patterns for external service dependencies
using two libraries:
- **pybreaker**: For synchronous methods
- **aiobreaker**: For asynchronous methods (native asyncio support)

Circuit breakers prevent cascading failures by failing fast when a service
is degraded or unavailable.

## Dual-Breaker Architecture

Clients that have both sync and async methods should use two circuit breakers:

```python
@attrs.define(frozen=False, slots=True)
class MyClient(CircuitBreakerMixin):
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)
    _async_breaker: aiobreaker.CircuitBreaker = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        # Sync breaker (pybreaker)
        self._init_circuit_breaker()

        # Async breaker (aiobreaker)
        name, fail_max, timeout = self._circuit_breaker_config()
        self._async_breaker = create_async_circuit_breaker(name, fail_max, timeout)

    @with_circuit_breaker("myservice")
    def sync_method(self) -> str:
        return requests.get(self.url).text

    @with_circuit_breaker_async("myservice")
    async def async_method(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
            return response.text
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

## Decorators

- `@with_circuit_breaker("service")`: For sync methods, uses `_breaker`
- `@with_circuit_breaker_async("service")`: For async methods, uses `_async_breaker`

Both decorators:
1. Check if circuit is open → fail fast with UpstreamError
2. Wrap the call to track success/failure
3. Convert CircuitBreakerError to UpstreamError for consistent error handling
"""

import functools
import logging
from collections.abc import Callable
from datetime import timedelta
from typing import NoReturn

import aiobreaker
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

    @staticmethod
    def before_call(
        cb: pybreaker.CircuitBreaker,
        func: Callable,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Log before calling protected function.

        Args:
            cb: The circuit breaker instance.
            func: Function being called.
            *args: Positional arguments (unused but required for signature).
            **kwargs: Keyword arguments (unused but required for signature).
        """
        logger.debug(
            "Circuit breaker call",
            extra={
                "circuit_breaker": cb.name,
                "state": str(cb.current_state),
                "function": func.__name__,
            },
        )

    def failure(
        self,
        cb: pybreaker.CircuitBreaker,
        exc: BaseException,
    ) -> None:
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
) -> pybreaker.CircuitBreaker:
    """Create a circuit breaker for an external service dependency.

    Args:
        name: Unique name for the circuit breaker (e.g., "ollama", "qdrant").
        failure_threshold: Number of consecutive failures before opening
        circuit. Default: 5.
        recovery_timeout: Seconds to wait before attempting recovery (moving to
            half-open state). Default: 60.

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
    msg = (
        f"{service_name} service is currently unavailable. "
        "The circuit breaker is open due to repeated failures. "
        "The service will be retried automatically after the recovery timeout."
    )
    raise UpstreamError(msg)


def _get_breaker_or_raise(instance: object) -> pybreaker.CircuitBreaker:
    """Get circuit breaker from instance or raise RuntimeError.

    Args:
        instance: Object that should have a _breaker attribute.

    Returns:
        The circuit breaker instance.

    Raises:
        RuntimeError: If instance has no _breaker attribute.
    """
    breaker = getattr(instance, "_breaker", None)
    if breaker is None:
        msg = (
            f"{instance.__class__.__name__} has no circuit breaker. "
            "Ensure the class inherits from CircuitBreakerMixin and "
            "calls _init_circuit_breaker() in __attrs_post_init__."
        )
        raise RuntimeError(msg)
    return breaker


def with_circuit_breaker(service_name: str):
    """Decorator to wrap synchronous method calls with circuit breaker protection.

    This decorator eliminates the need for nested function definitions and
    repetitive try/except blocks. It automatically:
    1. Checks if the circuit breaker is open (fail fast)
    2. Wraps the method call with the circuit breaker
    3. Handles CircuitBreakerError by raising UpstreamError

    Args:
        service_name: Service name for error messages and breaker
        identification.

    Returns:
        Decorator function that wraps methods with circuit breaker logic.

    Example:
        ```python
        @attrs.define(frozen=False, slots=True)
        class MyClient(CircuitBreakerMixin):
            _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)

            @with_circuit_breaker("myclient")
            def fetch_data(self, key: str) -> dict:
                # Method body without nested functions or try/except
                response = requests.get(f"{self.url}/data/{key}")
                response.raise_for_status()
                return response.json()
        ```

    Note:
        This decorator expects the instance to have a `_breaker` attribute
        (typically provided by CircuitBreakerMixin).

    See Also:
        with_circuit_breaker_async: Async version for coroutine methods.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            breaker = _get_breaker_or_raise(self)

            # Check if open first (fail fast)
            if breaker.current_state == pybreaker.STATE_OPEN:
                handle_circuit_breaker_error(service_name)

            # Wrap the function call
            def _impl():
                return func(self, *args, **kwargs)

            try:
                return breaker.call(_impl)
            except pybreaker.CircuitBreakerError:
                handle_circuit_breaker_error(service_name)

        return wrapper

    return decorator


def with_circuit_breaker_async(service_name: str):
    """Decorator to wrap async method calls with circuit breaker protection.

    This is the async version of `with_circuit_breaker`. It works with
    coroutine methods using the `aiobreaker` library which provides native
    asyncio support.

    **How it works:**
    1. Checks if circuit breaker is open → fail fast with UpstreamError
    2. Wraps the coroutine with `breaker.call_async()` for proper tracking
    3. On success → aiobreaker records success automatically
    4. On failure → aiobreaker records failure, may trigger state change

    Args:
        service_name: Service name for error messages and breaker
        identification.

    Returns:
        Decorator function that wraps async methods with circuit breaker logic.

    Example:
        ```python
        @attrs.define(frozen=False, slots=True)
        class MyAsyncClient(CircuitBreakerMixin):
            _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)
            _async_breaker: aiobreaker.CircuitBreaker = attrs.field(init=False)

            @with_circuit_breaker_async("myclient")
            async def fetch_data(self, key: str) -> dict:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{self.url}/data/{key}")
                    response.raise_for_status()
                    return response.json()
        ```

    Note:
        This decorator expects the instance to have a `_async_breaker` attribute
        which is an `aiobreaker.CircuitBreaker` instance. Falls back to
        `_breaker` if `_async_breaker` is not present (for backward compat).

        Uses the `aiobreaker` library which provides native asyncio support
        for circuit breaker patterns.

    See Also:
        with_circuit_breaker: Sync version for regular methods.
        create_async_circuit_breaker: Factory for async circuit breakers.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get async breaker - requires _async_breaker (aiobreaker.CircuitBreaker)
            breaker = getattr(self, "_async_breaker", None)
            if breaker is None:
                msg = (
                    f"{self.__class__.__name__} has no _async_breaker. "
                    "Ensure the class has _async_breaker attribute "
                    "(use create_async_circuit_breaker() to create one)."
                )
                raise RuntimeError(msg)

            # Check if open first (fail fast)
            # aiobreaker uses current_state which returns CircuitBreakerState enum
            current_state = getattr(breaker, "current_state", None)
            if current_state is not None:
                state_str = str(current_state).lower()
                if "open" in state_str and "half" not in state_str:
                    handle_circuit_breaker_error(service_name)

            # Create async callable for aiobreaker
            async def _impl():
                return await func(self, *args, **kwargs)

            try:
                # Use aiobreaker's call_async method for async functions
                return await breaker.call_async(_impl)
            except aiobreaker.CircuitBreakerError:
                handle_circuit_breaker_error(service_name)

        return wrapper

    return decorator


def create_async_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
) -> aiobreaker.CircuitBreaker:
    """Create an async-compatible circuit breaker for coroutine methods.

    Uses the `aiobreaker` library which provides native asyncio support,
    unlike pybreaker which requires Tornado for async.

    Args:
        name: Unique name for the circuit breaker (e.g., "qdrant").
        failure_threshold: Number of consecutive failures before opening
            circuit. Default: 5.
        recovery_timeout: Seconds to wait before attempting recovery.
            Default: 60.

    Returns:
        Configured aiobreaker.CircuitBreaker instance.

    Example:
        ```python
        _async_breaker = create_async_circuit_breaker(
            name="qdrant",
            failure_threshold=3,
            recovery_timeout=60,
        )

        @with_circuit_breaker_async("qdrant")
        async def search(self, query: str) -> list:
            return await self._client.search(query)
        ```

    Note:
        The async circuit breaker uses the same state transitions as pybreaker:
        - CLOSED → OPEN: After `failure_threshold` consecutive failures
        - OPEN → HALF_OPEN: After `recovery_timeout` seconds
        - HALF_OPEN → CLOSED: After first successful call
        - HALF_OPEN → OPEN: If call fails during recovery
    """
    return aiobreaker.CircuitBreaker(
        fail_max=failure_threshold,
        timeout_duration=timedelta(seconds=recovery_timeout),
        name=name,
    )
