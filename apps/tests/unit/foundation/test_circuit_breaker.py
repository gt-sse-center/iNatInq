"""Unit tests for foundation.circuit_breaker module.

This file tests the circuit breaker utilities which provide fault tolerance
for external service calls in the pipeline service using the pybreaker library.

# Test Coverage

The tests cover:
  - CircuitBreakerListener: State change logging, before_call logging, success/failure logging
  - create_circuit_breaker: Default configuration, custom configuration, listener attachment
  - Circuit Breaker Usage: Decorator usage, call() method, state transitions (OPEN after failures)
  - Error Handling: handle_circuit_breaker_error raises UpstreamError

# Test Structure

Tests use pytest class-based organization with mocking for logging verification.
Circuit breaker state transitions are tested using pybreaker's built-in functionality.

# Running Tests

Run with: pytest tests/unit/foundation/test_circuit_breaker.py
"""

import logging
from unittest.mock import MagicMock, patch

import aiobreaker
import aiobreaker.state as aio_state
import pybreaker
import pytest
from foundation.exceptions import UpstreamError

from foundation.circuit_breaker import (
    CircuitBreakerListener,
    _get_breaker_or_raise,
    create_async_circuit_breaker,
    create_circuit_breaker,
    handle_circuit_breaker_error,
    with_circuit_breaker,
    with_circuit_breaker_async,
)

# =============================================================================
# CircuitBreakerListener Tests
# =============================================================================


class TestCircuitBreakerListener:
    """Test suite for CircuitBreakerListener class."""

    def test_state_change_logs_transition(self) -> None:
        """Test that state_change logs circuit breaker state transitions.

        **Why this test is important:**
          - State transitions are critical events that need observability
          - Logging state changes helps with debugging and monitoring
          - Enables alerting when circuit breakers open
          - Critical for production troubleshooting and incident response

        **What it tests:**
          - State change callback logs warning with structured fields
          - Log includes circuit breaker name, old state, new state, failure count
          - Logging format matches structured logging pattern
        """
        mock_logger = MagicMock(spec=logging.Logger)
        with patch("foundation.circuit_breaker.logger", mock_logger):
            listener = CircuitBreakerListener()
            mock_cb = MagicMock(spec=pybreaker.CircuitBreaker)
            mock_cb.name = "test_service"
            mock_cb.fail_counter = 5

            # pybreaker uses string states, not enum attributes
            listener.state_change(
                mock_cb,
                "CLOSED",  # type: ignore[arg-type]
                "OPEN",  # type: ignore[arg-type]
            )

            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args[1]
            assert call_kwargs["extra"]["circuit_breaker"] == "test_service"
            assert call_kwargs["extra"]["old_state"] == "CLOSED"
            assert call_kwargs["extra"]["new_state"] == "OPEN"
            assert call_kwargs["extra"]["failure_count"] == 5

    def test_before_call_logs(self) -> None:
        """Test that before_call logs debug message.

        **Why this test is important:**
          - Debug logging helps trace circuit breaker operation
          - Enables detailed troubleshooting in development/staging
          - Validates that callback is invoked before function execution
          - Critical for debugging circuit breaker behavior

        **What it tests:**
          - before_call callback logs debug message
          - Log includes circuit breaker name in structured format
          - Logging format matches structured logging pattern
        """
        mock_logger = MagicMock(spec=logging.Logger)
        with patch("foundation.circuit_breaker.logger", mock_logger):
            listener = CircuitBreakerListener()
            mock_cb = MagicMock(spec=pybreaker.CircuitBreaker)
            mock_cb.name = "test_service"

            listener.before_call(mock_cb, lambda: None)

            mock_logger.debug.assert_called_once()
            call_kwargs = mock_logger.debug.call_args[1]
            assert call_kwargs["extra"]["circuit_breaker"] == "test_service"

    def test_success_logs(self) -> None:
        """Test that success logs debug message.

        **Why this test is important:**
          - Success logging helps track circuit breaker recovery
          - Enables monitoring of successful operations after failures
          - Validates that callback is invoked on successful execution
          - Critical for observability and performance monitoring

        **What it tests:**
          - success callback logs debug message
          - Log includes circuit breaker name and state
          - Logging format matches structured logging pattern
        """
        mock_logger = MagicMock(spec=logging.Logger)
        with patch("foundation.circuit_breaker.logger", mock_logger):
            listener = CircuitBreakerListener()
            mock_cb = MagicMock(spec=pybreaker.CircuitBreaker)
            mock_cb.name = "test_service"
            mock_cb.current_state = pybreaker.STATE_CLOSED

            listener.success(mock_cb)

            mock_logger.debug.assert_called_once()
            call_kwargs = mock_logger.debug.call_args[1]
            assert call_kwargs["extra"]["circuit_breaker"] == "test_service"
            assert call_kwargs["extra"]["state"] == str(pybreaker.STATE_CLOSED)

    def test_failure_logs(self) -> None:
        """Test that failure logs error message.

        **Why this test is important:**
          - Failure logging is critical for error tracking
          - Helps identify service degradation and failures
          - Enables alerting when failures occur
          - Critical for incident response and debugging

        **What it tests:**
          - failure callback logs error message
          - Log includes circuit breaker name, failure count, error message, error type
          - Logging format matches structured logging pattern
        """
        mock_logger = MagicMock(spec=logging.Logger)
        with patch("foundation.circuit_breaker.logger", mock_logger):
            listener = CircuitBreakerListener()
            mock_cb = MagicMock(spec=pybreaker.CircuitBreaker)
            mock_cb.name = "test_service"
            mock_cb.fail_counter = 3
            mock_cb.current_state = "closed"

            test_exception = ConnectionError("connection failed")

            listener.failure(mock_cb, test_exception)

            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs["extra"]["circuit_breaker"] == "test_service"
            assert call_kwargs["extra"]["failure_count"] == 3
            assert call_kwargs["extra"]["exception_message"] == "connection failed"
            assert call_kwargs["extra"]["exception_type"] == "ConnectionError"


# =============================================================================
# create_circuit_breaker Tests
# =============================================================================


class TestCreateCircuitBreaker:
    """Test suite for create_circuit_breaker function."""

    def test_creates_circuit_breaker_with_defaults(self) -> None:
        """Test that create_circuit_breaker creates breaker with defaults.

        **Why this test is important:**
          - Default configuration must work out of the box
          - Ensures sensible defaults for common use cases
          - Validates that breaker is created successfully
          - Critical for ease of use and backward compatibility

        **What it tests:**
          - Returns a pybreaker.CircuitBreaker instance
          - Name is set correctly
          - Default failure_threshold (5) is applied
          - Default recovery_timeout (60) is applied
        """
        breaker = create_circuit_breaker(name="test_service")

        assert isinstance(breaker, pybreaker.CircuitBreaker)
        assert breaker.name == "test_service"
        assert breaker.fail_max == 5
        assert breaker.reset_timeout == 60

    def test_creates_circuit_breaker_with_custom_config(self) -> None:
        """Test that create_circuit_breaker accepts custom configuration.

        **Why this test is important:**
          - Custom configuration allows tuning for different services
          - Different services need different failure thresholds and recovery times
          - Critical for adapting to service-specific requirements
          - Validates parameter passing to pybreaker

        **What it tests:**
          - Custom name is preserved
          - Custom failure_threshold is applied
          - Custom recovery_timeout is applied
        """
        breaker = create_circuit_breaker(
            name="custom_service",
            failure_threshold=10,
            recovery_timeout=120,
        )

        assert breaker.name == "custom_service"
        assert breaker.fail_max == 10
        assert breaker.reset_timeout == 120

    def test_has_listener_attached(self) -> None:
        """Test that created circuit breaker has listener attached.

        **Why this test is important:**
          - Listener provides observability through logging
          - Ensures listeners are automatically configured
          - Validates that logging integration works out of the box
          - Critical for production observability

        **What it tests:**
          - Circuit breaker has at least one listener
          - CircuitBreakerListener is in the listeners list
          - Listener is properly attached and functional
        """
        breaker = create_circuit_breaker(name="test_service")

        assert len(breaker.listeners) > 0
        assert any(isinstance(listener, CircuitBreakerListener) for listener in breaker.listeners)

    # =============================================================================
    # Circuit Breaker Usage Tests
    # =============================================================================

    def test_circuit_breaker_can_be_used_as_decorator(self) -> None:
        """Test that circuit breaker can be used as decorator.

        **Why this test is important:**
          - Decorator pattern is the primary usage pattern
          - Ensures decorator syntax works correctly
          - Validates that wrapped functions execute properly
          - Critical for ergonomics and ease of use

        **What it tests:**
          - Circuit breaker can be used as function decorator
          - Decorated function executes and returns result
          - Result is preserved correctly
        """
        breaker = create_circuit_breaker(
            name="test_service",
            failure_threshold=2,
        )

        @breaker
        def test_func() -> str:
            return "success"

        result = test_func()
        assert result == "success"

    def test_circuit_breaker_call_method(self) -> None:
        """Test that circuit breaker call() method works.

        **Why this test is important:**
          - call() method is an alternative to decorator pattern
          - Useful for inline usage without decorator syntax
          - Validates that call() method executes functions correctly
          - Critical for flexibility and different usage patterns

        **What it tests:**
          - call() method executes function and returns result
          - Result is preserved correctly
          - Function execution is wrapped by circuit breaker
        """
        breaker = create_circuit_breaker(
            name="test_service",
        )

        def test_func() -> str:
            return "success"

        result = breaker.call(test_func)
        assert result == "success"

    def test_circuit_breaker_opens_after_failures(self) -> None:
        """Test that circuit breaker opens after threshold failures.

        **Why this test is important:**
          - Circuit opening is the core fault tolerance mechanism
          - Prevents cascading failures by failing fast
          - Validates that failure threshold is respected
          - Critical for protecting downstream services

        **What it tests:**
          - Circuit breaker allows failures up to threshold
          - After threshold failures, circuit opens
          - Open circuit raises CircuitBreakerError instead of calling function
          - State transition CLOSED → OPEN occurs correctly
        """
        breaker = create_circuit_breaker(
            name="test_service",
            failure_threshold=2,
        )

        def failing_func() -> str:
            raise ConnectionError("connection failed")

        # fail_max=2 means after 2 failures, circuit opens
        # First failure
        with pytest.raises(ConnectionError):
            breaker.call(failing_func)

        # Second failure opens the circuit (fail_max=2 means circuit opens after 2 failures)
        with pytest.raises(pybreaker.CircuitBreakerError):
            breaker.call(failing_func)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestHandleCircuitBreakerError:
    """Test suite for handle_circuit_breaker_error function."""

    def test_raises_upstream_error(self) -> None:
        """Test that handle_circuit_breaker_error raises UpstreamError.

        **Why this test is important:**
          - Error conversion provides consistent error handling
          - UpstreamError maps to HTTP 502 in API layer
          - Ensures circuit breaker errors are handled consistently
          - Critical for API error handling and status codes

        **What it tests:**
          - Function raises UpstreamError exception
          - Error message includes service name
          - Exception type matches UpstreamError
        """
        with pytest.raises(UpstreamError, match="test_service service is currently unavailable"):
            handle_circuit_breaker_error("test_service")


# =============================================================================
# _get_breaker_or_raise Tests
# =============================================================================


class TestGetBreakerOrRaise:
    """Test suite for _get_breaker_or_raise helper function."""

    def test_returns_breaker_when_present(self) -> None:
        """Test that _get_breaker_or_raise returns breaker when present."""
        breaker = create_circuit_breaker("test")
        instance = MagicMock()
        instance._breaker = breaker

        result = _get_breaker_or_raise(instance)

        assert result is breaker

    def test_raises_runtime_error_when_missing(self) -> None:
        """Test that _get_breaker_or_raise raises RuntimeError when no breaker."""
        instance = MagicMock(spec=[])  # No _breaker attribute

        with pytest.raises(RuntimeError, match="has no circuit breaker"):
            _get_breaker_or_raise(instance)


# =============================================================================
# with_circuit_breaker Decorator Tests
# =============================================================================


class TestWithCircuitBreakerDecorator:
    """Test suite for with_circuit_breaker sync decorator."""

    def test_decorator_calls_function_successfully(self) -> None:
        """Test that decorated function executes and returns result."""

        class TestClient:
            _breaker = create_circuit_breaker("test", failure_threshold=3)

            @with_circuit_breaker("test")
            def do_work(self, value: int) -> int:
                return value * 2

        client = TestClient()
        result = client.do_work(5)

        assert result == 10

    def test_decorator_tracks_failures(self) -> None:
        """Test that decorator properly tracks failures with circuit breaker."""

        class TestClient:
            _breaker = create_circuit_breaker("test", failure_threshold=2)

            @with_circuit_breaker("test")
            def do_work(self) -> str:
                raise ConnectionError("failed")

        client = TestClient()

        # First failure
        with pytest.raises(ConnectionError):
            client.do_work()

        # Second failure should trigger circuit open (fail_max=2)
        with pytest.raises(UpstreamError, match="currently unavailable"):
            client.do_work()

    def test_decorator_fails_fast_when_open(self) -> None:
        """Test that decorator fails fast when circuit is open."""
        breaker = create_circuit_breaker("test", failure_threshold=1)

        class TestClient:
            _breaker = breaker
            call_count = 0

            @with_circuit_breaker("test")
            def do_work(self) -> str:
                self.call_count += 1
                raise ConnectionError("failed")

        client = TestClient()

        # Trigger circuit open
        with pytest.raises(UpstreamError):
            client.do_work()

        # Reset call count
        client.call_count = 0

        # Next call should fail fast (circuit open)
        with pytest.raises(UpstreamError, match="currently unavailable"):
            client.do_work()

        # Function should NOT have been called
        assert client.call_count == 0

    def test_decorator_raises_runtime_error_without_breaker(self) -> None:
        """Test that decorator raises RuntimeError if no _breaker attribute."""

        class TestClient:
            @with_circuit_breaker("test")
            def do_work(self) -> str:
                return "success"

        client = TestClient()

        with pytest.raises(RuntimeError, match="has no circuit breaker"):
            client.do_work()


# =============================================================================
# with_circuit_breaker_async Decorator Tests
# =============================================================================


class TestWithCircuitBreakerAsyncDecorator:
    """Test suite for with_circuit_breaker_async async decorator.

    Uses aiobreaker.CircuitBreaker for native asyncio support.
    """

    @pytest.mark.asyncio
    async def test_async_decorator_calls_function_successfully(self) -> None:
        """Test that decorated async function executes and returns result."""

        class TestClient:
            _async_breaker = create_async_circuit_breaker("test", failure_threshold=3)

            @with_circuit_breaker_async("test")
            async def do_work(self, value: int) -> int:
                return value * 2

        client = TestClient()
        result = await client.do_work(5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_async_decorator_tracks_failures(self) -> None:
        """Test that async decorator properly tracks failures and opens circuit."""

        class TestClient:
            _async_breaker = create_async_circuit_breaker("test", failure_threshold=2)

            @with_circuit_breaker_async("test")
            async def do_work(self) -> str:
                raise ConnectionError("failed")

        client = TestClient()

        # First failure - should record failure and raise original exception
        with pytest.raises(ConnectionError):
            await client.do_work()

        # Check failure was tracked (aiobreaker uses fail_counter)
        assert client._async_breaker.fail_counter == 1

        # Second failure opens circuit - aiobreaker raises CircuitBreakerError
        # which our decorator converts to UpstreamError
        with pytest.raises(UpstreamError, match="currently unavailable"):
            await client.do_work()

        # Circuit should now be open (aiobreaker uses current_state enum)
        assert client._async_breaker.current_state == aio_state.CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_async_decorator_fails_fast_when_open(self) -> None:
        """Test that async decorator fails fast when circuit is open."""
        breaker = create_async_circuit_breaker("test", failure_threshold=1)

        class TestClient:
            _async_breaker = breaker
            call_count = 0

            @with_circuit_breaker_async("test")
            async def do_work(self) -> str:
                self.call_count += 1
                raise ConnectionError("failed")

        client = TestClient()

        # First call fails and opens circuit (fail_max=1)
        # aiobreaker raises CircuitBreakerError when circuit opens,
        # which our decorator converts to UpstreamError
        with pytest.raises(UpstreamError, match="currently unavailable"):
            await client.do_work()

        # Circuit should be open
        assert client._async_breaker.current_state == aio_state.CircuitBreakerState.OPEN

        # Reset call count
        client.call_count = 0

        # Next call should fail fast (circuit open) - checked before calling func
        with pytest.raises(UpstreamError, match="currently unavailable"):
            await client.do_work()

        # Function should NOT have been called (fail-fast path)
        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_async_decorator_tracks_success(self) -> None:
        """Test that async decorator tracks success for half-open recovery."""

        class TestClient:
            _async_breaker = create_async_circuit_breaker("test", failure_threshold=3)

            @with_circuit_breaker_async("test")
            async def do_work(self) -> str:
                return "success"

        client = TestClient()

        # Successful call
        result = await client.do_work()

        assert result == "success"
        # No failures recorded
        assert client._async_breaker.fail_counter == 0

    @pytest.mark.asyncio
    async def test_async_decorator_raises_runtime_error_without_breaker(self) -> None:
        """Test that async decorator raises RuntimeError if no breakers."""

        class TestClient:
            @with_circuit_breaker_async("test")
            async def do_work(self) -> str:
                return "success"

        client = TestClient()

        with pytest.raises(RuntimeError, match="has no _async_breaker"):
            await client.do_work()

    @pytest.mark.asyncio
    async def test_async_decorator_preserves_function_metadata(self) -> None:
        """Test that async decorator preserves function name and docstring."""

        class TestClient:
            _async_breaker = create_async_circuit_breaker("test")

            @with_circuit_breaker_async("test")
            async def my_special_function(self) -> str:
                """This is my docstring."""
                return "success"

        # Check metadata is preserved
        assert TestClient.my_special_function.__name__ == "my_special_function"
        assert "This is my docstring" in (TestClient.my_special_function.__doc__ or "")

    @pytest.mark.asyncio
    async def test_async_decorator_requires_async_breaker(self) -> None:
        """Test that async decorator requires _async_breaker attribute."""

        class TestClient:
            # Only sync breaker, no async breaker - should fail
            _breaker = create_circuit_breaker("test", failure_threshold=3)

            @with_circuit_breaker_async("test")
            async def do_work(self, value: int) -> int:
                return value * 2

        client = TestClient()

        # Should raise RuntimeError because _async_breaker is required
        with pytest.raises(RuntimeError, match="has no _async_breaker"):
            await client.do_work(5)


# =============================================================================
# create_async_circuit_breaker Tests
# =============================================================================


class TestCreateAsyncCircuitBreaker:
    """Test suite for create_async_circuit_breaker function."""

    def test_creates_aiobreaker_with_defaults(self) -> None:
        """Test that create_async_circuit_breaker creates breaker with defaults."""
        from datetime import timedelta

        breaker = create_async_circuit_breaker(name="test_service")

        assert isinstance(breaker, aiobreaker.CircuitBreaker)
        assert breaker.name == "test_service"
        assert breaker.fail_max == 5
        assert breaker.timeout_duration == timedelta(seconds=60)

    def test_creates_aiobreaker_with_custom_config(self) -> None:
        """Test that create_async_circuit_breaker accepts custom configuration."""
        from datetime import timedelta

        breaker = create_async_circuit_breaker(
            name="custom_service",
            failure_threshold=10,
            recovery_timeout=120,
        )

        assert breaker.name == "custom_service"
        assert breaker.fail_max == 10
        assert breaker.timeout_duration == timedelta(seconds=120)
