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

import pybreaker
import pytest
from foundation.exceptions import UpstreamError
from foundation.circuit_breaker import (
    CircuitBreakerListener,
    create_circuit_breaker,
    handle_circuit_breaker_error,
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
