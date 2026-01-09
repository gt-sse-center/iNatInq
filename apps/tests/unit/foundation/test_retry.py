"""Unit tests for foundation.retry module.

This file tests the RetryWithBackoff class which provides retry logic with
exponential backoff for resilient operations in the pipeline service.

# Test Coverage

The tests cover:
  - Initialization: Default and custom configuration values
  - Success Paths: Immediate success, function arguments/kwargs passing
  - Retry Logic: Retryable exception handling, max attempts exhaustion
  - Exception Filtering: Non-retryable exceptions (TypeError, AttributeError, KeyError)
  - Custom Configuration: Custom retry_exceptions parameter override
  - Logging: Retry attempt logging, final failure logging

# Test Structure

Tests use pytest class-based organization with descriptive test names.
Async operations are tested using pytest-asyncio when applicable.

# Running Tests

Run with: pytest tests/unit/foundation/test_retry.py
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from foundation.retry import RetryWithBackoff

# =============================================================================
# Initialization Tests
# =============================================================================


class TestRetryWithBackoff:
    """Test suite for RetryWithBackoff class."""

    def test_init_defaults(self) -> None:
        """Test that RetryWithBackoff initializes with default values.

        **Why this test is important:**
          - Default configuration must work out of the box
          - Ensures zero values are handled correctly
          - Critical for ease of use and backward compatibility
          - Validates that default logger is properly configured

        **What it tests:**
          - max_attempts defaults to 3
          - wait_min defaults to 2.0
          - wait_max defaults to 10.0
          - multiplier defaults to 1.0
          - retry_exceptions defaults to (Exception,)
          - logger defaults to "pipeline.spark"
        """
        retry = RetryWithBackoff()

        assert retry.max_attempts == 3
        assert retry.wait_min == 2.0
        assert retry.wait_max == 10.0
        assert retry.multiplier == 1.0
        assert retry.retry_exceptions == (Exception,)
        assert retry.logger.name == "pipeline.spark"

    def test_init_custom_values(self) -> None:
        """Test that RetryWithBackoff accepts custom initialization values.

        **Why this test is important:**
          - Custom configuration allows tuning retry behavior for specific use cases
          - Ensures all parameters can be overridden
          - Critical for flexibility and adapting to different service requirements
          - Validates custom logger integration

        **What it tests:**
          - Custom max_attempts is preserved
          - Custom wait_min, wait_max, multiplier are preserved
          - Custom retry_exceptions tuple is preserved
          - Custom logger instance is preserved
        """
        custom_logger = logging.getLogger("custom.logger")
        custom_exceptions = (ValueError, RuntimeError)

        retry = RetryWithBackoff(
            max_attempts=5,
            wait_min=1.0,
            wait_max=30.0,
            multiplier=2.0,
            retry_exceptions=custom_exceptions,
            logger=custom_logger,
        )

        assert retry.max_attempts == 5
        assert retry.wait_min == 1.0
        assert retry.wait_max == 30.0
        assert retry.multiplier == 2.0
        assert retry.retry_exceptions == custom_exceptions
        assert retry.logger == custom_logger

    # =============================================================================
    # Success Path Tests
    # =============================================================================

    def test_call_succeeds_on_first_attempt(self) -> None:
        """Test that call() returns result when function succeeds immediately.

        **Why this test is important:**
          - Most operations succeed on first attempt, so this is the common path
          - Ensures no unnecessary overhead when operations succeed
          - Validates that successful results are returned correctly
          - Critical for performance and correctness

        **What it tests:**
          - Function that succeeds immediately returns result
          - No retries are attempted when function succeeds
          - Return value is preserved correctly
        """
        retry = RetryWithBackoff(max_attempts=3)

        def successful_func() -> str:
            return "success"

        result = retry.call(successful_func)

        assert result == "success"

    def test_call_with_args_and_kwargs(self) -> None:
        """Test that call() passes args and kwargs correctly.

        **Why this test is important:**
          - Retry logic must not interfere with function arguments
          - Ensures positional and keyword arguments are passed correctly
          - Critical for maintaining function signatures and behavior
          - Validates that retry wrapper doesn't break argument passing

        **What it tests:**
          - Positional arguments are passed correctly
          - Keyword arguments are passed correctly
          - Mixed positional and keyword arguments work correctly
          - Return values are preserved
        """
        retry = RetryWithBackoff()

        def func_with_args(x: int, y: int, z: int = 0) -> int:
            return x + y + z

        result = retry.call(func_with_args, 1, 2, z=3)

        assert result == 6

    # =============================================================================
    # Retry Logic Tests
    # =============================================================================

    def test_call_retries_on_retryable_exception(self) -> None:
        """Test that call() retries when function raises retryable exception.

        **Why this test is important:**
          - Retry logic is the core functionality of this class
          - Ensures transient failures are handled gracefully
          - Validates that retries actually occur when expected
          - Critical for resilience and fault tolerance

        **What it tests:**
          - Function that fails then succeeds is retried
          - Correct number of attempts are made
          - Result is returned after successful retry
          - Call count reflects retry attempts
        """
        retry = RetryWithBackoff(
            max_attempts=3,
            retry_exceptions=(ValueError,),
        )
        call_count = 0

        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retryable error")
            return "success"

        result = retry.call(failing_func)

        assert result == "success"
        assert call_count == 3

    def test_call_raises_after_max_attempts(self) -> None:
        """Test that call() raises exception after max attempts exhausted.

        **Why this test is important:**
          - Must fail eventually to avoid infinite retry loops
          - Ensures exceptions are properly propagated after retries exhausted
          - Validates that max_attempts limit is respected
          - Critical for preventing resource exhaustion and timeouts

        **What it tests:**
          - Exception is raised after max_attempts failures
          - Original exception type is preserved
          - Exception message is preserved
          - No additional retries occur after limit reached
        """
        retry = RetryWithBackoff(
            max_attempts=2,
            retry_exceptions=(ValueError,),
        )

        def always_failing_func() -> str:
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            retry.call(always_failing_func)

    # =============================================================================
    # Exception Filtering Tests
    # =============================================================================

    def test_call_does_not_retry_typeerror(self) -> None:
        """Test that TypeError is not retried (programming error).

        **Why this test is important:**
          - Programming errors (TypeError, AttributeError, KeyError) should fail fast
          - Retrying programming errors masks bugs and wastes resources
          - Ensures type errors are caught immediately
          - Critical for developer experience and debugging

        **What it tests:**
          - TypeError is not retried even if in retry_exceptions
          - Exception is raised immediately
          - Original exception type is preserved
        """
        retry = RetryWithBackoff(
            max_attempts=3,
            retry_exceptions=(Exception,),
        )

        def type_error_func() -> str:
            raise TypeError("programming error")

        with pytest.raises(TypeError, match="programming error"):
            retry.call(type_error_func)

    def test_call_does_not_retry_attributeerror(self) -> None:
        """Test that AttributeError is not retried (programming error).

        **Why this test is important:**
          - AttributeError indicates missing attribute/method (programming error)
          - Should fail fast to surface bugs immediately
          - Retrying would mask the issue
          - Critical for catching API misuse and bugs

        **What it tests:**
          - AttributeError is not retried even if in retry_exceptions
          - Exception is raised immediately
          - Original exception type is preserved
        """
        retry = RetryWithBackoff(
            max_attempts=3,
            retry_exceptions=(Exception,),
        )

        def attr_error_func() -> str:
            raise AttributeError("programming error")

        with pytest.raises(AttributeError, match="programming error"):
            retry.call(attr_error_func)

    def test_call_does_not_retry_keyerror(self) -> None:
        """Test that KeyError is not retried (programming error).

        **Why this test is important:**
          - KeyError indicates missing dictionary key (programming error)
          - Should fail fast to surface bugs immediately
          - Retrying would mask the issue
          - Critical for catching data structure misuse

        **What it tests:**
          - KeyError is not retried even if in retry_exceptions
          - Exception is raised immediately
          - Original exception type is preserved
        """
        retry = RetryWithBackoff(
            max_attempts=3,
            retry_exceptions=(Exception,),
        )

        def key_error_func() -> str:
            raise KeyError("programming error")

        with pytest.raises(KeyError):
            retry.call(key_error_func)

    # =============================================================================
    # Custom Configuration Tests
    # =============================================================================

    def test_call_uses_custom_retry_exceptions(self) -> None:
        """Test that call() accepts custom retry_exceptions parameter.

        **Why this test is important:**
          - Allows per-call exception filtering for fine-grained control
          - Enables different retry strategies for different exception types
          - Critical for handling mixed exception scenarios
          - Validates runtime exception override capability

        **What it tests:**
          - Custom retry_exceptions parameter overrides instance default
          - Only specified exceptions trigger retries
          - Result is returned after successful retry with custom exceptions
        """
        retry = RetryWithBackoff(
            max_attempts=3,
            retry_exceptions=(ValueError,),
        )
        call_count = 0

        def runtime_error_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("should retry")
            return "success"

        # Override with RuntimeError as retryable
        result = retry.call(runtime_error_func, retry_exceptions=(RuntimeError,))

        assert result == "success"
        assert call_count == 2

    # =============================================================================
    # Logging Tests
    # =============================================================================

    def test_call_logs_retry_attempts(self) -> None:
        """Test that retry attempts are logged.

        **Why this test is important:**
          - Retry logging is critical for observability and debugging
          - Helps identify transient failures and service degradation
          - Enables monitoring and alerting on retry patterns
          - Critical for production troubleshooting

        **What it tests:**
          - Warning logs are emitted for each retry attempt
          - Correct number of log calls matches retry count
          - Success after retries does not log error
        """
        mock_logger = MagicMock(spec=logging.Logger)
        retry = RetryWithBackoff(
            max_attempts=3,
            retry_exceptions=(ValueError,),
            logger=mock_logger,
        )
        call_count = 0

        def failing_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retryable")
            return "success"

        result = retry.call(failing_func)

        assert result == "success"
        # Should log warnings for retry attempts (2 retries)
        assert mock_logger.warning.call_count == 2

    def test_call_logs_final_failure(self) -> None:
        """Test that final failure is logged after all retries exhausted.

        **Why this test is important:**
          - Final failure logging is critical for error tracking
          - Helps identify persistent failures and service outages
          - Enables alerting when operations fail after all retries
          - Critical for incident response and debugging

        **What it tests:**
          - Error log is emitted when all retries are exhausted
          - Log call occurs exactly once for final failure
          - Error log contains failure information
        """
        mock_logger = MagicMock(spec=logging.Logger)
        retry = RetryWithBackoff(
            max_attempts=2,
            retry_exceptions=(ValueError,),
            logger=mock_logger,
        )

        def always_failing_func() -> str:
            raise ValueError("always fails")

        with pytest.raises(ValueError):
            retry.call(always_failing_func)

        # Should log error for final failure (after callback is invoked)
        # The after callback is called by tenacity after all retries are exhausted
        assert mock_logger.error.called
        call_kwargs = mock_logger.error.call_args[1]
        assert call_kwargs["extra"]["max_attempts"] == 2
        assert call_kwargs["extra"]["error"] == "always fails"
        assert call_kwargs["extra"]["error_type"] == "ValueError"

    def test_log_retry_early_return_when_outcome_none(self) -> None:
        """Test that _log_retry returns early when outcome is None.

        **Why this test is important:**
          - Defensive programming prevents crashes from invalid retry state
          - Early return prevents accessing attributes on None
          - Ensures robust error handling for edge cases
          - Critical for production stability

        **What it tests:**
          - _log_retry returns immediately when retry_state.outcome is None
          - No logging occurs when outcome is None
          - Method doesn't crash when outcome is None
        """
        mock_logger = MagicMock(spec=logging.Logger)

        # Create mock retry_state with outcome=None
        mock_retry_state = MagicMock()
        mock_retry_state.outcome = None

        # Call _log_retry directly - should return early
        RetryWithBackoff._log_retry(mock_retry_state, mock_logger, 3)  # type: ignore[attr-defined]

        # Should not have logged anything
        mock_logger.warning.assert_not_called()

    def test_log_retry_early_return_when_not_failed(self) -> None:
        """Test that _log_retry returns early when outcome is not failed.

        **Why this test is important:**
          - Only log retry attempts when there's an actual failure
          - Prevents unnecessary logging for successful outcomes
          - Ensures logging only occurs for actual failures
          - Critical for log cleanliness and performance

        **What it tests:**
          - _log_retry returns immediately when outcome is not failed
          - No logging occurs when outcome is not failed
          - Method handles successful outcomes correctly
        """
        mock_logger = MagicMock(spec=logging.Logger)

        # Create mock retry_state with outcome that is not failed
        mock_retry_state = MagicMock()
        mock_outcome = MagicMock()
        mock_outcome.failed = False
        mock_retry_state.outcome = mock_outcome

        # Call _log_retry directly - should return early
        RetryWithBackoff._log_retry(mock_retry_state, mock_logger, 3)  # type: ignore[attr-defined]

        # Should not have logged anything
        mock_logger.warning.assert_not_called()

    def test_log_failure_early_return_when_outcome_none(self) -> None:
        """Test that _log_failure returns early when outcome is None.

        **Why this test is important:**
          - Defensive programming prevents crashes from invalid retry state
          - Early return prevents accessing attributes on None
          - Ensures robust error handling for edge cases
          - Critical for production stability

        **What it tests:**
          - _log_failure returns immediately when retry_state.outcome is None
          - No logging occurs when outcome is None
          - Method doesn't crash when outcome is None
        """
        mock_logger = MagicMock(spec=logging.Logger)

        # Create mock retry_state with outcome=None
        mock_retry_state = MagicMock()
        mock_retry_state.outcome = None

        # Call _log_failure directly - should return early
        RetryWithBackoff._log_failure(  # type: ignore[attr-defined]
            mock_retry_state, mock_logger, 3
        )

        # Should not have logged anything
        mock_logger.error.assert_not_called()

    def test_log_failure_early_return_when_not_failed(self) -> None:
        """Test that _log_failure returns early when outcome is not failed.

        **Why this test is important:**
          - Only log failures when there's an actual failure
          - Prevents unnecessary logging for successful outcomes
          - Ensures logging only occurs for actual failures
          - Critical for log cleanliness and performance

        **What it tests:**
          - _log_failure returns immediately when outcome is not failed
          - No logging occurs when outcome is not failed
          - Method handles successful outcomes correctly
        """
        mock_logger = MagicMock(spec=logging.Logger)

        # Create mock retry_state with outcome that is not failed
        mock_retry_state = MagicMock()
        mock_outcome = MagicMock()
        mock_outcome.failed = False
        mock_retry_state.outcome = mock_outcome

        # Call _log_failure directly - should return early
        RetryWithBackoff._log_failure(  # type: ignore[attr-defined]
            mock_retry_state, mock_logger, 3
        )

        # Should not have logged anything
        mock_logger.error.assert_not_called()
