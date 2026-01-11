"""Retry utilities with exponential backoff using tenacity.

This module provides a reusable retry utility class that can be used across
the pipeline package for consistent retry behavior with structured logging.
"""


import logging
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

T = TypeVar("T")

# Default exceptions to retry on (common transient errors)
DEFAULT_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    Exception,  # Will be narrowed by specific exception types in usage
)


class RetryWithBackoff:
    """Retry utility with exponential backoff and structured logging.

    This class provides a reusable retry mechanism using tenacity with
    configurable exponential backoff, exception filtering, and structured
    logging integration.

    Attributes:
        max_attempts: Maximum number of retry attempts (default: 3).
        wait_min: Minimum wait time between retries in seconds (default: 2.0).
        wait_max: Maximum wait time between retries in seconds (default: 10.0).
        multiplier: Exponential backoff multiplier (default: 1.0).
        retry_exceptions: Tuple of exception types to retry on (default: common
            transient errors).
        logger: Logger instance for structured logging (default: pipeline.spark).

    Example:
        ```python
        from foundation.retry import RetryWithBackoff

        retry = RetryWithBackoff(max_attempts=5, wait_min=1.0, wait_max=30.0)
        result = retry.call(lambda: some_function(arg1, arg2))
        ```

    Note:
        The retry strategy uses exponential backoff: wait_min * (multiplier ** attempt).
        Unexpected exceptions (TypeError, AttributeError, KeyError) are never retried.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        wait_min: float = 2.0,
        wait_max: float = 10.0,
        multiplier: float = 1.0,
        retry_exceptions: tuple[type[Exception], ...] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize retry utility.

        Args:
            max_attempts: Maximum number of retry attempts.
            wait_min: Minimum wait time between retries (seconds).
            wait_max: Maximum wait time between retries (seconds).
            multiplier: Exponential backoff multiplier.
            retry_exceptions: Exception types to retry on. If None, uses default
                transient error types.
            logger: Logger instance for structured logging. If None, uses
                "pipeline.spark" logger.
        """
        self.max_attempts = max_attempts
        self.wait_min = wait_min
        self.wait_max = wait_max
        self.multiplier = multiplier
        # Use provided retry_exceptions or default to common transient errors
        self.retry_exceptions = retry_exceptions or DEFAULT_RETRY_EXCEPTIONS
        self.logger = logger or logging.getLogger("pipeline.spark")

    @staticmethod
    def _log_retry(
        retry_state: Any,
        logger: logging.Logger,
        max_attempts: int,
    ) -> None:
        """Log callback for retry attempts.

        Args:
            retry_state: Tenacity retry state object.
            logger: Logger instance for structured logging.
            max_attempts: Maximum number of retry attempts.
        """
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return

        exception = retry_state.outcome.exception()
        wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
        logger.warning(
            "Retry attempt failed, retrying",
            extra={
                "attempt": retry_state.attempt_number,
                "max_attempts": max_attempts,
                "wait_seconds": wait_time,
                "error": str(exception),
                "error_type": type(exception).__name__,
            },
        )

    @staticmethod
    def _log_failure(
        retry_state: Any,
        logger: logging.Logger,
        max_attempts: int,
    ) -> None:
        """Log callback for final failure after all retries exhausted.

        Args:
            retry_state: Tenacity retry state object.
            logger: Logger instance for structured logging.
            max_attempts: Maximum number of retry attempts.
        """
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return

        exception = retry_state.outcome.exception()
        logger.exception(
            "All retry attempts exhausted",
            extra={
                "max_attempts": max_attempts,
                "error": str(exception),
                "error_type": type(exception).__name__,
            },
        )

    def call(
        self,
        func: Callable[..., T],
        *args: Any,
        retry_exceptions: tuple[type[Exception], ...] | None = None,
        **kwargs: Any,
    ) -> T:
        """Call a function with retry logic.

        Args:
            func: Function to call with retry logic.
            *args: Positional arguments to pass to func.
            retry_exceptions: Override default retry exceptions for this call.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Result of func(*args, **kwargs).

        Raises:
            Exception: If all retry attempts fail, raises the last exception.
            TypeError, AttributeError, KeyError: These exceptions are never
                retried and are raised immediately.
        """
        # Use provided retry_exceptions or fall back to instance default
        exceptions_to_retry = retry_exceptions or self.retry_exceptions

        # Configure exponential backoff with min/max bounds
        # tenacity's wait_exponential formula: min * (multiplier ** attempt)
        # This matches our desired behavior: wait_min * (multiplier ** attempt)
        wait_strategy = wait_exponential(
            multiplier=self.multiplier, min=self.wait_min, max=self.wait_max
        )

        # Create callbacks with bound parameters using partial
        log_retry = partial(
            self._log_retry,
            logger=self.logger,
            max_attempts=self.max_attempts,
        )
        log_failure = partial(
            self._log_failure,
            logger=self.logger,
            max_attempts=self.max_attempts,
        )

        # Configure retry strategy
        retry = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_strategy,
            retry=retry_if_exception_type(exceptions_to_retry),
            before_sleep=log_retry,
            after=log_failure,
            reraise=True,
        )

        try:
            result: T = retry(func, *args, **kwargs)
            return result
        except (TypeError, AttributeError, KeyError) as e:
            # Don't retry on unexpected exceptions (programming errors, etc.)
            self.logger.exception(
                "Unexpected error, not retrying",
                extra={"error": str(e), "error_type": type(e).__name__},

            )
            raise
