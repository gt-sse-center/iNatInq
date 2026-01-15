"""Retry utilities with exponential backoff using tenacity.

This module provides reusable retry utilities for consistent retry behavior
with structured logging across all client wrappers.

## Components

### RetryWithBackoff
Class-based retry utility with exponential backoff and structured logging.

### ErrorClassifier (Protocol)
Protocol for classifying errors as retriable vs non-retriable. Clients
implement this to define their specific error classification logic.

### HTTPErrorClassifier
Base implementation with common HTTP status code classification:
- 5xx errors: Retriable (server-side issues)
- 4xx errors: Non-retriable (client-side issues)
- Throttling: Always retriable

### create_retry_logger
Factory function to create retry logging callbacks with custom error
detail extraction.

## Usage

```python
from foundation.retry import HTTPErrorClassifier, create_retry_logger

class MyClientClassifier(HTTPErrorClassifier):
    def is_retriable(self, exc: BaseException) -> bool:
        if isinstance(exc, MyConnectionError):
            return True
        if isinstance(exc, MyClientError):
            return self.is_retriable_http_status(exc.status_code)
        return False

    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        if isinstance(exc, MyClientError):
            return {"error_code": exc.code, "http_status": exc.status_code}
        return {}

classifier = MyClientClassifier()
log_retry = create_retry_logger(logger, classifier.get_error_details)
```
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, Protocol, TypeVar, runtime_checkable

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

T = TypeVar("T")

# Default exceptions to retry on (common transient errors)
DEFAULT_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    Exception,  # Will be narrowed by specific exception types in usage
)

# =============================================================================
# Error Classification
# =============================================================================

# Universal HTTP status codes that indicate transient/retriable errors
RETRIABLE_HTTP_STATUS_CODES: frozenset[str] = frozenset(
    {
        "500",  # Internal Server Error
        "502",  # Bad Gateway
        "503",  # Service Unavailable
        "504",  # Gateway Timeout
    }
)


@runtime_checkable
class ErrorClassifier(Protocol):
    """Protocol for error classification in retry logic.

    Clients implement this protocol to define which exceptions should
    trigger retries and how to extract error details for logging.

    Example:
        ```python
        class S3ErrorClassifier:
            def is_retriable(self, exc: BaseException) -> bool:
                if isinstance(exc, EndpointConnectionError):
                    return True
                return False

            def get_error_details(self, exc: BaseException) -> dict[str, Any]:
                if isinstance(exc, ClientError):
                    return {"error_code": exc.response.get("Error", {}).get("Code")}
                return {}
        ```
    """

    def is_retriable(self, exc: BaseException) -> bool:
        """Determine if an exception should trigger a retry.

        Args:
            exc: The exception to classify.

        Returns:
            True if the error is transient and should be retried,
            False if it's a permanent error that should fail immediately.
        """
        ...

    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        """Extract structured error details for logging.

        Args:
            exc: The exception to extract details from.

        Returns:
            Dictionary with client-specific error details (e.g., error_code,
            http_status, request_id). Empty dict if no details available.
        """
        ...


class HTTPErrorClassifier(ABC):
    """Base error classifier with HTTP status code classification.

    Provides common logic for HTTP-based clients:
    - 5xx status codes are retriable (server errors)
    - 4xx status codes are NOT retriable (client errors)
    - Throttling errors are always retriable

    Subclasses must implement `is_retriable()` and `get_error_details()`
    with their client-specific exception handling.

    Attributes:
        retriable_http_codes: Set of HTTP status codes considered retriable.
            Defaults to 500, 502, 503, 504.

    Example:
        ```python
        class QdrantErrorClassifier(HTTPErrorClassifier):
            def is_retriable(self, exc: BaseException) -> bool:
                if isinstance(exc, QdrantConnectionError):
                    return True
                if isinstance(exc, QdrantAPIError):
                    return self.is_retriable_http_status(exc.status_code)
                return False

            def get_error_details(self, exc: BaseException) -> dict[str, Any]:
                if isinstance(exc, QdrantAPIError):
                    return {"status": exc.status_code, "message": exc.message}
                return {}
        ```
    """

    retriable_http_codes: frozenset[str] = RETRIABLE_HTTP_STATUS_CODES

    def is_retriable_http_status(self, status: str | int) -> bool:
        """Check if an HTTP status code indicates a retriable error.

        Args:
            status: HTTP status code as string or int.

        Returns:
            True for 5xx server errors, False for 4xx client errors,
            False for unknown status codes (fail fast).
        """
        status_str = str(status)

        # 5xx errors are retriable
        if status_str in self.retriable_http_codes:
            return True

        # 4xx errors are NOT retriable (client errors)
        if status_str.startswith("4"):
            return False

        # Unknown - default to not retriable (fail fast)
        return False

    @abstractmethod
    def is_retriable(self, exc: BaseException) -> bool:
        """Determine if an exception should trigger a retry.

        Subclasses must implement this with their specific exception handling.
        Use `is_retriable_http_status()` for HTTP status code checks.

        Args:
            exc: The exception to classify.

        Returns:
            True if retriable, False otherwise.
        """

    @abstractmethod
    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        """Extract structured error details for logging.

        Subclasses must implement this to extract client-specific error
        details from exceptions.

        Args:
            exc: The exception to extract details from.

        Returns:
            Dictionary with error details for structured logging.
        """


# =============================================================================
# Retry Logging
# =============================================================================


def create_retry_logger(
    logger: logging.Logger,
    get_error_details: Callable[[BaseException], dict[str, Any]] | None = None,
    message: str = "Operation failed, retrying",
) -> Callable[[Any], None]:
    """Create a retry logging callback for tenacity.

    This factory creates a callback function suitable for tenacity's
    `before_sleep` parameter. It logs retry attempts with structured
    context including attempt number, wait time, and error details.

    Args:
        logger: Logger instance to use for logging.
        get_error_details: Optional function to extract additional error
            details from exceptions. Should return a dict with keys like
            "error_code", "http_status", etc.
        message: Log message template.

    Returns:
        Callback function for tenacity's before_sleep parameter.

    Example:
        ```python
        def get_s3_error_details(exc: BaseException) -> dict[str, Any]:
            if isinstance(exc, ClientError):
                return {"error_code": exc.response.get("Error", {}).get("Code")}
            return {}

        log_retry = create_retry_logger(logger, get_s3_error_details, "S3 operation failed")

        Retrying(before_sleep=log_retry, ...)
        ```
    """

    def log_retry(retry_state: Any) -> None:
        if retry_state.outcome is None or not retry_state.outcome.failed:
            return

        exc = retry_state.outcome.exception()
        wait_time = retry_state.next_action.sleep if retry_state.next_action else 0

        extra: dict[str, Any] = {
            "attempt": retry_state.attempt_number,
            "wait_seconds": round(wait_time, 2),
            "error_type": type(exc).__name__,
        }

        # Add client-specific error details if extractor provided
        if get_error_details is not None:
            extra.update(get_error_details(exc))

        logger.warning(message, extra=extra)

    return log_retry


# =============================================================================
# RetryWithBackoff Class (Original Implementation)
# =============================================================================


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
        wait_strategy = wait_exponential(multiplier=self.multiplier, min=self.wait_min, max=self.wait_max)

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
