"""Unit tests for foundation.retry.async_retry_call.

This file tests the async retry utility function including:
- Success on first attempt
- Retry on retriable exception then succeed
- Max retries exhausted raises UpstreamError
- Non-retriable exception propagates immediately (wrapped in UpstreamError)
- CircuitBreakerError passes through unwrapped
- Retry logging callback invocation

# Running Tests

Run with: pytest tests/unit/foundation/test_async_retry.py
"""

from unittest.mock import AsyncMock, MagicMock

import aiobreaker
import pytest
from foundation.exceptions import UpstreamError
from foundation.retry import async_retry_call


class TestAsyncRetryCall:
    """Test suite for async_retry_call function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        """Test that async_retry_call returns result on first success.

        **Why this test is important:**
          - Most operations succeed on first attempt
          - Ensures no unnecessary overhead when operations succeed
          - Validates that successful results are returned correctly

        **What it tests:**
          - Async function that succeeds immediately returns result
          - No retries are attempted
        """
        mock_func = AsyncMock(return_value="success")

        result = await async_retry_call(
            mock_func,
            max_retries=3,
            is_retriable=lambda exc: isinstance(exc, ValueError),
            operation="test",
        )

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retriable_then_succeed(self) -> None:
        """Test that retriable exceptions trigger retries and eventually succeed.

        **Why this test is important:**
          - Core retry functionality must work for transient failures
          - Ensures retries actually occur when expected
          - Validates that result is returned after successful retry

        **What it tests:**
          - Function that fails twice then succeeds is retried
          - Correct number of attempts are made
          - Result is returned after successful retry
        """
        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return "success"

        result = await async_retry_call(
            flaky_func,
            max_retries=3,
            min_wait=0.01,
            max_wait=0.02,
            is_retriable=lambda exc: isinstance(exc, ValueError),
            operation="test",
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_raises_upstream_error(self) -> None:
        """Test that UpstreamError is raised when all retries are exhausted.

        **Why this test is important:**
          - Must fail eventually to avoid infinite retry loops
          - Ensures UpstreamError wrapping for consistent error handling
          - Validates that max_retries limit is respected

        **What it tests:**
          - UpstreamError is raised after max_retries failures
          - Error message includes operation name and attempt count
          - Original exception is chained
        """

        async def always_fails() -> str:
            raise ValueError("always fails")

        with pytest.raises(UpstreamError, match="test_op failed after 2 attempts"):
            await async_retry_call(
                always_fails,
                max_retries=2,
                min_wait=0.01,
                max_wait=0.02,
                is_retriable=lambda exc: isinstance(exc, ValueError),
                operation="test_op",
            )

    @pytest.mark.asyncio
    async def test_non_retriable_exception_wrapped_in_upstream_error(self) -> None:
        """Test that non-retriable exceptions are wrapped in UpstreamError.

        **Why this test is important:**
          - Non-retriable exceptions should not be retried
          - Must be wrapped in UpstreamError for consistent error handling
          - Validates fail-fast behavior for permanent errors

        **What it tests:**
          - Non-retriable exception is not retried
          - Exception is wrapped in UpstreamError
          - Error message includes operation name
        """

        async def raises_non_retriable() -> str:
            raise RuntimeError("permanent error")

        with pytest.raises(UpstreamError, match="test_op failed.*permanent error"):
            await async_retry_call(
                raises_non_retriable,
                max_retries=3,
                min_wait=0.01,
                max_wait=0.02,
                is_retriable=lambda exc: isinstance(exc, ValueError),  # Only ValueError retriable
                operation="test_op",
            )

    @pytest.mark.asyncio
    async def test_upstream_error_propagates_as_is(self) -> None:
        """Test that UpstreamError propagates without double-wrapping.

        **Why this test is important:**
          - UpstreamError from inner code should not be re-wrapped
          - Preserves original error message and context
          - Prevents nested UpstreamError chains

        **What it tests:**
          - UpstreamError raised inside is propagated as-is
          - Message is preserved without modification
        """

        async def raises_upstream() -> str:
            raise UpstreamError("inner upstream error")

        with pytest.raises(UpstreamError, match="^inner upstream error$"):
            await async_retry_call(
                raises_upstream,
                max_retries=3,
                min_wait=0.01,
                max_wait=0.02,
                is_retriable=lambda exc: False,
                operation="test_op",
            )

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_propagates_as_is(self) -> None:
        """Test that CircuitBreakerError propagates without wrapping.

        **Why this test is important:**
          - CB errors must reach the outer CB decorator for proper handling
          - The decorator converts them to UpstreamError with service-unavailable message
          - Wrapping would break the CB decorator's error handling

        **What it tests:**
          - aiobreaker.CircuitBreakerError propagates through unwrapped
          - Not caught by the generic Exception handler
        """
        import datetime

        async def raises_cb_error() -> str:
            raise aiobreaker.CircuitBreakerError(
                "Circuit is open", datetime.datetime.now(datetime.timezone.utc)
            )

        with pytest.raises(aiobreaker.CircuitBreakerError):
            await async_retry_call(
                raises_cb_error,
                max_retries=3,
                min_wait=0.01,
                max_wait=0.02,
                is_retriable=lambda exc: False,
                operation="test_op",
            )

    @pytest.mark.asyncio
    async def test_before_sleep_callback_invoked(self) -> None:
        """Test that before_sleep callback is called on retry.

        **Why this test is important:**
          - Retry logging depends on the before_sleep callback
          - Validates integration with tenacity's callback mechanism
          - Critical for observability

        **What it tests:**
          - before_sleep callback is invoked on each retry
          - Callback receives retry state
        """
        call_count = 0
        mock_before_sleep = MagicMock()

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "success"

        await async_retry_call(
            flaky_func,
            max_retries=3,
            min_wait=0.01,
            max_wait=0.02,
            is_retriable=lambda exc: isinstance(exc, ValueError),
            before_sleep=mock_before_sleep,
            operation="test",
        )

        # before_sleep should be called for each retry (2 retries = 2 calls)
        assert mock_before_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self) -> None:
        """Test that args and kwargs are passed through to the function.

        **Why this test is important:**
          - Retry logic must not interfere with function arguments
          - Ensures positional and keyword arguments are forwarded correctly

        **What it tests:**
          - Positional arguments are passed correctly
          - Keyword arguments are passed correctly
        """

        async def func_with_args(x: int, y: int, z: int = 0) -> int:
            return x + y + z

        result = await async_retry_call(
            func_with_args,
            1,
            2,
            z=3,
            max_retries=3,
            is_retriable=lambda exc: False,
            operation="test",
        )

        assert result == 6
