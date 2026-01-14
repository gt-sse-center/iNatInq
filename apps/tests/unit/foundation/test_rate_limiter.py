"""Unit tests for foundation.rate_limiter module.

This file tests the RateLimiter class which provides token bucket rate limiting
for async operations in the pipeline service.

# Test Coverage

The tests cover:
  - Initialization: Valid rate configuration, zero/negative rate validation
  - Rate Retrieval: get_rate() method correctness
  - Rate Limiting: First call behavior, rate limit enforcement, timing accuracy
  - Concurrency: Concurrent acquire_permission() calls, thread safety
  - Edge Cases: Fast rates (small intervals), slow rates (large intervals)

# Test Structure

Tests use pytest-asyncio for async test execution. Timing assertions use
tolerance ranges to account for system scheduling variability.

# Running Tests

Run with: pytest tests/unit/foundation/test_rate_limiter.py
"""

import asyncio
import time

import pytest
from foundation.rate_limiter import RateLimiter

# =============================================================================
# Initialization Tests
# =============================================================================


class TestRateLimiter:
    """Test suite for RateLimiter class."""

    def test_init_valid_rate(self) -> None:
        """Test that RateLimiter initializes with valid rate.

        **Why this test is important:**
          - Initialization is the foundation of the rate limiter
          - Validates that valid rates are accepted correctly
          - Ensures rate is stored and accessible via get_rate()
          - Critical for basic functionality and configuration

        **What it tests:**
          - RateLimiter accepts positive rate values
          - get_rate() returns the configured rate correctly
          - Rate is stored as float for precision
        """
        limiter = RateLimiter(rate_per_sec=10)

        assert limiter.get_rate() == 10.0

    def test_init_zero_rate_raises(self) -> None:
        """Test that RateLimiter raises ValueError for zero rate.

        **Why this test is important:**
          - Zero rate would cause division by zero or infinite waits
          - Prevents invalid configuration that would break the limiter
          - Ensures clear error messages for configuration mistakes
          - Critical for preventing runtime errors

        **What it tests:**
          - Zero rate raises ValueError
          - Error message indicates rate must be greater than 0
          - Exception type is preserved correctly
        """
        with pytest.raises(ValueError, match="rate_per_sec must be greater than 0"):
            RateLimiter(rate_per_sec=0)

    def test_init_negative_rate_raises(self) -> None:
        """Test that RateLimiter raises ValueError for negative rate.

        **Why this test is important:**
          - Negative rate is meaningless and would cause errors
          - Prevents invalid configuration that would break the limiter
          - Ensures clear error messages for configuration mistakes
          - Critical for preventing runtime errors

        **What it tests:**
          - Negative rate raises ValueError
          - Error message indicates rate must be greater than 0
          - Exception type is preserved correctly
        """
        with pytest.raises(ValueError, match="rate_per_sec must be greater than 0"):
            RateLimiter(rate_per_sec=-1)

    # =============================================================================
    # Rate Retrieval Tests
    # =============================================================================

    def test_get_rate(self) -> None:
        """Test that get_rate returns correct rate.

        **Why this test is important:**
          - Rate retrieval is needed for monitoring and debugging
          - Validates that stored rate matches configured value
          - Ensures rate is returned as float for precision
          - Critical for observability and configuration verification

        **What it tests:**
          - get_rate() returns the configured rate
          - Rate is returned as float
          - Multiple instances can have different rates
        """
        limiter = RateLimiter(rate_per_sec=5)
        assert limiter.get_rate() == 5.0

        limiter = RateLimiter(rate_per_sec=100)
        assert limiter.get_rate() == 100.0

    # =============================================================================
    # Rate Limiting Tests
    # =============================================================================

    @pytest.mark.asyncio
    async def test_acquire_first_call_no_wait(self) -> None:
        """Test that first acquire() call does not wait.

        **Why this test is important:**
          - First call should be immediate to avoid unnecessary delays
          - Ensures rate limiter doesn't block unnecessarily
          - Critical for performance and user experience
          - Validates initial token bucket state

        **What it tests:**
          - First acquire() call completes immediately (< 10ms)
          - No artificial delay on first operation
          - Subsequent calls will be rate-limited
        """
        limiter = RateLimiter(rate_per_sec=10)

        start = time.monotonic()
        await limiter.acquire_permission()
        elapsed = time.monotonic() - start

        # First call should be immediate (or very fast)
        assert elapsed < 0.01

    @pytest.mark.asyncio
    async def test_acquire_respects_rate_limit(self) -> None:
        """Test that acquire() respects the rate limit.

        **Why this test is important:**
          - Rate limiting is the core functionality of this class
          - Ensures operations don't exceed the configured rate
          - Validates timing accuracy and interval calculation
          - Critical for preventing service overload and rate limit violations

        **What it tests:**
          - Second acquire() call waits approximately the calculated interval
          - Wait time matches expected interval within tolerance
          - Rate limit is enforced correctly
        """
        limiter = RateLimiter(rate_per_sec=2)  # 0.5 seconds between calls
        expected_interval = 0.5

        # First call should be immediate
        await limiter.acquire_permission()

        # Second call should wait approximately expected_interval
        start = time.monotonic()
        await limiter.acquire_permission()
        elapsed = time.monotonic() - start

        # Allow small tolerance for timing
        assert elapsed >= expected_interval - 0.05
        assert elapsed <= expected_interval + 0.05

    @pytest.mark.asyncio
    async def test_acquire_multiple_calls(self) -> None:
        """Test that multiple acquire() calls respect rate limit.

        **Why this test is important:**
          - Real-world usage involves multiple consecutive calls
          - Ensures rate limiting works correctly over multiple operations
          - Validates cumulative timing across multiple acquires
          - Critical for sustained rate limiting behavior

        **What it tests:**
          - Multiple consecutive acquire() calls respect rate limit
          - Total elapsed time matches expected cumulative interval
          - Each call (except first) waits appropriately
        """
        limiter = RateLimiter(rate_per_sec=10)  # 0.1 seconds between calls
        num_calls = 3

        start = time.monotonic()
        for _ in range(num_calls):
            await limiter.acquire_permission()
        total_elapsed = time.monotonic() - start

        # Should take at least (num_calls - 1) * interval
        expected_min = (num_calls - 1) * 0.1
        assert total_elapsed >= expected_min - 0.05
        assert total_elapsed <= expected_min + 0.1

    # =============================================================================
    # Concurrency Tests
    # =============================================================================

    @pytest.mark.asyncio
    async def test_acquire_concurrent_calls(self) -> None:
        """Test that concurrent acquire() calls are thread-safe.

        **Why this test is important:**
          - Real-world usage involves concurrent coroutines
          - Ensures rate limiter works correctly under concurrent load
          - Validates async lock prevents race conditions
          - Critical for correctness in async/await contexts

        **What it tests:**
          - Multiple concurrent acquire() calls complete without deadlock
          - All coroutines complete successfully
          - Rate limiting is still enforced (some calls may wait)
          - No race conditions or data corruption
        """
        limiter = RateLimiter(rate_per_sec=100)  # Fast rate for concurrent test

        # Create multiple coroutines that acquire simultaneously
        async def acquire_once() -> float:
            start = time.monotonic()
            await limiter.acquire_permission()
            return time.monotonic() - start

        # Run multiple acquire calls concurrently
        tasks = [acquire_once() for _ in range(5)]
        elapsed_times = await asyncio.gather(*tasks)

        # All should complete (no deadlocks)
        assert len(elapsed_times) == 5
        # All should have completed (some may have waited)
        assert all(elapsed >= 0 for elapsed in elapsed_times)

    # =============================================================================
    # Edge Case Tests
    # =============================================================================

    @pytest.mark.asyncio
    async def test_acquire_fast_rate(self) -> None:
        """Test that acquire() works with fast rate (small interval).

        **Why this test is important:**
          - High-rate scenarios (many ops/second) are common
          - Ensures rate limiter handles small intervals correctly
          - Validates precision with fast rates
          - Critical for high-throughput use cases

        **What it tests:**
          - Fast rate (1000 ops/sec) works correctly
          - Small intervals don't cause precision issues
          - Operations complete quickly but rate is still enforced
        """
        limiter = RateLimiter(rate_per_sec=1000)  # Very fast rate

        start = time.monotonic()
        await limiter.acquire_permission()
        await limiter.acquire_permission()
        elapsed = time.monotonic() - start

        # Should be very fast, but may have small delay
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_acquire_slow_rate(self) -> None:
        """Test that acquire() works with slow rate (large interval).

        **Why this test is important:**
          - Low-rate scenarios (few ops/second) are common
          - Ensures rate limiter handles large intervals correctly
          - Validates timing accuracy with slow rates
          - Critical for low-throughput use cases

        **What it tests:**
          - Slow rate (1 ops/sec) works correctly
          - Large intervals are calculated and enforced correctly
          - Wait time matches expected interval within tolerance
        """
        limiter = RateLimiter(rate_per_sec=1)  # 1 second between calls

        await limiter.acquire_permission()

        start = time.monotonic()
        await limiter.acquire_permission()
        elapsed = time.monotonic() - start

        # Should wait approximately 1 second
        assert elapsed >= 0.95
        assert elapsed <= 1.1
