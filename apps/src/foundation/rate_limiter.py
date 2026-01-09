"""Rate limiter utilities for async workloads.

This module provides a token bucket rate limiter implementation for controlling
the rate of async operations, ensuring operations do not exceed specified limits.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter for async workloads.

    Implements a simple rate limiting mechanism using a token bucket algorithm
    to control the rate of async operations. Ensures that operations do not exceed
    the specified rate per second by sleeping when necessary.

    Attributes:
        _interval: Time interval between allowed operations (in seconds).
        _last: Timestamp of the last operation (monotonic time).
        _throttle_lock: Async lock to ensure thread-safe rate limiting.

    Example:
        ```python
        from pipeline.foundation.rate_limiter import RateLimiter

        limiter = RateLimiter(rate_per_sec=5)
        await limiter.acquire_permission()  # Blocks if necessary to respect rate limit
        # Perform operation
        ```

    Note:
        This implementation uses monotonic time to avoid issues with system clock
        adjustments. The internal throttle lock ensures thread-safe operation when
        called concurrently from multiple coroutines.
    """

    def __init__(self, rate_per_sec: int) -> None:
        """Initialize rate limiter.

        Args:
            rate_per_sec: Maximum number of operations allowed per second.
                Must be greater than 0.

        Raises:
            ValueError: If rate_per_sec is less than or equal to 0.
        """
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be greater than 0")
        self._interval = 1.0 / rate_per_sec
        self._last = 0.0
        self._throttle_lock = asyncio.Lock()

    async def acquire_permission(self) -> None:
        """Acquire permission to perform an operation.

        Blocks (via sleep) if necessary to ensure the rate limit is not exceeded.
        Uses monotonic time to avoid issues with system clock adjustments.

        This method is async-safe and can be called concurrently from multiple
        coroutines. The internal lock ensures only one coroutine adjusts the
        rate limiting state at a time.

        Example:
            ```python
            limiter = RateLimiter(rate_per_sec=10)
            for _ in range(100):
                await limiter.acquire_permission()
                await perform_operation()
            ```
        """
        async with self._throttle_lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self._interval:
                await asyncio.sleep(self._interval - delta)
            self._last = time.monotonic()

    def get_rate(self) -> float:
        """Get the current rate limit in requests per second.

        Returns:
            Maximum number of requests per second configured for this limiter.

        Example:
            ```python
            limiter = RateLimiter(rate_per_sec=10)
            rate = limiter.get_rate()  # Returns: 10.0
            ```
        """
        return 1.0 / self._interval
