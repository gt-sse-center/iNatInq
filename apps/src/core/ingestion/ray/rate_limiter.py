"""Ray actor for distributed rate limiting.

This module provides a Ray actor that wraps the foundation RateLimiter for
distributed rate limiting across multiple Ray workers.
"""

import ray

from foundation.rate_limiter import RateLimiter


@ray.remote
class RateLimiterActor:
    """Ray actor for distributed rate limiting using token bucket algorithm.

    This actor wraps the foundation RateLimiter class to provide distributed
    rate limiting coordination across multiple Ray workers. The core algorithm
    is provided by the foundation RateLimiter, and this actor adds the Ray
    distribution layer.

    Attributes:
        _limiter: Foundation RateLimiter instance that provides the core algorithm.

    Example:
        ```python
        import ray
        from core.ingestion.ray.rate_limiter import RateLimiterActor

        limiter = RateLimiterActor.remote(rate_per_sec=5)
        await limiter.acquire.remote()
        ```

    Note:
        This class uses composition to delegate rate limiting logic to the
        foundation RateLimiter class, ensuring a single source of truth for
        the rate limiting algorithm.
    """

    def __init__(self, rate_per_sec: int) -> None:
        """Initialize rate limiter actor.

        Args:
            rate_per_sec: Maximum requests per second.
        """
        self._limiter = RateLimiter(rate_per_sec)

    async def acquire(self) -> bool:
        """Acquire permission to make a request.

        This method delegates to the foundation RateLimiter's acquire method
        and returns True when permission is granted (Ray interface convention).

        Returns:
            True when permission is granted.
        """
        await self._limiter.acquire()
        return True

    def get_rate(self) -> float:
        """Get the current rate limit in requests per second.

        Returns:
            Maximum number of requests per second configured for this limiter.
        """
        return self._limiter.get_rate()

