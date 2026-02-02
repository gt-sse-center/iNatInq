"""Rate limiting utilities for distributed Ray workers.

This module provides:
- RateLimiterActor: Ray actor for distributed rate limiting
- RayActorRateLimiter: Adapter to use Ray actor with local interface
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ray

from foundation.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from ray import ObjectRef


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
        from core.ingestion.shared import RateLimiterActor

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

        This method delegates to the foundation RateLimiter's acquire_permission
        method and returns True when permission is granted (Ray interface convention).

        Returns:
            True when permission is granted.
        """
        await self._limiter.acquire_permission()
        return True

    def get_rate(self) -> float:
        """Get the current rate limit in requests per second.

        Returns:
            Maximum number of requests per second configured for this limiter.
        """
        return self._limiter.get_rate()


class RayActorRateLimiter:
    """Adapter that wraps a Ray rate limiter actor as a local RateLimiter.

    This allows processing pipelines to use a distributed rate limiter
    actor transparently through the same interface as a local RateLimiter.

    Example:
        ```python
        from core.ingestion.shared import RateLimiterActor, RayActorRateLimiter

        # Create distributed actor
        actor = RateLimiterActor.remote(rate_per_sec=10)

        # Wrap for local interface
        limiter = RayActorRateLimiter(actor)
        await limiter.acquire()
        ```
    """

    def __init__(self, actor: ObjectRef[Any]) -> None:
        """Initialize with a Ray rate limiter actor.

        Args:
            actor: Ray actor with an `acquire` remote method.
        """
        self._actor = actor

    async def acquire(self) -> None:
        """Acquire permission from the distributed rate limiter."""
        await self._actor.acquire.remote()

    def get_rate(self) -> float:
        """Get the rate limit (not used, but required by interface)."""
        return 0.0
