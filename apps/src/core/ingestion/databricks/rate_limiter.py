"""Ray actor for distributed rate limiting (Databricks runtime)."""

import ray

from foundation.rate_limiter import RateLimiter


@ray.remote
class RateLimiterActor:
    """Ray actor for distributed rate limiting using token bucket algorithm."""

    def __init__(self, rate_per_sec: int) -> None:
        self._limiter = RateLimiter(rate_per_sec)

    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        await self._limiter.acquire_permission()
        return True

    def get_rate(self) -> float:
        """Return the configured requests-per-second rate."""
        return self._limiter.get_rate()
