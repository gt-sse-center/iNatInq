"""Ray actor for distributed rate limiting (Databricks runtime)."""

import ray

from foundation.rate_limiter import RateLimiter


@ray.remote
class RateLimiterActor:
    """Ray actor for distributed rate limiting using token bucket algorithm."""

    def __init__(self, rate_per_sec: int) -> None:
        self._limiter = RateLimiter(rate_per_sec)

    async def acquire(self) -> bool:
        await self._limiter.acquire_permission()
        return True

    def get_rate(self) -> float:
        return self._limiter.get_rate()
