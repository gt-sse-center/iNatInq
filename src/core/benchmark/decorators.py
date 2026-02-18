"""Timing decorators for benchmark operations.

Provides decorators to measure execution time of sync and async functions,
and a benchmark_provider decorator for provider warmup + timed measurement.

Example:
    ```python
    from core.benchmark.decorators import timed, benchmark_provider

    @timed
    def search(query):
        return db.search(query)

    result, elapsed = search("red bird")

    @benchmark_provider(warmup=5, cooldown=0.1)
    async def run_provider(provider, query):
        return await provider.search_async(query)
    ```
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def timed(fn: Callable) -> Callable:
    """Decorator that returns (result, elapsed_seconds) for sync or async functions.

    Args:
        fn: The function to time.

    Returns:
        Wrapped function that returns a tuple of (original_result, elapsed_seconds).
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
            start = time.perf_counter()
            result = await fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return result, elapsed

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return sync_wrapper


def benchmark_provider(
    warmup: int = 5,
    cooldown: float = 0.1,
) -> Callable:
    """Decorator that runs warmup calls before a timed measurement.

    Executes the decorated function `warmup` times (untimed), waits
    `cooldown` seconds, then runs one timed call returning (result, elapsed).

    Args:
        warmup: Number of warmup calls before timing.
        cooldown: Seconds to wait between warmup and measurement.

    Returns:
        Decorator that wraps async functions with warmup + timed behavior.
    """

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
                for _ in range(warmup):
                    await fn(*args, **kwargs)

                await asyncio.sleep(cooldown)

                start = time.perf_counter()
                result = await fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                return result, elapsed

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
            for _ in range(warmup):
                fn(*args, **kwargs)

            time.sleep(cooldown)

            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return result, elapsed

        return sync_wrapper

    return decorator
