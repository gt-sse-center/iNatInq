"""Async utilities for resource management and cleanup.

This module provides utilities for handling async resources, particularly
focused on properly closing async clients across different event loop scenarios.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_coroutine_sync(coro: Any) -> Any:
    """Run an async coroutine from synchronous code, even inside a running event loop.

    Environments like Databricks notebooks and Jupyter run an event loop in the
    background, which makes ``asyncio.run()`` raise
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``.

    This helper detects that situation and offloads the coroutine to a dedicated
    thread with its own event loop, avoiding the conflict.

    Args:
        coro: An awaitable coroutine object.

    Returns:
        The value returned by the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside a running loop (Databricks / Jupyter / IPython).
    # Run the coroutine in a fresh event loop on a worker thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def close_async_resource(resource: Any, resource_name: str, close_method: str = "close") -> None:
    """Close an async resource by awaiting its close method.

    Since this is an async function, it is always called from within a running
    event loop and can directly await the resource's close coroutine.

    Args:
        resource: Async resource to close (e.g., AsyncQdrantClient).
        resource_name: Name for logging (e.g., "qdrant_client").
        close_method: Name of the close method to call (default: "close").

    Example:
        ```python
        # Called via asyncio.run from a synchronous close() method
        def close(self) -> None:
            if self._async_client is not None:
                asyncio.run(close_async_resource(
                    self._async_client,
                    "my_client",
                    "close"
                ))

        # Or directly from async code
        async def cleanup(self) -> None:
            await close_async_resource(self._client, "my_client")
        ```
    """
    try:
        await getattr(resource, close_method)()
        logger.debug(
            "Async resource closed successfully",
            extra={"resource": resource_name},
        )
    except Exception as e:
        logger.exception(
            "Error closing async resource",
            extra={"resource": resource_name, "error": str(e)},
        )
