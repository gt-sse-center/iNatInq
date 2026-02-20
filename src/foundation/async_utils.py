"""Async utilities for resource management and cleanup.

This module provides utilities for handling async resources, particularly
focused on properly closing async clients across different event loop scenarios.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code, safe for both bare and nested event loops.

    When no event loop is running (normal script execution), this delegates to
    ``asyncio.run()``.  When a loop is already running (e.g. Databricks /
    Jupyter notebooks), it applies ``nest_asyncio`` to the current loop and
    uses ``loop.run_until_complete()`` so the coroutine can execute without
    raising ``RuntimeError``.

    Args:
        coro: The coroutine to execute.

    Returns:
        The value returned by the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import nest_asyncio

    nest_asyncio.apply(loop)
    return loop.run_until_complete(coro)


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
        # Called via run_coroutine from a synchronous close() method
        def close(self) -> None:
            if self._async_client is not None:
                run_coroutine(close_async_resource(
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
