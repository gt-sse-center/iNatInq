"""Async utilities for resource management and cleanup.

This module provides utilities for handling async resources, particularly
focused on properly closing async clients across different event loop scenarios.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
