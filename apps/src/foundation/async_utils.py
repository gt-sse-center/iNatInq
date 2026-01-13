"""Async utilities for resource management and cleanup.

This module provides utilities for handling async resources, particularly
focused on properly closing async clients across different event loop scenarios.
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _create_close_callback(resource_name: str):
    """Create a callback to log the result of an async close operation.

    Args:
        resource_name: Name of the resource being closed (for logging).

    Returns:
        Callback function that logs success or errors.
    """

    def callback(task: asyncio.Task) -> None:
        try:
            task.result()
            logger.debug("Async resource closed successfully", extra={"resource": resource_name})
        except Exception as e:
            logger.exception(
                "Error closing async resource",
                extra={"resource": resource_name, "error": str(e)},
            )

    return callback


async def close_async_resource(resource: Any, resource_name: str, close_method: str = "close") -> None:
    """Close an async resource, handling all event loop scenarios.

    This function handles three possible event loop states:
    1. **Running event loop**: Schedules close as a background task (cannot await
       since this is called from synchronous context). Errors are logged via
       callback but don't propagate to caller.
    2. **Existing but stopped event loop**: Runs close synchronously using
       `run_until_complete()`. Errors are caught and logged.
    3. **No event loop**: Creates a new event loop using `asyncio.run()` to execute
       the async close. Errors are caught and logged.

    Args:
        resource: Async resource to close (e.g., AsyncQdrantClient).
        resource_name: Name for logging (e.g., "qdrant_client").
        close_method: Name of the close method to call (default: "close").

    Example:
        ```python
        # In a synchronous close() method
        def close(self) -> None:
            if self._async_client is not None:
                asyncio.run(close_async_resource(
                    self._async_client,
                    "my_client",
                    "close"
                ))
        ```

    Note:
        This function is designed to be called from synchronous code (like a
        synchronous close() method). It handles the complexity of dealing with
        async cleanup in a sync context.
    """
    close_coro = getattr(resource, close_method)()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Event loop is running (we're being called from async context)
            # Schedule close as a background task and attach callback for error logging
            task = loop.create_task(close_coro)
            task.add_done_callback(_create_close_callback(resource_name))
            logger.debug(
                "Scheduled async resource close as background task",
                extra={"resource": resource_name},
            )
        else:
            # Event loop exists but is not running
            # We can safely run the coroutine to completion
            try:
                loop.run_until_complete(close_coro)
                logger.debug(
                    "Async resource closed successfully",
                    extra={"resource": resource_name},
                )
            except Exception as e:
                logger.exception(
                    "Error closing async resource",
                    extra={"resource": resource_name, "error": str(e)},
                )
    except RuntimeError:
        # No event loop exists
        # Create a new event loop to run the close coroutine
        try:
            asyncio.run(close_coro)
            logger.debug(
                "Async resource closed successfully (new event loop)",
                extra={"resource": resource_name},
            )
        except Exception as e:
            logger.exception(
                "Error closing async resource",
                extra={"resource": resource_name, "error": str(e)},
            )
