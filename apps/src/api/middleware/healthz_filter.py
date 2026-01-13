"""Middleware to suppress logging for /healthz endpoint requests.

This middleware prevents uvicorn access logs from being written for healthz
requests, reducing noise from Kubernetes liveness/readiness probes.

The middleware intercepts requests to /healthz and suppresses logging at the
middleware level, which is more reliable than filtering log records after they're
created.
"""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import Request, Response
    from collections.abc import Awaitable, Callable

# Get the uvicorn access logger
_access_logger = logging.getLogger("uvicorn.access")


class HealthzFilterMiddleware(BaseHTTPMiddleware):
    """Middleware to suppress access logs for /healthz endpoint.

    This middleware filters out uvicorn access log entries for /healthz requests
    by temporarily disabling the access logger during healthz request processing.

    Usage:
        Add this middleware to your FastAPI app:
        ```python
        from pipeline.api.middleware import HealthzFilterMiddleware

        app.add_middleware(HealthzFilterMiddleware)
        ```

    Note:
        This approach is more reliable than filtering log records because it
        prevents the log entries from being created in the first place, rather
        than filtering them after creation.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and suppress logging for /healthz endpoints.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler in the chain.

        Returns:
            The HTTP response.
        """
        # If this is a healthz request, temporarily disable the access logger
        if request.url.path == "/healthz":
            # Disable all handlers for the access logger
            original_level = _access_logger.level
            _access_logger.setLevel(logging.CRITICAL + 1)  # Effectively disable

            try:
                response = await call_next(request)
            finally:
                # Restore the original log level
                _access_logger.setLevel(original_level)

            return response

        # For non-healthz requests, process normally
        return await call_next(request)
