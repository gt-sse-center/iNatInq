"""Middleware to suppress logging for /healthz endpoint requests.

This middleware prevents uvicorn access logs from being written for healthz
requests, reducing noise from Kubernetes liveness/readiness probes.

The middleware intercepts requests to /healthz and suppresses logging at the
middleware level, which is more reliable than filtering log records after they're
created.
"""

import logging

from starlette.types import ASGIApp, Receive, Scope, Send

# Get the uvicorn access logger
_access_logger = logging.getLogger("uvicorn.access")


class HealthzFilterMiddleware:
    """Pure ASGI middleware to suppress access logs for /healthz endpoint.

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

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process request and suppress logging for /healthz endpoints."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["path"] == "/healthz":
            original_level = _access_logger.level
            _access_logger.setLevel(logging.CRITICAL + 1)  # Effectively disable
            try:
                await self.app(scope, receive, send)
            finally:
                _access_logger.setLevel(original_level)
            return

        await self.app(scope, receive, send)
