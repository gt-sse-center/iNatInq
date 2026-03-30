"""Request logging middleware for the pipeline service.

This middleware provides structured JSON logging for HTTP requests, logging both
request start and completion with timing information. This follows the pattern
used in the smarts service (services-main) for consistent logging across services.
"""

import time
from logging import Logger, getLogger

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Use a separate logger for our middleware to avoid conflicts with uvicorn.access
logger: Logger = getLogger("pipeline.access")


class LoggerMiddleware:
    """Pure ASGI middleware to log HTTP requests with structured JSON output.

    This middleware logs:
    - Request start: path, method, remote address
    - Request completion: path, status code, method, duration, remote address

    Usage:
        Add this middleware to your FastAPI app:
        ```python
        from pipeline.api.middleware import LoggerMiddleware

        app.add_middleware(LoggerMiddleware)
        ```

    Note:
        This middleware logs to the "pipeline.access" logger, which is configured
        to output structured JSON. The logs include timing information calculated
        using `time.perf_counter()` for high precision.

        This follows the pattern established in the smarts service (services-main)
        for consistent logging behavior across services.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process request and log start/completion with timing."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]

        # Skip logging for /healthz requests to reduce noise
        if path == "/healthz":
            await self.app(scope, receive, send)
            return

        query_string: bytes = scope.get("query_string", b"")
        if query_string:
            path += f"?{query_string.decode('latin-1')}"

        client = scope.get("client")
        remote_addr: str = client[0] if client else "unknown"
        method: str = scope["method"]

        # Log request start
        logger.info(
            msg="request started",
            extra={
                "request": {
                    "path": path,
                    "method": method,
                    "remoteAddr": remote_addr,
                }
            },
        )

        # Track timing and capture response status code
        start_time: float = time.perf_counter()
        status_code: int = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        finish_time: float = time.perf_counter()

        # Log request completion
        logger.info(
            msg="request completed",
            extra={
                "response": {
                    "path": path,
                    "statuscode": status_code,
                    "method": method,
                    "since": finish_time - start_time,
                    "remoteAddr": remote_addr,
                }
            },
        )
