"""Request logging middleware for the pipeline service.

This middleware provides structured JSON logging for HTTP requests, logging both
request start and completion with timing information. This follows the pattern
used in the smarts service (services-main) for consistent logging across services.
"""

import time
from collections.abc import Awaitable, Callable
from logging import Logger, getLogger

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Use a separate logger for our middleware to avoid conflicts with uvicorn.access
logger: Logger = getLogger("pipeline.access")


class LoggerMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests with structured JSON output.

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
        This middleware logs to the "uvicorn.access" logger, which is configured
        to output structured JSON. The logs include timing information calculated
        using `time.perf_counter()` for high precision.

        This follows the pattern established in the smarts service (services-main)
        for consistent logging behavior across services.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and log start/completion with timing.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler in the chain.

        Returns:
            The HTTP response.
        """
        path: str = request.url.path

        # Skip logging for /healthz requests to reduce noise
        if path == "/healthz":
            return await call_next(request)

        if request.query_params:
            path += f"?{request.query_params}"

        # Log request start
        logger.info(
            msg="request started",
            extra={
                "request": {
                    "path": path,
                    "method": request.method,
                    "remoteAddr": request.client.host if request.client else "unknown",
                }
            },
        )

        # Process request and measure duration
        start_time: float = time.perf_counter()
        response: Response = await call_next(request)
        finish_time: float = time.perf_counter()

        # Log request completion
        logger.info(
            msg="request completed",
            extra={
                "response": {
                    "path": path,
                    "statuscode": response.status_code,
                    "method": request.method,
                    "since": finish_time - start_time,
                    "remoteAddr": request.client.host if request.client else "unknown",
                }
            },
        )

        return response
