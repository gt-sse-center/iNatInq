"""Exception handler middleware for the pipeline service.

This middleware catches unhandled exceptions and converts them to appropriate
HTTP responses with proper error formatting.
"""

import logging
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.exceptions import BadRequestError, PipelineError, PipelineTimeoutError, UpstreamError

logger = logging.getLogger("uvicorn.error")


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and convert them to HTTP responses.

    This middleware catches:
    - PipelineError hierarchy: Converts to appropriate HTTP status codes
        - BadRequestError → 400
        - PipelineTimeoutError → 504
        - UpstreamError → 502
        - PipelineError (base) → 500
    - HTTPException: Returns appropriate HTTP status with error details
    - Other exceptions: Returns 500 with generic error message

    Usage:
        Add this middleware to your FastAPI app:
        ```python
        from pipeline.api.middleware import ExceptionHandlerMiddleware

        app.add_middleware(ExceptionHandlerMiddleware)
        ```

    Note:
        This should be added after other middleware but before routes to catch
        exceptions from route handlers.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and handle any exceptions.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler in the chain.

        Returns:
            The HTTP response, or an error response if an exception occurred.
        """
        try:
            return await call_next(request)

        except BadRequestError as e:
            logger.exception(
                "bad request",
                extra={"error": {"statuscode": 400, "message": str(e)}},
            )
            return JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": str(e)},
            )

        except PipelineTimeoutError as e:
            logger.exception(
                "pipeline timeout",
                extra={"error": {"statuscode": 504, "message": str(e)}},
            )
            return JSONResponse(
                status_code=504,
                content={"error": "Gateway Timeout", "message": str(e)},
            )

        except UpstreamError as e:
            logger.exception(
                "upstream error",
                extra={"error": {"statuscode": 502, "message": str(e)}},
            )
            return JSONResponse(
                status_code=502,
                content={"error": "Bad Gateway", "message": str(e)},
            )

        except PipelineError as e:
            logger.exception(
                "pipeline error",
                extra={"error": {"statuscode": 500, "message": str(e)}},
            )
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "message": str(e)},
            )

        except HTTPException as http_exception:
            logger.exception(
                "http exception",
                extra={
                    "error": {
                        "statuscode": http_exception.status_code,
                    }
                },
            )

            return JSONResponse(
                status_code=http_exception.status_code,
                content={
                    "error": "Client Error",
                    "message": str(http_exception.detail),
                },
            )

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            # Common Python errors that should be caught
            logger.exception(
                "validation error",
                extra={
                    "error": {
                        "statuscode": 500,
                        "message": str(e),
                    }
                },
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected validation error occurred.",
                },
            )

        except (RuntimeError, OSError) as e:
            # System-level errors
            logger.exception(
                "system error",
                extra={
                    "error": {
                        "statuscode": 500,
                        "message": str(e),
                    }
                },
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "A system error occurred.",
                },
            )

        except Exception as e:
            # Final catch-all for any other unhandled exceptions.
            # This is intentionally broad to catch any unexpected errors and return
            # a proper HTTP 500 response rather than crashing the application.
            logger.exception(
                "internal error",
                extra={
                    "error": {
                        "statuscode": 500,
                        "message": str(e),
                    }
                },
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred.",
                },
            )
