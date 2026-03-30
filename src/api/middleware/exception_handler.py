"""Exception handler middleware for the pipeline service.

This middleware catches unhandled exceptions and converts them to appropriate
HTTP responses with proper error formatting.
"""

import logging

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from core.exceptions import BadRequestError, PipelineError, PipelineTimeoutError
from foundation.exceptions import UpstreamError

logger = logging.getLogger("uvicorn.error")


class ExceptionHandlerMiddleware:
    """Pure ASGI middleware to handle exceptions and convert them to HTTP responses.

    This middleware catches:
    - PipelineError hierarchy: Converts to appropriate HTTP status codes
        - BadRequestError -> 400
        - PipelineTimeoutError -> 504
        - UpstreamError -> 502
        - PipelineError (base) -> 500
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

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process request and handle any exceptions."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)

        except BadRequestError as e:
            logger.exception(
                "bad request",
                extra={"error": {"statuscode": 400, "message": str(e)}},
            )
            response = JSONResponse(
                status_code=400,
                content={"error": "Bad Request", "message": str(e)},
            )
            await response(scope, receive, send)

        except PipelineTimeoutError as e:
            logger.exception(
                "pipeline timeout",
                extra={"error": {"statuscode": 504, "message": str(e)}},
            )
            response = JSONResponse(
                status_code=504,
                content={"error": "Gateway Timeout", "message": "Request timed out."},
            )
            await response(scope, receive, send)

        except UpstreamError as e:
            logger.exception(
                "upstream error",
                extra={"error": {"statuscode": 502, "message": str(e)}},
            )
            response = JSONResponse(
                status_code=502,
                content={"error": "Bad Gateway", "message": "An upstream service error occurred."},
            )
            await response(scope, receive, send)

        except PipelineError as e:
            logger.exception(
                "pipeline error",
                extra={"error": {"statuscode": 500, "message": str(e)}},
            )
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error", "message": "A pipeline error occurred."},
            )
            await response(scope, receive, send)

        except HTTPException as http_exception:
            logger.exception(
                "http exception",
                extra={
                    "error": {
                        "statuscode": http_exception.status_code,
                    }
                },
            )
            response = JSONResponse(
                status_code=http_exception.status_code,
                content={
                    "error": "Client Error",
                    "message": str(http_exception.detail),
                },
            )
            await response(scope, receive, send)

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
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected validation error occurred.",
                },
            )
            await response(scope, receive, send)

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
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "A system error occurred.",
                },
            )
            await response(scope, receive, send)

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
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred.",
                },
            )
            await response(scope, receive, send)
