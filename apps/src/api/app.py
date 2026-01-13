"""FastAPI application factory for the pipeline service.

This module provides the `create_app()` function that constructs and configures
the FastAPI application instance. This factory pattern allows:
- Easy testing (create app instances in tests)
- Configuration injection (if needed in the future)
- Clear separation of app creation from app execution

## App Structure

The FastAPI app includes:
- **Title**: "Pipeline API"
- **Version**: "0.1.0"
- **Routes**: All routes from `pipeline.api.routes` (mounted at root)
- **Middleware**: CORS, healthz filter, exception handler
- **Metrics**: Prometheus instrumentation via prometheus-fastapi-instrumentator
- **OpenAPI**: Auto-generated documentation at `/docs` and `/redoc`

## Usage

```python
from pipeline.api.app import create_app

app = create_app()
# Use with uvicorn: uvicorn src.main:app
```

The entrypoint (`src/main.py`) calls `create_app()` and exposes the result as `app`
for uvicorn to discover.

## Middleware

The app includes the following middleware (in order):
1. **CORS**: Enables cross-origin requests (configurable via settings)
2. **HealthzFilterMiddleware**: Suppresses access logs for `/healthz` requests
3. **ExceptionHandlerMiddleware**: Catches unhandled exceptions and converts to HTTP responses

This follows the pattern used in the smarts service (services-main) for consistent
behavior across services.

## OpenAPI Documentation

FastAPI automatically generates OpenAPI 3.0 documentation:
- `/docs`: Interactive Swagger UI
- `/redoc`: ReDoc documentation
- `/openapi.json`: OpenAPI schema JSON

The OpenAPI schema includes detailed descriptions from route docstrings and Pydantic models.
"""

from __future__ import annotations

import logging
from logging.config import dictConfig

from api.middleware import (
    ExceptionHandlerMiddleware,
    HealthzFilterMiddleware,
    LoggerMiddleware,
)
from api.routes import router
from core.exceptions import (
    BadRequestError,
    PipelineError,
    PipelineTimeoutError,
    UpstreamError,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from foundation.logger import LOGGING_CONFIG
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger("uvicorn.error")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    This function constructs a new FastAPI app with:
    - OpenAPI metadata (title, description, version, contact, license)
    - CORS middleware for cross-origin requests
    - Middleware to suppress /healthz request logs
    - Exception handler middleware for error responses
    - All routes from `pipeline.api.routes` mounted at the root
    - Prometheus metrics instrumentation

    Returns:
        A configured `FastAPI` instance ready to use with uvicorn or other ASGI
        servers.

    Note:
        The app is created fresh on each call. In production, you typically call
        this once at startup and reuse the instance. The entrypoint
        (`src/main.py`) calls this function once and exposes the result as a
        module-level `app` variable.

        Middleware order matters: CORS first, then healthz filter, then
        exception handler.  This ensures proper request/response processing and
        error handling.
    """
    #             ╭─────────────────────────────────────────────────────────╮
    #             │                        Logging                          │
    #             ╰─────────────────────────────────────────────────────────╯

    # Configure logging with structured JSON formatter This suppresses uvicorn's
    # default access logger and uses our custom formatter
    dictConfig(config=LOGGING_CONFIG)

    app = FastAPI(
        title="Pipeline API",
        description=(
            "ML pipeline orchestrator service for end-to-end data processing workflows. "
            "Provides endpoints for generating embeddings, storing vectors, and running "
            "distributed Spark and Ray jobs for parallel data processing."
        ),
        version="0.1.0",
        contact={
            "name": "Pipeline Service",
            "url": "https://github.com/modern-web-application",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        # OpenAPI schema customization
        openapi_tags=[
            {
                "name": "health",
                "description": ("Health check endpoints for Kubernetes liveness and readiness probes"),
            },
            {
                "name": "embeddings",
                "description": "Text embedding generation via Ollama using nomic-embed-text model",
            },
            {
                "name": "vector-store",
                "description": "Vector storage and retrieval operations via Qdrant vector database",
            },
            {
                "name": "spark-jobs",
                "description": (
                    "Spark job management via Kubernetes Spark Operator. "
                    "Submit, monitor, and manage distributed Spark jobs for processing "
                    "S3 documents into vector embeddings."
                ),
            },
            {
                "name": "ray-jobs",
                "description": (
                    "Ray job management via Ray Jobs API. "
                    "Submit, monitor, and manage distributed Ray jobs for "
                    "processing S3 documents into vector embeddings."
                ),
            },
        ],
    )

    #             ╭─────────────────────────────────────────────────────────╮
    #             │                        Middleware                       │
    #             ╰─────────────────────────────────────────────────────────╯

    # Order of middleware matters. First in fires first when request received;
    # last after response generated.

    # CORS middleware - must be first to handle preflight requests
    app.add_middleware(
        middleware_class=CORSMiddleware,
        allow_origins=["*"],  # In production, configure via settings
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    # Logger middleware - provides structured JSON logging for requests
    app.add_middleware(LoggerMiddleware)

    # Healthz filter - suppresses access logs for /healthz requests
    # Must be after LoggerMiddleware to filter its logs
    app.add_middleware(HealthzFilterMiddleware)

    # Exception handler - catches unhandled exceptions and converts to HTTP
    # responses
    app.add_middleware(ExceptionHandlerMiddleware)

    #             ╭─────────────────────────────────────────────────────────╮
    #             │                   Exception Handlers                     │
    #             ╰─────────────────────────────────────────────────────────╯

    # Add custom exception handlers for specific exception types
    # These run before the middleware and allow us to handle HTTPException
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTPException from FastAPI routes."""
        logger.exception(
            "http exception",
            extra={"error": {"statuscode": exc.status_code}},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "Client Error", "message": str(exc.detail)},
        )

    @app.exception_handler(BadRequestError)
    async def bad_request_handler(_request: Request, exc: BadRequestError) -> JSONResponse:
        """Handle BadRequestError from service layer."""
        logger.exception("bad request", extra={"error": {"statuscode": 400, "message": str(exc)}})
        return JSONResponse(status_code=400, content={"error": "Bad Request", "message": str(exc)})

    @app.exception_handler(PipelineTimeoutError)
    async def timeout_handler(_request: Request, exc: PipelineTimeoutError) -> JSONResponse:
        """Handle PipelineTimeoutError from service layer."""
        logger.exception(
            "pipeline timeout",
            extra={"error": {"statuscode": 504, "message": str(exc)}},
        )
        return JSONResponse(status_code=504, content={"error": "Gateway Timeout", "message": str(exc)})

    @app.exception_handler(UpstreamError)
    async def upstream_handler(_request: Request, exc: UpstreamError) -> JSONResponse:
        """Handle UpstreamError from service layer."""
        logger.exception("upstream error", extra={"error": {"statuscode": 502, "message": str(exc)}})
        return JSONResponse(status_code=502, content={"error": "Bad Gateway", "message": str(exc)})

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(_request: Request, exc: PipelineError) -> JSONResponse:
        """Handle generic PipelineError from service layer."""
        logger.exception("pipeline error", extra={"error": {"statuscode": 500, "message": str(exc)}})
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "message": str(exc)},
        )

    #             ╭─────────────────────────────────────────────────────────╮
    #             │                        Routers                          │
    #             ╰─────────────────────────────────────────────────────────╯

    app.include_router(router)

    #             ╭─────────────────────────────────────────────────────────╮
    #             │                        Metrics                            │
    #             ╰─────────────────────────────────────────────────────────╯

    # Instrument FastAPI app with Prometheus metrics
    # This automatically exposes /metrics endpoint with standard HTTP metrics
    Instrumentator().instrument(app=app).expose(app=app)

    return app
