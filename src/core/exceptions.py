"""Exception hierarchy for the pipeline package.

This module defines a framework-agnostic exception hierarchy that allows:
- Service layer code to raise errors without HTTP dependencies
- API routes to translate exceptions into appropriate HTTP status codes
- CLI tools and tests to handle errors consistently

## Exception Hierarchy

All exceptions inherit from `PipelineError`:

- `BadRequestError`: Client sent invalid input (maps to HTTP 400)
- `UpstreamError`: Dependency service failed (maps to HTTP 502)
- `PipelineTimeoutError`: Operation timed out (maps to HTTP 504)

## HTTP Mapping

The `pipeline.api.routes` module translates these exceptions:

```python
BadRequestError          → HTTP 400 (Bad Request)
UpstreamError            → HTTP 502 (Bad Gateway)
PipelineTimeoutError     → HTTP 504 (Gateway Timeout)
PipelineError (base)     → HTTP 500 (Internal Server Error)
```

## Usage

```python
from core.exceptions import BadRequestError, UpstreamError

if invalid_input:
    raise BadRequestError("Invalid input provided")

try:
    result = call_external_service()
except Exception as e:
    raise UpstreamError(f"Service call failed: {e}") from e
```
"""

# Re-export UpstreamError from foundation for compatibility
from foundation.exceptions import UpstreamError  # noqa: F401


class PipelineError(Exception):
    """Base exception class for all pipeline-related errors.

    All custom exceptions in this module inherit from `PipelineError` to allow
    catch-all error handling and consistent error translation in API routes.

    This exception maps to HTTP 500 (Internal Server Error) if not caught
    and translated by a more specific exception handler.
    """


class BadRequestError(PipelineError):
    """Exception raised when the client provides invalid input.

    This exception indicates a client error (e.g., missing required fields,
    invalid field values, length mismatches). It maps to HTTP 400 (Bad Request)
    in API routes.

    Examples:
        - Metadata list length doesn't match texts length
        - Empty texts list when non-empty is required
        - Invalid collection name format
    """


class _UpstreamError(PipelineError):
    """Exception raised when an upstream dependency service fails.

    This exception indicates that a required external service (MinIO, Spark,
    Ollama, Qdrant, or Kubernetes API) is unavailable, returned an error, or
    failed to complete a request. It maps to HTTP 502 (Bad Gateway) in API
    routes.

    Examples:
        - Ollama service is unreachable or returns an error
        - Qdrant upsert operation fails
        - Kubernetes API returns an error when creating a Job
        - S3/MinIO bucket operation fails
        - Spark job fails to start or crashes

    Note:
        This is a catch-all for dependency failures. In production, you might
        want to create more specific exceptions (e.g., `OllamaError`,
        `QdrantError`) for better error categorization and monitoring.

        This class is kept for documentation but UpstreamError is re-exported
        from foundation.exceptions to ensure consistency across the codebase.
    """


class PipelineTimeoutError(PipelineError):
    """Exception raised when an operation exceeds its timeout.

    This exception indicates that a long-running operation (e.g., waiting for a
    Spark job to complete) exceeded its configured timeout. It maps to HTTP 504
    (Gateway Timeout) in API routes.

    Examples:
        - Spark job takes longer than `timeout_s` to complete
        - Ollama embedding request exceeds timeout
        - Waiting for Kubernetes Job status times out

    Note:
        We use `PipelineTimeoutError` instead of Python's built-in
        `TimeoutError`
        to avoid conflicts and maintain our exception hierarchy.
    """
