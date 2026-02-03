# Middleware

FastAPI middleware for request handling, logging, and error management.

This directory contains middleware components that enhance the FastAPI
application with cross-cutting concerns like logging, exception handling, and
request filtering.

## Design Principles

1. **Order Matters**: Middleware order determines execution order (first added =
   first executed on request, last executed on response)
2. **Non-Blocking**: Middleware should not block request processing
   unnecessarily
3. **Error Handling**: Middleware should handle errors gracefully and provide
   structured responses
4. **Observability**: Middleware should enhance logging and metrics without
   duplicating effort

## Middleware Stack

The middleware is added in this order (in `pipeline.api.app.create_app()`):

1. **CORSMiddleware** (FastAPI built-in): Handles CORS preflight requests
2. **LoggerMiddleware**: Logs request start/completion with structured JSON
3. **HealthzFilterMiddleware**: Suppresses access logs for `/healthz` requests
4. **ExceptionHandlerMiddleware**: Catches unhandled exceptions and returns
   structured error responses

## Modules

### `logger.py`

Request logging middleware with structured JSON output.

**Purpose:**

- Logs request start with path, method, and remote address
- Logs request completion with status code, duration, and remote address
- Skips logging for `/healthz` requests to reduce noise

**Features:**

- Structured JSON logging (compatible with log aggregation systems)
- High-precision timing using `time.perf_counter()`
- Automatic `/healthz` filtering

**Usage:**

```python
from pipeline.api.middleware import LoggerMiddleware

app.add_middleware(LoggerMiddleware)
```

**Log Output:**

```json
{
  "time": "2025-12-17 20:06:33,976",
  "level": "INFO",
  "logger_name": "pipeline.access",
  "message": "request started",
  "request": {
    "path": "/search",
    "method": "GET",
    "remoteAddr": "127.0.0.1"
  }
}
```

### `healthz_filter.py`

Middleware to suppress access logs for `/healthz` endpoint.

**Purpose:**

- Prevents uvicorn access logs from being written for healthz requests
- Reduces log noise from Kubernetes liveness/readiness probes

**How It Works:**

- Temporarily disables the uvicorn access logger during `/healthz` request
  processing
- Re-enables the logger after the request completes

**Usage:**

```python
from pipeline.api.middleware import HealthzFilterMiddleware

app.add_middleware(HealthzFilterMiddleware)
```

**Note:** This middleware must be added after `LoggerMiddleware` to filter its
logs as well.

### `exception_handler.py`

Exception handling middleware for structured error responses.

**Purpose:**

- Catches unhandled exceptions from route handlers
- Converts exceptions to structured JSON error responses
- Maps `PipelineError` hierarchy to appropriate HTTP status codes

**Error Mapping:**

- `HTTPException` → Returns appropriate status code with error details
- `BadRequestError` → 400 Bad Request
- `UpstreamError` → 502 Bad Gateway
- `PipelineTimeoutError` → 504 Gateway Timeout
- `PipelineError` (base) → 500 Internal Server Error
- Generic `Exception` → 500 Internal Server Error (with generic message)

**Usage:**

```python
from pipeline.api.middleware import ExceptionHandlerMiddleware

app.add_middleware(ExceptionHandlerMiddleware)
```

**Error Response Format:**

```json
{
  "error": "Bad Gateway",
  "message": "Ollama error 404: model not found"
}
```

**Logging:** All exceptions are logged with full traceback to `uvicorn.error`
logger before returning the error response.

## Middleware Order

The order of middleware addition matters:

```python
# 1. CORS - must be first to handle preflight requests
app.add_middleware(CORSMiddleware, ...)

# 2. Logger - logs all requests (before filtering)
app.add_middleware(LoggerMiddleware)

# 3. Healthz Filter - suppresses /healthz logs
app.add_middleware(HealthzFilterMiddleware)

# 4. Exception Handler - catches all exceptions (should be last)
app.add_middleware(ExceptionHandlerMiddleware)
```

**Execution Flow:**

```
Request → CORS → Logger → HealthzFilter → ExceptionHandler → Route Handler
                                                                    ↓
Response ← CORS ← Logger ← HealthzFilter ← ExceptionHandler ←──────┘
```

## Testing

Middleware can be tested by creating a test FastAPI app:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipeline.api.middleware import LoggerMiddleware, ExceptionHandlerMiddleware

app = FastAPI()
app.add_middleware(LoggerMiddleware)
app.add_middleware(ExceptionHandlerMiddleware)

@app.get("/test")
def test():
    return {"status": "ok"}

client = TestClient(app)
response = client.get("/test")
assert response.status_code == 200
```

## Dependencies

- `fastapi`: FastAPI framework
- `starlette.middleware.base.BaseHTTPMiddleware`: Base middleware class
- `pipeline.errors`: Error hierarchy for exception handling
- `pipeline.foundation.logger`: Structured logging configuration
