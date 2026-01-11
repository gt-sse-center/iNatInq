# Foundation

Foundation utilities and shared infrastructure components.

This directory contains foundational components that provide cross-cutting
functionality for applications, such as logging configuration, retry logic,
circuit breakers, and shared utilities.

## Design Principles

1. **Framework Agnostic**: Foundation components have minimal dependencies
2. **Reusability**: Components can be used across the entire application
3. **Configuration**: Foundation components are configured once at application
   startup
4. **Observability**: Foundation components enhance observability (logging,
   metrics)

## Directory Structure

```
foundation/
├── http.py          # Shared HTTP utilities (retry logic, session management)
├── retry.py         # Retry utilities with exponential backoff
└── logger/          # Structured JSON logging configuration
```

## Modules

### `http.py`

Shared HTTP utilities for retry logic and session management.

**Purpose:**

- Provides reusable HTTP client utilities for consistent retry behavior
- Enables connection pooling across the application
- Used by HTTP clients for better performance

**Functions:**

- `create_retry_session(...) -> requests.Session`: Creates a requests session
  with retry logic for transient failures

**Usage:**

```python
from foundation.http import create_retry_session

# Create session with default retry configuration
session = create_retry_session()

# Create session with custom retry configuration
session = create_retry_session(
    max_retries=5,
    backoff_factor=2.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST", "GET"]
)

# Use with Ollama client for connection pooling
from clients.ollama import OllamaClient
client = OllamaClient(base_url="http://ollama:11434", model="nomic-embed-text")
client.set_session(session)
```

**Retry Strategy:**

- **Default retries**: 3 attempts
- **Backoff factor**: 1.0 seconds (exponential backoff)
- **Status codes**: 429, 500, 502, 503, 504
- **Methods**: POST, GET

**Benefits:**

- Consistent retry behavior across all HTTP clients
- Connection pooling for better performance
- Configurable retry strategy
- Single place to update retry logic

### `retry.py`

Retry utilities with exponential backoff using tenacity.

**Purpose:**

- Provides reusable retry mechanism with exponential backoff
- Integrates with structured logging for observability
- Configurable exception filtering and retry strategies
- Used across the application for consistent retry behavior

**Classes:**

- `RetryWithBackoff`: Retry utility class with exponential backoff and
  structured logging

**Usage:**

```python
from foundation.retry import RetryWithBackoff

# Create retry utility with default settings
retry = RetryWithBackoff()

# Use with custom configuration
retry = RetryWithBackoff(
    max_attempts=5,
    wait_min=1.0,
    wait_max=30.0,
    multiplier=2.0,
    retry_exceptions=(UpstreamError, ClientError, OSError, ValueError),
)

# Call a function with retry logic
result = retry.call(lambda: some_function(arg1, arg2))

# Override retry exceptions for a specific call
result = retry.call(
    func,
    retry_exceptions=(CustomError,),
)
```

**Retry Strategy:**

- **Exponential backoff**: `wait_min * (multiplier ** attempt)`
- **Default retries**: 3 attempts
- **Default wait**: 2.0s minimum, 10.0s maximum
- **Default multiplier**: 1.0 (linear backoff)
- **Exception filtering**: Retries transient errors, never retries programming
  errors

**Features:**

- Structured logging with attempt numbers, wait times, and error details
- Configurable exception types to retry on
- Automatic filtering of unexpected exceptions (TypeError, AttributeError,
  KeyError)
- Per-call exception override support

**Benefits:**

- Consistent retry behavior across the application
- Better observability with structured logging
- Reusable across different modules
- Easy to configure and customize

### `logger.py`

Structured JSON logging configuration for applications.

**Purpose:**

- Provides custom JSON formatter for Python's `logging` module
- Configures uvicorn loggers to use structured JSON output
- Ensures consistent, machine-readable log format across the application

**Components:**

#### `logger.py`

Main logging configuration module.

**Features:**

- **CustomJSONFormatter**: Formats log records as JSON strings
- **LOGGING_CONFIG**: Standard Python `logging.config.dictConfig` compatible
  configuration
- **OpenTelemetry Support**: Handles trace context injection (if
  `LoggingInstrumentor` is used)

**Usage:**

```python
from logging.config import dictConfig
from foundation.logger import LOGGING_CONFIG

dictConfig(config=LOGGING_CONFIG)
```

**Log Format:**

```json
{
  "time": "2025-12-17 20:06:33,976",
  "process_name": "MainProcess",
  "process_id": 1,
  "thread_name": "MainThread",
  "thread_id": 281473789468704,
  "level": "INFO",
  "logger_name": "app.access",
  "pathname": "/app/src/middleware/logger.py",
  "line": 67,
  "message": "request started",
  "request": {
    "path": "/search",
    "method": "GET",
    "remoteAddr": "127.0.0.1"
  }
}
```

**Logger Configuration:**

- `uvicorn`: General uvicorn logs (INFO level)
- `uvicorn.access`: Access logs (WARNING level - suppressed, use
  LoggerMiddleware instead)
- `uvicorn.error`: Error logs (INFO level)
- `uvicorn.asgi`: ASGI logs (INFO level)
- `app.access`: Custom logger for application access logs (INFO level)
- `app.error`: Custom logger for application error logs (ERROR level)

**OpenTelemetry Integration:** If `LoggingInstrumentor` is used, the formatter
automatically includes:

- `otel_service_name`: Service name
- `otel_trace_sampled`: Whether trace is sampled
- `trace_id`: OpenTelemetry trace ID
- `span_id`: OpenTelemetry span ID

**Custom Attributes:** The formatter supports custom attributes added via
`extra` parameter:

- `request`: Request information (from LoggerMiddleware)
- `response`: Response information (from LoggerMiddleware)
- `error`: Error information (from ExceptionHandlerMiddleware)

## Integration

Foundation components are initialized at application startup:

```python
from logging.config import dictConfig
from foundation.logger import LOGGING_CONFIG

def create_app():
    # Initialize logging
    dictConfig(config=LOGGING_CONFIG)

    # ... rest of app setup
```

## Dependencies

Foundation components have minimal dependencies:

- Python standard library (`logging`, `json`)
- Optional: OpenTelemetry (if trace context is needed)

Foundation components do NOT depend on:

- FastAPI
- External service clients
- Business logic services

## Testing

Foundation components can be tested independently:

```python
from logging.config import dictConfig
from foundation.logger import LOGGING_CONFIG
import logging

# Apply configuration
dictConfig(config=LOGGING_CONFIG)

# Test logging
logger = logging.getLogger("app.access")
logger.info("test message", extra={"request": {"path": "/test"}})
```

## Future Additions

Potential additions to the foundation directory:

- **Metrics**: Prometheus metrics configuration
- **Tracing**: OpenTelemetry tracing setup
- **Configuration**: Shared configuration utilities
- **Validation**: Common validation helpers
- **Utilities**: Shared utility functions
