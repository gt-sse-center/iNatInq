"""Error classification for bounded metrics labels.

Maps exceptions to one of five error types for use in Prometheus metric
labels. This prevents cardinality explosion from unbounded exception messages.

The ``isinstance`` check ordering is **critical** — ``TimeoutError`` inherits
from ``OSError`` in Python 3.11+, so it must be checked first. Similarly,
``requests`` library exceptions inherit from ``OSError`` and must be checked
before the builtin ``OSError`` catch-all.
"""

import asyncio
from typing import Literal

import aiobreaker
import pybreaker
import requests.exceptions

from foundation.exceptions import UpstreamError

ErrorType = Literal["timeout", "connection", "upstream", "circuit_open", "unknown"]


def classify_error(exc: BaseException) -> ErrorType:
    """Classify an exception into a bounded error_type label.

    Args:
        exc: The exception to classify.

    Returns:
        One of: ``"timeout"``, ``"connection"``, ``"upstream"``,
        ``"circuit_open"``, ``"unknown"``.

    Example:
        ```python
        try:
            response = requests.get(url, timeout=5)
        except Exception as e:
            error_type = classify_error(e)
            CLIENT_ERRORS_TOTAL.labels(
                client="api",
                operation="get",
                error_type=error_type
            ).inc()
            raise
        ```
    """
    # Check requests.Timeout BEFORE TimeoutError (requests.Timeout inherits from requests.ConnectionError -> OSError)
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"

    # Check TimeoutError BEFORE OSError (TimeoutError inherits from OSError in Python 3.11+)
    # asyncio.TimeoutError is the same as TimeoutError in Python 3.11+
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"

    # Check requests.ConnectionError BEFORE builtin ConnectionError/OSError
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "connection"

    # Check builtin connection errors (ConnectionError and OSError)
    if isinstance(exc, (ConnectionError, OSError)):
        return "connection"

    # Check UpstreamError (application-level wrapper for external service failures)
    if isinstance(exc, UpstreamError):
        return "upstream"

    # Check circuit breaker errors (both sync pybreaker and async aiobreaker)
    if isinstance(exc, (pybreaker.CircuitBreakerError, aiobreaker.CircuitBreakerError)):
        return "circuit_open"

    # Fallback for unrecognized exceptions
    return "unknown"
