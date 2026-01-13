"""Shared HTTP utilities for retry logic and session management.

This module provides reusable HTTP client utilities that can be used across the
pipeline package for consistent retry behavior and connection pooling.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 1.0
DEFAULT_STATUS_FORCELIST = [429, 500, 502, 503, 504]
DEFAULT_ALLOWED_METHODS = ["POST", "GET"]


def create_retry_session(
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: list[int] | None = None,
    allowed_methods: list[str] | None = None,
) -> requests.Session:
    """Create a requests session with retry logic for transient failures.

    This function creates a configured requests.Session with automatic retry
    logic for transient HTTP errors. The retry strategy uses exponential backoff
    and can be customized via parameters.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).
        backoff_factor: Base backoff time in seconds (default: 1.0).
        status_forcelist: HTTP status codes that should trigger a retry
            (default: [429, 500, 502, 503, 504]).
        allowed_methods: HTTP methods that are allowed to retry
            (default: ["POST", "GET"]).

    Returns:
        Configured requests.Session with retry strategy.

    Example:
        ```python
        from foundation.http import create_retry_session

        session = create_retry_session(max_retries=5)
        response = session.post("http://api.example.com/endpoint", json={"data": "value"})
        ```

    Note:
        The session mounts retry adapters for both HTTP and HTTPS protocols.
        Connection pooling is handled automatically by requests.Session.
    """
    if status_forcelist is None:
        status_forcelist = DEFAULT_STATUS_FORCELIST
    if allowed_methods is None:
        allowed_methods = DEFAULT_ALLOWED_METHODS

    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
