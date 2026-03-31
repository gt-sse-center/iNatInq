"""Helper functions for E2E tests.

Provides utilities for URL construction, service health checking, and common
operations across E2E tests.
"""

import time

import httpx


def api_url(path: str) -> str:
    """Return the full URL for a Pipeline API endpoint.

    Args:
        path: API path (with or without leading slash).

    Returns:
        Full URL with localhost:8000 base.

    Example:
        >>> api_url("/healthz")
        'http://localhost:8000/healthz'
        >>> api_url("search/images?q=test")
        'http://localhost:8000/search/images?q=test'
    """
    if not path.startswith("/"):
        path = f"/{path}"
    return f"http://localhost:8000{path}"


def minio_url() -> str:
    """Return the MinIO API endpoint URL.

    Returns:
        MinIO URL at localhost:9000.
    """
    return "http://localhost:9000"


def qdrant_url() -> str:
    """Return the Qdrant API endpoint URL.

    Returns:
        Qdrant URL at localhost:6333.
    """
    return "http://localhost:6333"


def redis_url() -> str:
    """Return the Redis connection string.

    Returns:
        Redis URL at localhost:8334/0 (NOT 6379 which is Ray GCS).
    """
    return "redis://localhost:8334/0"


def wait_for_service(url: str, timeout: int = 120) -> None:
    """Poll a service URL until it responds with 200 or timeout expires.

    Args:
        url: Service health check URL to poll.
        timeout: Maximum seconds to wait (default: 120).

    Raises:
        TimeoutError: If service doesn't become healthy within timeout.

    Example:
        >>> wait_for_service("http://localhost:8000/healthz", timeout=30)
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.5)

    elapsed = time.time() - start
    raise TimeoutError(f"Service at {url} not healthy after {elapsed:.1f}s")
