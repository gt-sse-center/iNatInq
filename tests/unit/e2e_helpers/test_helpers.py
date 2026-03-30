"""Unit tests for E2E test helper functions.

These are pure unit tests that don't require Docker Compose.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from tests.e2e.helpers import api_url, minio_url, qdrant_url, redis_url, wait_for_service


def test_api_url_with_leading_slash():
    """api_url should return localhost:8000 with the given path."""
    result = api_url("/healthz")
    assert result == "http://localhost:8000/healthz"


def test_api_url_without_leading_slash():
    """api_url should handle paths without leading slash."""
    result = api_url("healthz")
    assert result == "http://localhost:8000/healthz"


def test_api_url_with_query_params():
    """api_url should preserve query parameters."""
    result = api_url("/search/images?q=test&limit=5")
    assert result == "http://localhost:8000/search/images?q=test&limit=5"


def test_minio_url():
    """minio_url should return localhost:9000."""
    assert minio_url() == "http://localhost:9000"


def test_qdrant_url():
    """qdrant_url should return localhost:6333."""
    assert qdrant_url() == "http://localhost:6333"


def test_redis_url():
    """redis_url should return Redis connection string with port 8334."""
    # Redis is on host port 8334 (NOT 6379 which is Ray GCS)
    assert redis_url() == "redis://localhost:8334/0"


def test_wait_for_service_success():
    """wait_for_service should succeed when service responds 200."""
    mock_response = Mock()
    mock_response.status_code = 200

    with patch("httpx.get", return_value=mock_response):
        # Should not raise
        wait_for_service("http://localhost:8000/healthz", timeout=5)


def test_wait_for_service_timeout():
    """wait_for_service should raise TimeoutError when service doesn't respond."""
    with patch("httpx.get", side_effect=httpx.RequestError("Connection refused")):
        with pytest.raises(TimeoutError, match="not healthy after"):
            wait_for_service("http://localhost:8000/healthz", timeout=1)


def test_wait_for_service_retries_on_non_200():
    """wait_for_service should retry on non-200 status codes."""
    mock_response = Mock()
    mock_response.status_code = 503

    with patch("httpx.get", return_value=mock_response):
        with pytest.raises(TimeoutError, match="not healthy after"):
            wait_for_service("http://localhost:8000/healthz", timeout=1)


def test_wait_for_service_succeeds_after_retries():
    """wait_for_service should succeed if service becomes healthy during retry."""
    responses = [
        Mock(status_code=503),
        Mock(status_code=503),
        Mock(status_code=200),
    ]

    with patch("httpx.get", side_effect=responses):
        # Should succeed on third attempt
        wait_for_service("http://localhost:8000/healthz", timeout=5)
