"""Unit tests for core.metrics.classify module.

This file tests the classify_error function that maps exceptions to bounded
error-type labels for Prometheus metrics.

# Test Coverage

The tests cover:
  - UpstreamError with error_kind="circuit_open" → "circuit_open"
  - UpstreamError without error_kind → "upstream"
  - Timeout exceptions (builtin, asyncio, requests) → "timeout"
  - Connection exceptions (builtin, requests) → "connection"
  - Unrecognised exceptions → "unknown"

# Running Tests

Run with: uv run pytest tests/unit/core/metrics/test_classify.py -v
"""

import requests.exceptions

from core.metrics.classify import classify_error
from foundation.exceptions import UpstreamError


class TestClassifyError:
    """Test suite for classify_error."""

    # --- circuit_open -------------------------------------------------------

    def test_upstream_error_with_circuit_open_kind(self) -> None:
        """UpstreamError carrying error_kind='circuit_open' is classified as circuit_open.

        This is the key scenario: circuit breaker decorators convert
        CircuitBreakerError → UpstreamError(error_kind="circuit_open") before
        the metrics decorator sees the exception.
        """
        exc = UpstreamError("service unavailable", error_kind="circuit_open")
        assert classify_error(exc) == "circuit_open"

    # --- upstream -----------------------------------------------------------

    def test_upstream_error_without_kind(self) -> None:
        """Plain UpstreamError (no error_kind) is classified as upstream."""
        exc = UpstreamError("something broke")
        assert classify_error(exc) == "upstream"

    def test_upstream_error_with_none_kind(self) -> None:
        """UpstreamError with explicit error_kind=None is classified as upstream."""
        exc = UpstreamError("something broke", error_kind=None)
        assert classify_error(exc) == "upstream"

    # --- timeout ------------------------------------------------------------

    def test_builtin_timeout_error(self) -> None:
        """Builtin TimeoutError is classified as timeout."""
        assert classify_error(TimeoutError("timed out")) == "timeout"

    def test_asyncio_timeout_error(self) -> None:
        """asyncio.TimeoutError is classified as timeout."""
        assert classify_error(TimeoutError()) == "timeout"

    def test_requests_timeout(self) -> None:
        """requests.exceptions.Timeout is classified as timeout."""
        assert classify_error(requests.exceptions.Timeout("read timed out")) == "timeout"

    # --- connection ---------------------------------------------------------

    def test_builtin_connection_error(self) -> None:
        """Builtin ConnectionError is classified as connection."""
        assert classify_error(ConnectionError("refused")) == "connection"

    def test_os_error(self) -> None:
        """OSError is classified as connection."""
        assert classify_error(OSError("network unreachable")) == "connection"

    def test_requests_connection_error(self) -> None:
        """requests.exceptions.ConnectionError is classified as connection."""
        assert classify_error(requests.exceptions.ConnectionError("failed")) == "connection"

    # --- unknown ------------------------------------------------------------

    def test_unknown_exception(self) -> None:
        """Unrecognised exceptions fall through to unknown."""
        assert classify_error(ValueError("bad value")) == "unknown"

    def test_unknown_runtime_error(self) -> None:
        """RuntimeError is not mapped to any specific category."""
        assert classify_error(RuntimeError("oops")) == "unknown"
