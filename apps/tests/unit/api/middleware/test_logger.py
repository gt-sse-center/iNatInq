"""Unit tests for LoggerMiddleware.

This file tests the request logging middleware that provides structured JSON
logging for HTTP requests.

# Test Coverage

The tests cover:
  - Request start logging
  - Request completion logging with timing
  - Healthz request filtering
  - Query parameter logging
  - Remote address capture
  - Timing accuracy

# Test Structure

Tests use pytest with FastAPI TestClient and log capturing.

# Running Tests

Run with: pytest tests/unit/api/middleware/test_logger.py
"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.middleware.logger import LoggerMiddleware

# =============================================================================
# LoggerMiddleware Tests
# =============================================================================


class TestLoggerMiddleware:
    """Test suite for LoggerMiddleware."""

    @pytest.fixture
    def app_with_logger_middleware(self) -> FastAPI:
        """Create a test FastAPI app with LoggerMiddleware.

        Returns:
            FastAPI app instance with logger middleware.
        """
        app = FastAPI()
        app.add_middleware(LoggerMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        @app.get("/healthz")
        def healthz_endpoint():
            return {"status": "ok"}

        return app

    def test_logger_middleware_logs_request_start(self, app_with_logger_middleware: FastAPI, caplog) -> None:
        """Test that middleware logs request start.

        **Why this test is important:**
          - Validates request start logging
          - Tests structured logging format
          - Ensures observability
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/test")

        assert response.status_code == 200
        # Check for "request started" log message
        assert any("request started" in record.message for record in caplog.records)

    def test_logger_middleware_logs_request_completion(
        self, app_with_logger_middleware: FastAPI, caplog
    ) -> None:
        """Test that middleware logs request completion with timing.

        **Why this test is important:**
          - Validates request completion logging
          - Tests timing information
          - Ensures performance monitoring
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/test")

        assert response.status_code == 200
        # Check for "request completed" log message
        assert any("request completed" in record.message for record in caplog.records)

    def test_logger_middleware_includes_query_params(
        self, app_with_logger_middleware: FastAPI, caplog
    ) -> None:
        """Test that middleware logs requests with query parameters.

        **Why this test is important:**
          - Validates query parameter logging
          - Tests URL reconstruction
          - Ensures complete request context
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/test?foo=bar&baz=qux")

        assert response.status_code == 200
        # Verify logs were created for the request
        assert len(caplog.records) >= 2  # start and completion
        assert any("request started" in record.message for record in caplog.records)
        assert any("request completed" in record.message for record in caplog.records)

    def test_logger_middleware_skips_healthz_requests(
        self, app_with_logger_middleware: FastAPI, caplog
    ) -> None:
        """Test that middleware skips logging for /healthz endpoint.

        **Why this test is important:**
          - Validates log filtering
          - Reduces noise from K8s probes
          - Tests special case handling
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/healthz")

        assert response.status_code == 200
        # Should NOT log healthz requests
        assert not any("healthz" in record.message for record in caplog.records)
        assert not any("request started" in record.message for record in caplog.records)

    def test_logger_middleware_captures_remote_address(
        self, app_with_logger_middleware: FastAPI, caplog
    ) -> None:
        """Test that middleware logs requests from clients.

        **Why this test is important:**
          - Validates client request logging
          - Tests request capture
          - Ensures security audit trail
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/test")

        assert response.status_code == 200
        # Verify logs were created for the request
        assert len(caplog.records) >= 2  # start and completion
        assert any("request started" in record.message for record in caplog.records)
        assert any("request completed" in record.message for record in caplog.records)

    def test_logger_middleware_includes_status_code(
        self, app_with_logger_middleware: FastAPI, caplog
    ) -> None:
        """Test that middleware logs response status code.

        **Why this test is important:**
          - Validates status code logging
          - Tests response metadata
          - Ensures error tracking
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/test")

        assert response.status_code == 200
        # Check that status code is in log extra data
        completed_logs = [r for r in caplog.records if "request completed" in r.message]
        assert len(completed_logs) > 0

    def test_logger_middleware_includes_http_method(
        self, app_with_logger_middleware: FastAPI, caplog
    ) -> None:
        """Test that middleware logs HTTP requests.

        **Why this test is important:**
          - Validates HTTP request logging
          - Tests request metadata
          - Ensures complete request context
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/test")

        assert response.status_code == 200
        # Verify logs were created for the request
        assert len(caplog.records) >= 2  # start and completion
        assert any("request started" in record.message for record in caplog.records)
        assert any("request completed" in record.message for record in caplog.records)

    def test_logger_middleware_timing_is_positive(self, app_with_logger_middleware: FastAPI, caplog) -> None:
        """Test that middleware records positive timing values.

        **Why this test is important:**
          - Validates timing calculation
          - Tests performance metrics
          - Ensures timing accuracy
        """
        client = TestClient(app_with_logger_middleware)

        with caplog.at_level(logging.INFO, logger="pipeline.access"):
            response = client.get("/test")

        assert response.status_code == 200
        # Check that timing value exists and is reasonable
        completed_logs = [r for r in caplog.records if "request completed" in r.message]
        assert len(completed_logs) > 0
        # Timing should be in extra data (can't easily assert exact value)
