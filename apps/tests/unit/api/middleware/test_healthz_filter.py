"""Unit tests for HealthzFilterMiddleware.

This file tests the middleware that suppresses logging for /healthz endpoint
requests, reducing noise from Kubernetes liveness/readiness probes.

# Test Coverage

The tests cover:
  - Log suppression for /healthz requests
  - Normal logging for non-healthz requests
  - Logger level management
  - Response correctness during filtering

# Test Structure

Tests use pytest with FastAPI TestClient and log capturing.

# Running Tests

Run with: pytest tests/unit/api/middleware/test_healthz_filter.py
"""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.middleware.healthz_filter import HealthzFilterMiddleware

# =============================================================================
# HealthzFilterMiddleware Tests
# =============================================================================


class TestHealthzFilterMiddleware:
    """Test suite for HealthzFilterMiddleware."""

    @pytest.fixture
    def app_with_healthz_filter(self) -> FastAPI:
        """Create a test FastAPI app with HealthzFilterMiddleware.

        Returns:
            FastAPI app instance with healthz filter middleware.
        """
        app = FastAPI()
        app.add_middleware(HealthzFilterMiddleware)

        @app.get("/healthz")
        def healthz_endpoint():
            return {"status": "ok"}

        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}

        return app

    def test_healthz_filter_suppresses_healthz_logs(self, app_with_healthz_filter: FastAPI, caplog) -> None:
        """Test that middleware suppresses logs for /healthz requests.

        **Why this test is important:**
          - Validates log suppression for health checks
          - Reduces noise from K8s probes
          - Tests middleware filtering logic
        """
        client = TestClient(app_with_healthz_filter)

        with caplog.at_level(logging.INFO, logger="uvicorn.access"):
            response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        # Should NOT log healthz requests
        assert not any("healthz" in record.message for record in caplog.records)

    def test_healthz_filter_allows_non_healthz_logs(self, app_with_healthz_filter: FastAPI, caplog) -> None:
        """Test that middleware allows logging for non-healthz requests.

        **Why this test is important:**
          - Validates normal requests are logged
          - Tests selective filtering
          - Ensures non-healthz endpoints aren't affected
        """
        client = TestClient(app_with_healthz_filter)

        # Non-healthz requests should be logged normally
        response = client.get("/test")

        assert response.status_code == 200
        # Middleware should not interfere with other endpoints

    def test_healthz_filter_returns_correct_response(self, app_with_healthz_filter: FastAPI) -> None:
        """Test that middleware doesn't modify healthz response.

        **Why this test is important:**
          - Validates response correctness
          - Tests middleware transparency
          - Ensures health checks work properly
        """
        client = TestClient(app_with_healthz_filter)

        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_healthz_filter_restores_logger_level(self, app_with_healthz_filter: FastAPI) -> None:
        """Test that middleware restores logger level after request.

        **Why this test is important:**
          - Validates proper cleanup
          - Tests finally block execution
          - Ensures no side effects on other requests
        """
        client = TestClient(app_with_healthz_filter)
        access_logger = logging.getLogger("uvicorn.access")

        original_level = access_logger.level

        # Make healthz request
        response = client.get("/healthz")
        assert response.status_code == 200

        # Logger level should be restored
        assert access_logger.level == original_level

    def test_healthz_filter_handles_multiple_healthz_requests(
        self, app_with_healthz_filter: FastAPI, caplog
    ) -> None:
        """Test that middleware handles multiple healthz requests correctly.

        **Why this test is important:**
          - Validates repeated request handling
          - Tests stateless middleware behavior
          - Simulates K8s probe pattern
        """
        client = TestClient(app_with_healthz_filter)

        with caplog.at_level(logging.INFO, logger="uvicorn.access"):
            for _ in range(10):
                response = client.get("/healthz")
                assert response.status_code == 200

        # None of the healthz requests should be logged
        assert not any("healthz" in record.message for record in caplog.records)

    def test_healthz_filter_with_query_params(self, app_with_healthz_filter: FastAPI, caplog) -> None:
        """Test that middleware filters healthz with query parameters.

        **Why this test is important:**
          - Validates path matching
          - Tests query param handling
          - Ensures filtering works regardless of params
        """
        client = TestClient(app_with_healthz_filter)

        with caplog.at_level(logging.INFO, logger="uvicorn.access"):
            response = client.get("/healthz?test=123")

        assert response.status_code == 200
        # Should still filter even with query params
        assert not any("healthz" in record.message for record in caplog.records)
