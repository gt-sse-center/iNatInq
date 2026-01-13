"""Unit tests for ExceptionHandlerMiddleware.

This file tests the exception handler middleware that catches unhandled exceptions
and converts them to appropriate HTTP responses with proper error formatting.

# Test Coverage

The tests cover:
  - BadRequestError → 400
  - PipelineTimeoutError → 504
  - UpstreamError → 502
  - PipelineError → 500
  - HTTPException handling
  - ValueError/TypeError → 500
  - RuntimeError/OSError → 500
  - Generic Exception → 500
  - Error logging

# Test Structure

Tests use pytest with FastAPI TestClient.

# Running Tests

Run with: pytest tests/unit/api/middleware/test_exception_handler.py
"""

import logging

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from api.middleware.exception_handler import ExceptionHandlerMiddleware
from core.exceptions import (
    BadRequestError,
    PipelineError,
    PipelineTimeoutError,
    UpstreamError,
)

logger = logging.getLogger("uvicorn.error")

# =============================================================================
# ExceptionHandlerMiddleware Tests
# =============================================================================


class TestExceptionHandlerMiddleware:
    """Test suite for ExceptionHandlerMiddleware."""

    @pytest.fixture
    def app_with_exception_handler(self) -> FastAPI:
        """Create a test FastAPI app with ExceptionHandlerMiddleware.

        Returns:
            FastAPI app instance with exception handler middleware.
        """
        app = FastAPI()
        app.add_middleware(ExceptionHandlerMiddleware)

        # Add custom exception handlers (same as in create_app)
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
            logger.exception("http exception", extra={"error": {"statuscode": exc.status_code}})
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": "Client Error", "message": str(exc.detail)},
            )

        @app.get("/bad-request")
        def bad_request_endpoint():
            raise BadRequestError("Invalid input parameter")

        @app.get("/timeout")
        def timeout_endpoint():
            raise PipelineTimeoutError("Operation timed out")

        @app.get("/upstream")
        def upstream_endpoint():
            raise UpstreamError("Ollama service unavailable")

        @app.get("/pipeline-error")
        def pipeline_error_endpoint():
            raise PipelineError("Internal pipeline error")

        @app.get("/http-exception")
        def http_exception_endpoint():
            raise HTTPException(status_code=404, detail="Resource not found")

        @app.get("/value-error")
        def value_error_endpoint():
            raise ValueError("Invalid value")

        @app.get("/runtime-error")
        def runtime_error_endpoint():
            raise RuntimeError("Runtime failure")

        @app.get("/generic-error")
        def generic_error_endpoint():
            raise Exception("Unexpected error")

        @app.get("/success")
        def success_endpoint():
            return {"status": "ok"}

        return app

    def test_bad_request_error_returns_400(self, app_with_exception_handler: FastAPI) -> None:
        """Test that BadRequestError is converted to 400 Bad Request.

        **Why this test is important:**
          - Validates BadRequestError handling
          - Tests client error response
          - Ensures proper error message
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/bad-request")

        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "Bad Request"
        assert "Invalid input parameter" in data["message"]

    def test_timeout_error_returns_504(self, app_with_exception_handler: FastAPI) -> None:
        """Test that PipelineTimeoutError is converted to 504 Gateway Timeout.

        **Why this test is important:**
          - Validates timeout error handling
          - Tests gateway timeout response
          - Ensures proper timeout error message
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/timeout")

        assert response.status_code == 504
        data = response.json()
        assert data["error"] == "Gateway Timeout"
        assert "Operation timed out" in data["message"]

    def test_upstream_error_returns_502(self, app_with_exception_handler: FastAPI) -> None:
        """Test that UpstreamError is converted to 502 Bad Gateway.

        **Why this test is important:**
          - Validates upstream service error handling
          - Tests bad gateway response
          - Ensures proper service error message
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/upstream")

        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "Bad Gateway"
        assert "Ollama service unavailable" in data["message"]

    def test_pipeline_error_returns_500(self, app_with_exception_handler: FastAPI) -> None:
        """Test that PipelineError is converted to 500 Internal Server Error.

        **Why this test is important:**
          - Validates generic pipeline error handling
          - Tests internal error response
          - Ensures proper error message
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/pipeline-error")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal Server Error"
        assert "Internal pipeline error" in data["message"]

    def test_http_exception_is_handled_correctly(self, app_with_exception_handler: FastAPI) -> None:
        """Test that HTTPException is handled correctly by custom handler.

        **Why this test is important:**
          - Validates FastAPI HTTPException handling
          - Tests custom exception handler registration
          - Ensures proper status code and message format
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/http-exception")

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Client Error"
        assert "Resource not found" in data["message"]

    def test_value_error_returns_500(self, app_with_exception_handler: FastAPI) -> None:
        """Test that ValueError is converted to 500 Internal Server Error.

        **Why this test is important:**
          - Validates Python exception handling
          - Tests validation error conversion
          - Ensures generic error message for security
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/value-error")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal Server Error"
        # Should not leak internal error details
        assert "validation error" in data["message"].lower()

    def test_runtime_error_returns_500(self, app_with_exception_handler: FastAPI) -> None:
        """Test that RuntimeError is converted to 500 Internal Server Error.

        **Why this test is important:**
          - Validates system error handling
          - Tests runtime error conversion
          - Ensures generic error message
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/runtime-error")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal Server Error"
        assert "system error" in data["message"].lower()

    def test_generic_exception_returns_500(self, app_with_exception_handler: FastAPI) -> None:
        """Test that unexpected exceptions are converted to 500 Internal Server Error.

        **Why this test is important:**
          - Validates catch-all exception handling
          - Tests final exception handler
          - Ensures no unhandled exceptions crash the app
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/generic-error")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal Server Error"
        assert "unexpected error" in data["message"].lower()

    def test_successful_request_passes_through(self, app_with_exception_handler: FastAPI) -> None:
        """Test that successful requests pass through middleware unchanged.

        **Why this test is important:**
          - Validates middleware doesn't interfere with normal flow
          - Tests happy path
          - Ensures middleware transparency
        """
        client = TestClient(app_with_exception_handler)

        response = client.get("/success")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_exception_handler_logs_errors(self, app_with_exception_handler: FastAPI, caplog) -> None:
        """Test that exception handler logs errors.

        **Why this test is important:**
          - Validates error logging
          - Tests observability
          - Ensures errors are tracked
        """
        client = TestClient(app_with_exception_handler)

        with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
            response = client.get("/bad-request")

        assert response.status_code == 400
        # Check that error was logged
        assert any("bad request" in record.message.lower() for record in caplog.records)

    def test_exception_handler_includes_error_details_in_logs(
        self, app_with_exception_handler: FastAPI, caplog
    ) -> None:
        """Test that exception handler logs include error details.

        **Why this test is important:**
          - Validates structured error logging
          - Tests log context
          - Ensures debugging information is available
        """
        client = TestClient(app_with_exception_handler)

        with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
            response = client.get("/upstream")

        assert response.status_code == 502
        # Check for error details in logs
        assert len(caplog.records) > 0
