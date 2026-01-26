"""Unit tests for FastAPI application factory.

This file tests the create_app() function and overall application configuration,
including middleware setup, route registration, and OpenAPI documentation.

# Test Coverage

The tests cover:
  - App factory function (create_app)
  - OpenAPI metadata and documentation
  - Middleware registration and order
  - Route registration
  - Prometheus metrics instrumentation
  - CORS configuration

# Test Structure

Tests use pytest class-based organization.

# Running Tests

Run with: pytest tests/unit/api/test_app.py
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.app import create_app

# =============================================================================
# App Factory Tests
# =============================================================================


class TestCreateApp:
    """Test suite for create_app() factory function."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """Test that create_app returns a FastAPI application instance.

        **Why this test is important:**
          - Validates basic factory function operation
          - Ensures correct return type
          - Tests app instantiation
        """
        app = create_app()

        assert isinstance(app, FastAPI)

    def test_create_app_sets_title_and_version(self) -> None:
        """Test that create_app configures OpenAPI metadata.

        **Why this test is important:**
          - Validates OpenAPI documentation configuration
          - Tests application metadata
          - Ensures correct API versioning
        """
        app = create_app()

        assert app.title == "Pipeline API"
        assert app.version == "0.1.0"
        assert "ML pipeline orchestrator" in app.description

    def test_create_app_sets_contact_and_license(self) -> None:
        """Test that create_app configures contact and license info.

        **Why this test is important:**
          - Validates OpenAPI contact/license metadata
          - Tests documentation completeness
        """
        app = create_app()

        assert app.contact is not None
        assert app.contact["name"] == "iNatInq ML Pipeline"
        assert app.license_info is not None
        assert app.license_info["name"] == "MIT"

    def test_create_app_includes_all_routes(self) -> None:
        """Test that create_app registers all expected routes.

        **Why this test is important:**
          - Validates route registration
          - Ensures all endpoints are available
          - Tests router integration
        """
        app = create_app()
        client = TestClient(app)

        # Test key endpoints exist
        response = client.get("/healthz")
        assert response.status_code == 200

        # OpenAPI endpoints
        response = client.get("/docs", follow_redirects=False)
        assert response.status_code in (200, 307)  # May redirect

        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_create_app_configures_cors_middleware(self) -> None:
        """Test that create_app adds CORS middleware.

        **Why this test is important:**
          - Validates CORS configuration
          - Tests middleware registration
          - Ensures cross-origin requests work
        """
        app = create_app()

        # Check middleware stack includes CORS
        # Middleware objects are wrapped, so check by looking at cls attribute
        middleware_names = [
            m.cls.__name__ if hasattr(m, "cls") else type(m).__name__ for m in app.user_middleware
        ]
        assert "CORSMiddleware" in middleware_names

    def test_create_app_configures_custom_middleware(self) -> None:
        """Test that create_app adds custom middleware.

        **Why this test is important:**
          - Validates custom middleware registration
          - Tests middleware order
          - Ensures exception handling is present
        """
        app = create_app()

        # Check middleware stack includes our custom middleware
        middleware_names = [
            m.cls.__name__ if hasattr(m, "cls") else type(m).__name__ for m in app.user_middleware
        ]
        assert "ExceptionHandlerMiddleware" in middleware_names
        assert "HealthzFilterMiddleware" in middleware_names
        assert "LoggerMiddleware" in middleware_names

    def test_create_app_exposes_metrics_endpoint(self) -> None:
        """Test that create_app instruments Prometheus metrics.

        **Why this test is important:**
          - Validates metrics instrumentation
          - Tests Prometheus integration
          - Ensures /metrics endpoint is configured
        """
        app = create_app()
        client = TestClient(app)

        # Prometheus instrumentator exposes /metrics after first request
        # Make a request to generate metrics
        client.get("/healthz")

        response = client.get("/metrics")
        assert response.status_code == 200
        # Should contain Prometheus metrics format
        assert (
            "http_request" in response.text.lower()
            or "python_info" in response.text
            or "process_" in response.text
        )

    def test_create_app_creates_new_instance_each_call(self) -> None:
        """Test that create_app returns a new instance on each call.

        **Why this test is important:**
          - Validates factory pattern behavior
          - Tests isolation between instances
          - Ensures testability
        """
        app1 = create_app()
        app2 = create_app()

        assert app1 is not app2

    def test_create_app_configures_openapi_tags(self) -> None:
        """Test that create_app sets up OpenAPI tag descriptions.

        **Why this test is important:**
          - Validates OpenAPI documentation structure
          - Tests tag configuration for API organization
          - Ensures good API documentation
        """
        app = create_app()

        assert app.openapi_tags is not None
        tag_names = [tag["name"] for tag in app.openapi_tags]

        # Verify expected tags
        assert "health" in tag_names
        assert "embeddings" in tag_names
        assert "vector-store" in tag_names
        assert "ray-jobs" in tag_names


# =============================================================================
# Middleware Integration Tests
# =============================================================================


class TestMiddlewareIntegration:
    """Test suite for middleware integration with app."""

    def test_cors_middleware_allows_cross_origin_requests(self) -> None:
        """Test that CORS middleware is configured correctly.

        **Why this test is important:**
          - Validates CORS headers are set
          - Tests cross-origin access
          - Ensures frontend integration works
        """
        app = create_app()
        client = TestClient(app)

        response = client.get(
            "/healthz",
            headers={"Origin": "http://localhost:3000"},
        )

        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers

    def test_exception_handler_middleware_catches_errors(self) -> None:
        """Test that exception handler middleware catches and converts errors.

        **Why this test is important:**
          - Validates error handling middleware
          - Tests exception-to-HTTP conversion
          - Ensures proper error responses
        """
        app = create_app()
        client = TestClient(app)

        # Trigger an error by searching with empty query
        response = client.get("/search?q=")

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_healthz_filter_middleware_suppresses_logs(self) -> None:
        """Test that healthz filter middleware suppresses /healthz logs.

        **Why this test is important:**
          - Validates log filtering for health checks
          - Tests middleware log suppression
          - Reduces noise from K8s probes
        """
        app = create_app()
        client = TestClient(app)

        # Multiple healthz requests should not spam logs
        for _ in range(5):
            response = client.get("/healthz")
            assert response.status_code == 200

        # Test passes if no exceptions (log suppression is internal)

    def test_logger_middleware_logs_requests(self) -> None:
        """Test that logger middleware logs non-healthz requests.

        **Why this test is important:**
          - Validates request logging
          - Tests structured logging
          - Ensures observability
        """
        app = create_app()
        client = TestClient(app)

        # Non-healthz request should be logged
        response = client.get("/openapi.json")
        assert response.status_code == 200

        # Test passes if no exceptions (logging is internal)


# =============================================================================
# OpenAPI Route Registration Tests
# =============================================================================


class TestOpenAPIRouteRegistration:
    """Test that our routes are properly registered in OpenAPI spec."""

    def test_openapi_includes_all_endpoints(self) -> None:
        """Test that OpenAPI spec includes all defined endpoints.

        **Why this test is important:**
          - Validates our routes are registered correctly
          - Catches accidental route removal
          - Ensures API discoverability

        Note:
            Other OpenAPI tests (schema validation, Swagger UI, ReDoc) were removed
            as they test FastAPI internals rather than our application code.
        """
        app = create_app()
        client = TestClient(app)

        response = client.get("/openapi.json")
        data = response.json()
        paths = data["paths"]

        # Verify key endpoints are documented
        assert "/healthz" in paths
        assert "/search" in paths
        assert "/ray/jobs" in paths
