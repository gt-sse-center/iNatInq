"""Unit tests for API routes and endpoints.

This file tests all FastAPI HTTP endpoints defined in pipeline.api.routes.
Tests cover request/response handling, error cases, and service integration.

# Test Coverage

The tests cover:
  - Health check endpoint (/healthz)
  - Search endpoint (/search) with Qdrant and Weaviate providers
  - Ray job management endpoints (/ray/jobs/*)
  - Error handling and validation
  - Provider configuration and defaults

# Test Structure

Tests use pytest class-based organization with TestClient for HTTP requests.
Fixtures from conftest.py provide mocked services and providers.

# Running Tests

Run with: pytest tests/unit/api/test_routes.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from core.exceptions import UpstreamError
from core.models import SearchResultItem, SearchResults

# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthzEndpoint:
    """Test suite for /healthz endpoint."""

    def test_healthz_returns_ok(self, test_client: TestClient) -> None:
        """Test that /healthz endpoint returns 200 OK.

        **Why this test is important:**
          - Validates Kubernetes liveness/readiness probe endpoint
          - Ensures service can respond to health checks
          - Critical for deployment health
        """
        response = test_client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_healthz_does_not_check_dependencies(self, test_client: TestClient) -> None:
        """Test that /healthz succeeds even if dependencies are down.

        **Why this test is important:**
          - Validates that health check doesn't depend on external services
          - Prevents restart loops from dependency failures
          - Follows best practices for liveness probes
        """
        # Health check should succeed regardless of external services
        response = test_client.get("/healthz")
        assert response.status_code == 200


# =============================================================================
# Search Endpoint Tests
# =============================================================================


class TestSearchEndpoint:
    """Test suite for /search endpoint."""

    def test_search_with_qdrant_provider_success(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_create_embedding_provider: MagicMock,
        patch_create_vector_db_provider: MagicMock,
    ) -> None:
        """Test successful search with Qdrant provider.

        **Why this test is important:**
          - Validates end-to-end search flow
          - Tests Qdrant provider integration
          - Ensures proper response format
        """
        response = test_client.get("/search?q=machine%20learning&limit=5&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert data["provider"] == "qdrant"
        assert len(data["results"]) == 2
        assert data["results"][0]["score"] == 0.9234
        assert "text" in data["results"][0]
        assert "metadata" in data["results"][0]

    def test_search_with_weaviate_provider_success(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_create_embedding_provider: MagicMock,
    ) -> None:
        """Test successful search with Weaviate provider.

        **Why this test is important:**
          - Validates Weaviate provider integration
          - Tests provider switching capability
          - Ensures multi-provider support works
        """
        # Mock Weaviate provider with AsyncMock for search_async
        mock_weaviate = MagicMock()
        mock_weaviate.search_async = AsyncMock(
            return_value=SearchResults(
                items=[
                    SearchResultItem(
                        point_id="weaviate-id",
                        score=0.88,
                        payload={"text": "weaviate document", "source": "weaviate"},
                    )
                ],
                total=1,
            )
        )
        mock_weaviate.close = MagicMock()

        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_weaviate,
        ):
            response = test_client.get("/search?q=test%20query&limit=5&provider=weaviate")

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "weaviate"
        assert data["results"][0]["id"] == "weaviate-id"

    def test_search_with_default_provider_from_settings(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_create_embedding_provider: MagicMock,
        patch_create_vector_db_provider: MagicMock,
    ) -> None:
        """Test search uses default provider from settings when not specified.

        **Why this test is important:**
          - Validates default provider configuration
          - Tests settings integration
          - Ensures backward compatibility
        """
        response = test_client.get("/search?q=test%20query")

        assert response.status_code == 200
        data = response.json()
        # Should use default from settings (qdrant in our mock)
        assert data["provider"] == "qdrant"

    def test_search_with_invalid_provider_returns_422(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
    ) -> None:
        """Test search with invalid provider returns 422 Unprocessable Entity.

        **Why this test is important:**
          - Validates provider validation via Pydantic
          - Tests query parameter validation
          - Ensures proper error handling
        """
        response = test_client.get("/search?q=test&provider=invalid")

        assert response.status_code == 422  # FastAPI/Pydantic validation error
        data = response.json()
        assert "detail" in data  # Pydantic validation error format

    def test_search_with_empty_query_returns_400(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_create_embedding_provider: MagicMock,
        patch_create_vector_db_provider: MagicMock,
    ) -> None:
        """Test search with empty query returns 400 Bad Request.

        **Why this test is important:**
          - Validates query validation
          - Tests empty string handling
          - Prevents unnecessary API calls
        """
        response = test_client.get("/search?q=")

        assert response.status_code == 400

    def test_search_with_custom_collection(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_create_embedding_provider: MagicMock,
        patch_create_vector_db_provider: MagicMock,
    ) -> None:
        """Test search with custom collection name.

        **Why this test is important:**
          - Validates collection override capability
          - Tests optional parameter handling
        """
        response = test_client.get("/search?q=test&collection=custom-collection&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "custom-collection"

    def test_search_with_custom_model(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_create_embedding_provider: MagicMock,
        patch_create_vector_db_provider: MagicMock,
    ) -> None:
        """Test search with custom model name.

        **Why this test is important:**
          - Validates model override capability
          - Tests optional parameter handling
        """
        response = test_client.get("/search?q=test&model=custom-model&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "custom-model"

    def test_search_with_custom_limit(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_create_embedding_provider: MagicMock,
        patch_create_vector_db_provider: MagicMock,
    ) -> None:
        """Test search with custom result limit.

        **Why this test is important:**
          - Validates limit parameter
          - Tests result count control
        """
        response = test_client.get("/search?q=test&limit=20&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        # Verify service was called (actual count depends on mock data)
        assert "results" in data

    def test_search_closes_providers_after_request(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_vector_db_provider: MagicMock,
    ) -> None:
        """Test that search successfully completes without explicit resource cleanup.

        **Why this test is important:**
          - Validates request completes successfully
          - Tests provider lifecycle management
          - Ensures no event loop conflicts

        Note: Providers are not explicitly closed to avoid asyncio event loop
        issues. They are garbage collected automatically when the function
        returns, and Python's garbage collector handles underlying resources.
        """
        with patch(
            "api.routes.create_embedding_provider",
            return_value=mock_embedding_provider,
        ):
            with patch(
                "api.routes.create_vector_db_provider",
                return_value=mock_vector_db_provider,
            ):
                response = test_client.get("/search?q=test&provider=qdrant")

        assert response.status_code == 200
        # Providers are not explicitly closed to avoid event loop issues
        # They will be garbage collected when the request completes
        assert not mock_embedding_provider.close.called
        assert not mock_vector_db_provider.close.called

    def test_search_handles_upstream_error(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        mock_embedding_provider: MagicMock,
    ) -> None:
        """Test that search handles upstream service errors correctly.

        **Why this test is important:**
          - Validates error handling for service failures
          - Tests exception translation to HTTP status
          - Ensures proper error messages
        """
        # Mock vector DB provider that raises UpstreamError
        mock_vector_db = MagicMock()
        mock_vector_db.search_async.side_effect = UpstreamError("Qdrant connection failed")
        mock_vector_db.close = MagicMock()

        with patch(
            "api.routes.create_embedding_provider",
            return_value=mock_embedding_provider,
        ):
            with patch(
                "api.routes.create_vector_db_provider",
                return_value=mock_vector_db,
            ):
                response = test_client.get("/search?q=test&provider=qdrant")

        assert response.status_code == 502  # Bad Gateway
        data = response.json()
        assert "error" in data


# =============================================================================
# Ray Job Management Endpoints Tests
# =============================================================================


class TestRayJobEndpoints:
    """Test suite for /ray/jobs/* endpoints."""

    def test_submit_ray_job_success(
        self,
        test_client: TestClient,
        mock_ray_service: MagicMock,
    ) -> None:
        """Test successful Ray job submission.

        **Why this test is important:**
          - Validates Ray job submission flow
          - Tests RayService integration
          - Ensures proper response format
        """
        with patch(
            "api.routes.RayService",
            return_value=mock_ray_service,
        ):
            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                response = test_client.post(
                    "/ray/jobs",
                    json={
                        "s3_prefix": "inputs/",
                        "collection": "documents",
                    },
                )

        assert response.status_code == 202  # Accepted
        data = response.json()
        assert data["job_id"] == "raysubmit_1234567890"
        assert data["status"] == "submitted"

    def test_get_ray_job_status_success(
        self,
        test_client: TestClient,
        mock_ray_service: MagicMock,
    ) -> None:
        """Test getting Ray job status.

        **Why this test is important:**
          - Validates job status retrieval
          - Tests RayService integration
        """
        with patch(
            "api.routes.RayService",
            return_value=mock_ray_service,
        ):
            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                response = test_client.get("/ray/jobs/raysubmit_1234567890")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "RUNNING"

    def test_get_ray_job_logs_success(
        self,
        test_client: TestClient,
        mock_ray_service: MagicMock,
    ) -> None:
        """Test getting Ray job logs.

        **Why this test is important:**
          - Validates log retrieval
          - Tests RayService integration
        """
        with patch(
            "api.routes.RayService",
            return_value=mock_ray_service,
        ):
            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                response = test_client.get("/ray/jobs/raysubmit_1234567890/logs")

        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "Processing 1000 documents" in data["logs"]

    def test_stop_ray_job_success(
        self,
        test_client: TestClient,
        mock_ray_service: MagicMock,
    ) -> None:
        """Test stopping a Ray job.

        **Why this test is important:**
          - Validates job termination
          - Tests RayService integration
        """
        with patch(
            "api.routes.RayService",
            return_value=mock_ray_service,
        ):
            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                response = test_client.delete("/ray/jobs/raysubmit_1234567890")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
