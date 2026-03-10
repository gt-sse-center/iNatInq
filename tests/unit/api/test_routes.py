"""Unit tests for API routes and endpoints.

This file tests all FastAPI HTTP endpoints defined in pipeline.api.routes.
Tests cover request/response handling, error cases, and service integration.

# Test Coverage

The tests cover:
  - Health check endpoint (/healthz)
  - Search endpoint (/search) with Qdrant and Weaviate providers
  - Image search endpoint (/search/images) with CLIP embeddings
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
# Image Search Endpoint Tests
# =============================================================================


class TestImageSearchEndpoint:
    """Test suite for /search/images endpoint."""

    def test_image_search_with_qdrant_provider_success(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test successful image search with Qdrant provider.

        **Why this test is important:**
          - Validates end-to-end image search flow
          - Tests CLIP + Qdrant provider integration
          - Ensures proper response format for image results
        """
        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_image_vector_db_provider,
        ):
            response = test_client.get("/search/images?q=sunset%20over%20ocean&limit=5&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "sunset over ocean"
        assert data["provider"] == "qdrant"
        assert data["collection"] == "documents"
        assert len(data["results"]) == 2
        assert data["results"][0]["score"] == 0.8234
        assert "s3_key" in data["results"][0]
        assert "s3_uri" in data["results"][0]
        assert data["results"][0]["format"] == "jpeg"
        assert data["results"][0]["width"] == 1920
        assert data["results"][0]["height"] == 1080

    def test_image_search_with_weaviate_provider_success(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
    ) -> None:
        """Test successful image search with Weaviate provider.

        **Why this test is important:**
          - Validates Weaviate provider integration for image search
          - Tests provider switching capability
          - Ensures multi-provider support works
        """
        # Mock Weaviate provider with AsyncMock for search_async
        mock_weaviate = MagicMock()
        mock_weaviate.search_async = AsyncMock(
            return_value=SearchResults(
                items=[
                    SearchResultItem(
                        point_id="weaviate-img-id",
                        score=0.88,
                        payload={
                            "s3_key": "images/cat.jpg",
                            "s3_uri": "s3://bucket/images/cat.jpg",
                            "format": "jpeg",
                        },
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
            response = test_client.get("/search/images?q=fluffy%20cat&limit=5&provider=weaviate")

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "weaviate"
        assert data["results"][0]["id"] == "weaviate-img-id"
        assert data["results"][0]["s3_key"] == "images/cat.jpg"

    def test_image_search_with_default_provider_from_settings(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test image search uses default provider from settings when not specified.

        **Why this test is important:**
          - Validates default provider configuration
          - Tests settings integration
          - Ensures backward compatibility
        """
        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_image_vector_db_provider,
        ):
            response = test_client.get("/search/images?q=test%20query")

        assert response.status_code == 200
        data = response.json()
        # Should use default from settings (qdrant in our mock)
        assert data["provider"] == "qdrant"

    def test_image_search_with_invalid_provider_returns_422(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
    ) -> None:
        """Test image search with invalid provider returns 422 Unprocessable Entity.

        **Why this test is important:**
          - Validates provider validation via Pydantic
          - Tests query parameter validation
          - Ensures proper error handling
        """
        response = test_client.get("/search/images?q=test&provider=pinecone")

        assert response.status_code == 422  # FastAPI/Pydantic validation error
        data = response.json()
        assert "detail" in data  # Pydantic validation error format

    def test_image_search_with_empty_query_returns_400(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test image search with empty query returns 400 Bad Request.

        **Why this test is important:**
          - Validates query validation
          - Tests empty string handling
          - Prevents unnecessary API calls
        """
        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_image_vector_db_provider,
        ):
            response = test_client.get("/search/images?q=")

        assert response.status_code == 400

    def test_image_search_with_custom_collection(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test image search with custom collection name.

        **Why this test is important:**
          - Validates collection override capability
        """
        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_image_vector_db_provider,
        ):
            response = test_client.get("/search/images?q=test&collection=photos&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["collection"] == "photos"

    def test_image_search_returns_clip_model_name(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test image search returns CLIP model name in response.

        **Why this test is important:**
          - Validates model name is included in response
          - Tests EmbeddingConfig integration
        """
        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_image_vector_db_provider,
        ):
            response = test_client.get("/search/images?q=test&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "ViT-B/32"  # From mock_embedding_config

    def test_image_search_handles_upstream_error(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
    ) -> None:
        """Test that image search handles upstream service errors correctly.

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
            "api.routes.create_vector_db_provider",
            return_value=mock_vector_db,
        ):
            response = test_client.get("/search/images?q=test&provider=qdrant")

        assert response.status_code == 502  # Bad Gateway
        data = response.json()
        assert "error" in data

    def test_image_search_handles_clip_error(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test that image search handles embedding provider errors correctly.

        **Why this test is important:**
          - Validates error handling for embedding provider failures
          - Tests exception translation to HTTP status
          - Ensures proper error messages
        """
        # Mock embedding provider that raises UpstreamError
        mock_provider = MagicMock()
        mock_provider.embed_text = AsyncMock(side_effect=UpstreamError("Embedding connection failed"))

        with patch(
            "api.routes.create_embedding_provider",
            return_value=mock_provider,
        ):
            with patch(
                "api.routes.create_vector_db_provider",
                return_value=mock_image_vector_db_provider,
            ):
                response = test_client.get("/search/images?q=test&provider=qdrant")

        assert response.status_code == 502  # Bad Gateway
        data = response.json()
        assert "error" in data

    def test_image_search_optional_fields_can_be_null(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
    ) -> None:
        """Test that image search handles optional fields being null.

        **Why this test is important:**
          - Validates that optional fields like thumbnail_key can be null
          - Tests response model flexibility
        """
        # Mock provider returning results with null optional fields
        mock_vector_db = MagicMock()
        mock_vector_db.search_async = AsyncMock(
            return_value=SearchResults(
                items=[
                    SearchResultItem(
                        point_id="img-1",
                        score=0.9,
                        payload={
                            "s3_key": "images/test.jpg",
                            "s3_uri": "s3://bucket/images/test.jpg",
                            # format, width, height, thumbnail_key are all missing
                        },
                    )
                ],
                total=1,
            )
        )
        mock_vector_db.close = MagicMock()

        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_vector_db,
        ):
            response = test_client.get("/search/images?q=test&provider=qdrant")

        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["format"] is None
        assert data["results"][0]["width"] is None
        assert data["results"][0]["height"] is None
        assert data["results"][0]["thumbnail_key"] is None


# =============================================================================
# Ray Job Management Endpoints Tests
# =============================================================================


class TestRayJobEndpoints:
    """Test suite for /ray/jobs/* endpoints."""

    def test_submit_ray_image_job_success(
        self,
        test_client: TestClient,
        mock_ray_service: MagicMock,
    ) -> None:
        """Test successful Ray image job submission.

        **Why this test is important:**
          - Validates image job submission flow
          - Tests RayService.submit_image_job integration
          - Ensures proper response format (job_id, s3_bucket, s3_prefix, collection)
        """
        with patch(
            "api.routes.RayService",
            return_value=mock_ray_service,
        ):
            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                with patch("api.routes.MinIOConfig.from_env") as mock_minio:
                    mock_minio.return_value.endpoint_url = "http://minio:9000"
                    mock_minio.return_value.access_key_id = "minioadmin"
                    mock_minio.return_value.secret_access_key = "minioadmin"
                    response = test_client.post(
                        "/ray/jobs/images",
                        json={
                            "s3_bucket": "pipeline",
                            "s3_prefix": "images/",
                            "collection": "documents",
                        },
                    )

        assert response.status_code == 202
        data = response.json()
        assert data["job_id"] == "raysubmit_1234567890"
        assert data["status"] == "submitted"
        assert data["s3_bucket"] == "pipeline"
        assert data["s3_prefix"] == "images/"
        assert data["collection"] == "documents"
        mock_ray_service.submit_image_job.assert_called_once()
        call_kw = mock_ray_service.submit_image_job.call_args.kwargs
        assert call_kw["s3_bucket"] == "pipeline"
        assert call_kw["s3_prefix"] == "images/"
        assert call_kw["collection"] == "documents"

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


# =============================================================================
# Databricks Job Management Endpoints Tests
# =============================================================================


class TestDatabricksJobEndpoints:
    """Test suite for /databricks/jobs/* endpoints."""

    def test_submit_databricks_image_job_s3_success(self, test_client: TestClient) -> None:
        """Test S3 image job submission path uses submit_image_job."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.submit_image_job.return_value = 456
            mock_service_cls.return_value = mock_service

            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                with patch("api.routes.MinIOConfig.from_env") as mock_minio:
                    mock_minio.return_value.endpoint_url = "http://minio:9000"
                    mock_minio.return_value.access_key_id = "minioadmin"
                    mock_minio.return_value.secret_access_key = "minioadmin"
                    mock_minio.return_value.bucket = "pipeline"
                    with patch("api.routes.EmbeddingConfig.from_env") as mock_embed_cfg:
                        mock_embed_cfg.return_value = MagicMock()
                        response = test_client.post(
                            "/databricks/jobs/images",
                            json={
                                "source": "s3",
                                "s3_prefix": "images/",
                                "collection": "documents",
                            },
                        )

        assert response.status_code == 202
        data = response.json()
        assert data["run_id"] == "456"
        assert data["source"] == "s3"
        mock_minio.assert_called_once_with("ml-system")
        mock_service.submit_image_job.assert_called_once()
        mock_service.submit_inat_image_job.assert_not_called()

    def test_submit_databricks_image_job_inat_success(self, test_client: TestClient) -> None:
        """Test iNat image job submission path bypasses MinIO and uses dedicated submit method."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.submit_inat_image_job.return_value = 789
            mock_service_cls.return_value = mock_service

            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                with patch("api.routes.MinIOConfig.from_env") as mock_minio:
                    with patch("api.routes.EmbeddingConfig.from_env") as mock_embed_cfg:
                        mock_embed_cfg.return_value = MagicMock()
                        response = test_client.post(
                            "/databricks/jobs/images",
                            json={
                                "source": "inat",
                                "collection": "documents",
                            },
                        )

        assert response.status_code == 202
        data = response.json()
        assert data["run_id"] == "789"
        assert data["source"] == "inat"
        assert data["s3_prefix"] is None
        mock_minio.assert_not_called()
        mock_service.submit_inat_image_job.assert_called_once_with(
            namespace="ml-system",
            embedding_config=mock_embed_cfg.return_value,
            collection="documents",
        )
        mock_service.submit_image_job.assert_not_called()

    def test_submit_databricks_image_job_s3_defaults_empty_prefix(self, test_client: TestClient) -> None:
        """Test S3 image submission defaults missing s3_prefix to bucket root."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.submit_image_job.return_value = 987
            mock_service_cls.return_value = mock_service

            with patch("api.routes.get_settings") as mock_settings:
                mock_settings.return_value.k8s_namespace = "ml-system"
                with patch("api.routes.MinIOConfig.from_env") as mock_minio:
                    mock_minio.return_value.endpoint_url = "http://minio:9000"
                    mock_minio.return_value.access_key_id = "minioadmin"
                    mock_minio.return_value.secret_access_key = "minioadmin"
                    mock_minio.return_value.bucket = "pipeline"
                    with patch("api.routes.EmbeddingConfig.from_env") as mock_embed_cfg:
                        mock_embed_cfg.return_value = MagicMock()
                        response = test_client.post(
                            "/databricks/jobs/images",
                            json={
                                "source": "s3",
                                "collection": "documents",
                            },
                        )

        assert response.status_code == 202
        data = response.json()
        assert data["run_id"] == "987"
        assert data["source"] == "s3"
        assert data["s3_prefix"] == ""
        mock_service.submit_image_job.assert_called_once()
        call_kw = mock_service.submit_image_job.call_args.kwargs
        assert call_kw["s3_prefix"] == ""
        mock_service.submit_inat_image_job.assert_not_called()

    def test_get_databricks_job_status_success(self, test_client: TestClient) -> None:
        """Test getting Databricks job status."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.get_run_status.return_value = {
                "life_cycle_state": "RUNNING",
                "result_state": None,
                "state_message": "Job is running",
            }
            mock_service_cls.return_value = mock_service

            response = test_client.get("/databricks/jobs/123")

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "123"
        assert data["life_cycle_state"] == "RUNNING"
        assert data["state_message"] == "Job is running"

    def test_get_databricks_job_logs_success(self, test_client: TestClient) -> None:
        """Test getting Databricks job logs."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.get_run_output.return_value = "log output"
            mock_service_cls.return_value = mock_service

            response = test_client.get("/databricks/jobs/456/logs")

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "456"
        assert data["logs"] == "log output"

    def test_stop_databricks_job_success(self, test_client: TestClient) -> None:
        """Test stopping a Databricks job run."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service_cls.return_value = mock_service

            response = test_client.delete("/databricks/jobs/789")

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "789"
        assert data["status"] == "stopped"

    def test_submit_databricks_job_failure_returns_500(self, test_client: TestClient) -> None:
        """Test that submit returns 500 on Databricks service error."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.submit_image_job.side_effect = Exception("boom")
            mock_service_cls.return_value = mock_service

            response = test_client.post(
                "/databricks/jobs/images",
                json={
                    "s3_prefix": "inputs/",
                    "collection": "documents",
                },
            )

        assert response.status_code == 500

    def test_get_databricks_job_status_failure_returns_500(self, test_client: TestClient) -> None:
        """Test that status returns 500 on Databricks service error."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.get_run_status.side_effect = Exception("boom")
            mock_service_cls.return_value = mock_service

            response = test_client.get("/databricks/jobs/123")

        assert response.status_code == 500

    def test_get_databricks_job_logs_failure_returns_500(self, test_client: TestClient) -> None:
        """Test that logs returns 500 on Databricks service error."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.get_run_output.side_effect = Exception("boom")
            mock_service_cls.return_value = mock_service

            response = test_client.get("/databricks/jobs/456/logs")

        assert response.status_code == 500

    def test_stop_databricks_job_failure_returns_500(self, test_client: TestClient) -> None:
        """Test that stop returns 500 on Databricks service error."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.stop_run.side_effect = Exception("boom")
            mock_service_cls.return_value = mock_service

            response = test_client.delete("/databricks/jobs/789")

        assert response.status_code == 500
