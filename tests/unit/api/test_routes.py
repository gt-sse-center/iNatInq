"""Unit tests for API routes and endpoints.

This file tests all FastAPI HTTP endpoints defined in pipeline.api.routes.
Tests cover request/response handling, error cases, and service integration.

# Test Coverage

The tests cover:
  - Health check endpoint (/healthz)
  - Image search endpoint (/search/images) with CLIP embeddings
  - Ray job management endpoints (/ray/jobs/images, /ray/jobs/process-dlq,
    /ray/jobs/{job_id}, /ray/jobs/{job_id}/logs, DELETE /ray/jobs/{job_id})
  - Databricks job management endpoints (/databricks/jobs/images,
    /databricks/jobs/cdc-producer, /databricks/jobs/cdc-consumer,
    /databricks/jobs/process-dlq, /databricks/jobs/{run_id},
    /databricks/jobs/{run_id}/logs, DELETE /databricks/jobs/{run_id})
  - Ingestion metrics endpoint (/ingestion/metrics)
  - Cache invalidation endpoint (DELETE /cache)
  - Error handling, validation, and exception propagation
  - Provider configuration and defaults
  - Per-request model override on /search/images

# Test Structure

Tests use pytest class-based organization with TestClient for HTTP requests.
Fixtures from conftest.py provide mocked services and providers.

# Running Tests

Run with: pytest tests/unit/api/test_routes.py
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from core.models import SearchResultItem, SearchResults
from foundation.exceptions import UpstreamError


def _counter(name: str, labels: dict[Any, Any]) -> float:
    """Return current counter value from the global registry (0.0 if unseen)."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _histogram_count(name: str, labels: dict[Any, Any]) -> float:
    """Return the _count sample for a histogram (0.0 if unseen)."""
    return REGISTRY.get_sample_value(f"{name}_count", labels) or 0.0


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
        with (
            patch("api.routes.RayService", return_value=mock_ray_service),
            patch("api.routes.MinIOConfig.from_env") as mock_minio,
        ):
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
            response = test_client.delete("/ray/jobs/raysubmit_1234567890")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"

    def test_submit_ray_image_job_upstream_error_returns_502(
        self,
        test_client: TestClient,
    ) -> None:
        """UpstreamError from RayService propagates through asyncio.to_thread as HTTP 502.

        This verifies the exception re-raise fix: UpstreamError must not be
        swallowed by the broad ``except Exception`` and re-wrapped as
        PipelineError (which would yield 500).
        """
        mock_service = MagicMock()
        mock_service.submit_image_job.side_effect = UpstreamError("Ray cluster unreachable")

        with (
            patch("api.routes.RayService", return_value=mock_service),
            patch("api.routes.MinIOConfig.from_env") as mock_minio,
        ):
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

        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "Bad Gateway"

    def test_process_dlq_entries_ray_success(
        self,
        test_client: TestClient,
        mock_ray_service: MagicMock,
    ) -> None:
        """POST /ray/jobs/process-dlq submits image job with pull_from_dlq=True."""
        with (
            patch("api.routes.RayService", return_value=mock_ray_service),
            patch("api.routes.get_settings") as mock_settings,
            patch("api.routes.MinIOConfig.from_env") as mock_minio,
        ):
            mock_settings.return_value.vector_db.collection = "documents"
            mock_minio.return_value.endpoint_url = "http://minio:9000"
            mock_minio.return_value.access_key_id = "minioadmin"
            mock_minio.return_value.secret_access_key = "minioadmin"
            mock_minio.return_value.bucket = "pipeline"

            response = test_client.post("/ray/jobs/process-dlq")

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "submitted"
        assert data["job_id"] == "raysubmit_1234567890"

        mock_ray_service.submit_image_job.assert_called_once()
        call_kw = mock_ray_service.submit_image_job.call_args.kwargs
        assert call_kw["pull_from_dlq"] == True
        assert call_kw["s3_bucket"] == "pipeline"
        assert call_kw["collection"] == "documents"


# =============================================================================
# Databricks Job Management Endpoints Tests
# =============================================================================


class TestDatabricksJobEndpoints:
    """Test suite for /databricks/jobs/* endpoints."""

    def test_submit_databricks_image_job_s3_success(self, test_client: TestClient) -> None:
        """Test S3 image job submission path uses submit_image_job."""
        with (
            patch("api.routes.DatabricksRayService") as mock_service_cls,
            patch("api.routes.MinIOConfig.from_env") as mock_minio,
            patch("api.routes.EmbeddingConfig.from_env") as mock_embed_cfg,
        ):
            mock_service = MagicMock()
            mock_service.submit_image_job.return_value = 456
            mock_service_cls.return_value = mock_service
            mock_minio.return_value.endpoint_url = "http://minio:9000"
            mock_minio.return_value.access_key_id = "minioadmin"
            mock_minio.return_value.secret_access_key = "minioadmin"
            mock_minio.return_value.bucket = "pipeline"
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
        mock_minio.assert_called_once_with()
        mock_service.submit_image_job.assert_called_once()
        mock_service.submit_inat_image_job.assert_not_called()

    def test_submit_databricks_image_job_inat_success(self, test_client: TestClient) -> None:
        """Test iNat image job submission path bypasses MinIO and uses dedicated submit method."""
        with (
            patch("api.routes.DatabricksRayService") as mock_service_cls,
            patch("api.routes.MinIOConfig.from_env") as mock_minio,
            patch("api.routes.EmbeddingConfig.from_env") as mock_embed_cfg,
        ):
            mock_service = MagicMock()
            mock_service.submit_inat_image_job.return_value = 789
            mock_service_cls.return_value = mock_service
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
            embedding_config=mock_embed_cfg.return_value,
            collection="documents",
        )
        mock_service.submit_image_job.assert_not_called()

    def test_submit_databricks_image_job_s3_defaults_empty_prefix(self, test_client: TestClient) -> None:
        """Test S3 image submission defaults missing s3_prefix to bucket root."""
        with (
            patch("api.routes.DatabricksRayService") as mock_service_cls,
            patch("api.routes.MinIOConfig.from_env") as mock_minio,
            patch("api.routes.EmbeddingConfig.from_env") as mock_embed_cfg,
        ):
            mock_service = MagicMock()
            mock_service.submit_image_job.return_value = 987
            mock_service_cls.return_value = mock_service
            mock_minio.return_value.endpoint_url = "http://minio:9000"
            mock_minio.return_value.access_key_id = "minioadmin"
            mock_minio.return_value.secret_access_key = "minioadmin"
            mock_minio.return_value.bucket = "pipeline"
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

    def test_process_dlq_entries_databricks_success(self, test_client: TestClient) -> None:
        """POST /databricks/jobs/process-dlq submits image job with pull_from_dlq=True."""
        with (
            patch("api.routes.DatabricksRayService") as mock_service_cls,
            patch("api.routes.get_settings") as mock_settings,
            patch("api.routes.MinIOConfig.from_env") as mock_minio,
            patch("api.routes.EmbeddingConfig.from_env") as mock_embed_cfg,
        ):
            mock_service = MagicMock()
            mock_service.submit_image_job.return_value = 456
            mock_service_cls.return_value = mock_service
            mock_settings.return_value.vector_db.collection = "documents"
            mock_minio.return_value.endpoint_url = "http://minio:9000"
            mock_minio.return_value.access_key_id = "minioadmin"
            mock_minio.return_value.secret_access_key = "minioadmin"
            mock_minio.return_value.bucket = "pipeline"
            mock_embed_cfg.return_value = MagicMock()
            response = test_client.post("/databricks/jobs/process-dlq")

        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "submitted"
        assert data["run_id"] == "456"

        mock_service.submit_image_job.assert_called_once()
        call_kw = mock_service.submit_image_job.call_args.kwargs
        assert call_kw["pull_from_dlq"] == True
        assert call_kw["s3_bucket"] == "pipeline"
        assert call_kw["collection"] == "documents"

    def test_submit_databricks_image_job_rejects_s3_autoloader_source(self, test_client: TestClient) -> None:
        """Image job endpoint should reject s3_autoloader source."""
        response = test_client.post(
            "/databricks/jobs/images",
            json={
                "source": "s3_autoloader",
                "collection": "documents",
            },
        )

        assert response.status_code == 422

    @patch("api.routes.DatabricksRayService")
    def test_submit_databricks_cdc_producer_job_success(
        self,
        mock_service_cls: MagicMock,
        test_client: TestClient,
    ) -> None:
        """CDC producer submission should use dedicated Databricks service method."""
        mock_service = MagicMock()
        mock_service.submit_s3_autoloader_job.return_value = 222
        mock_service_cls.return_value = mock_service

        response = test_client.post("/databricks/jobs/cdc-producer")

        assert response.status_code == 202
        data = response.json()
        assert data["run_id"] == "222"
        assert data["status"] == "submitted"
        mock_service.submit_s3_autoloader_job.assert_called_once_with()
        mock_service.submit_image_job.assert_not_called()
        mock_service.submit_inat_image_job.assert_not_called()

    def test_submit_databricks_cdc_producer_job_failure_returns_500(self, test_client: TestClient) -> None:
        """CDC producer submission returns 500 on Databricks service error."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.submit_s3_autoloader_job.side_effect = Exception("boom")
            mock_service_cls.return_value = mock_service

            response = test_client.post("/databricks/jobs/cdc-producer")

        assert response.status_code == 500

    @patch("api.routes.EmbeddingConfig.from_env")
    @patch("api.routes.MinIOConfig.from_env")
    @patch("api.routes.get_settings")
    @patch("api.routes.DatabricksRayService")
    def test_submit_databricks_cdc_consumer_job_success(
        self,
        mock_service_cls: MagicMock,
        mock_settings: MagicMock,
        mock_minio: MagicMock,
        mock_embed_cfg: MagicMock,
        test_client: TestClient,
    ) -> None:
        """CDC consumer submission should use dedicated Databricks service method."""
        mock_service = MagicMock()
        mock_service.submit_s3_bronze_image_job.return_value = 333
        mock_service_cls.return_value = mock_service
        mock_settings.return_value.vector_db.collection = "documents"
        mock_minio.return_value.endpoint_url = "http://minio:9000"
        mock_minio.return_value.access_key_id = "minioadmin"
        mock_minio.return_value.secret_access_key = "minioadmin"
        mock_minio.return_value.bucket = "pipeline"
        mock_embed_cfg.return_value = MagicMock()

        response = test_client.post("/databricks/jobs/cdc-consumer")

        assert response.status_code == 202
        data = response.json()
        assert data["run_id"] == "333"
        assert data["status"] == "submitted"
        mock_service.submit_s3_bronze_image_job.assert_called_once_with(
            s3_endpoint="http://minio:9000",
            s3_access_key_id="minioadmin",
            s3_secret_access_key="minioadmin",
            s3_bucket="pipeline",
            embedding_config=mock_embed_cfg.return_value,
            collection="documents",
        )

    def test_submit_databricks_cdc_consumer_job_failure_returns_500(self, test_client: TestClient) -> None:
        """CDC consumer submission returns 500 on Databricks service error."""
        with patch("api.routes.DatabricksRayService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.submit_s3_bronze_image_job.side_effect = Exception("boom")
            mock_service_cls.return_value = mock_service

            response = test_client.post("/databricks/jobs/cdc-consumer")

        assert response.status_code == 500

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


class TestIngestionMetricsEndpoint:
    """Test suite for POST /ingestion/metrics endpoint."""

    def test_successful_images_increment_counter(self, test_client: TestClient) -> None:
        """Successful count from payload increments status=success counter."""
        before = _counter(
            "inatinq_ingestion_documents_processed_total",
            {"status": "success", "pipeline": "local ray ingestion pipeline"},
        )

        for _ in range(8):
            test_client.post(
                "/ingestion/metrics", json={"pipeline": "local ray ingestion pipeline", "successful": 1}
            )
        test_client.post(
            "/ingestion/metrics", json={"pipeline": "local ray ingestion pipeline", "successful": 2}
        )

        after = _counter(
            "inatinq_ingestion_documents_processed_total",
            {"status": "success", "pipeline": "local ray ingestion pipeline"},
        )
        assert after - before == 10

    def test_failed_images_increment_counter(self, test_client: TestClient) -> None:
        """Failed count from payload increments status=failed counter."""
        before = _counter(
            "inatinq_ingestion_documents_processed_total",
            {"status": "failed", "pipeline": "local ray ingestion pipeline"},
        )
        for _ in range(3):
            test_client.post(
                "/ingestion/metrics", json={"pipeline": "local ray ingestion pipeline", "failed": 1}
            )

        after = _counter(
            "inatinq_ingestion_documents_processed_total",
            {"status": "failed", "pipeline": "local ray ingestion pipeline"},
        )
        assert after - before == 3

    def test_batch_duration_observed_in_histogram(self, test_client: TestClient) -> None:
        """batch_duration_seconds is recorded as a histogram observation."""
        before = _histogram_count(
            "inatinq_ingestion_batch_duration_seconds",
            {"pipeline": "local ray ingestion pipeline"},
        )

        test_client.post(
            "/ingestion/metrics",
            json={"pipeline": "local ray ingestion pipeline", "successful": 5, "batch_duration_seconds": 8.5},
        )

        after = _histogram_count(
            "inatinq_ingestion_batch_duration_seconds",
            {"pipeline": "local ray ingestion pipeline"},
        )
        assert after - before == 1

    def test_checkpoint_save_increments_counter(self, test_client: TestClient) -> None:
        """checkpoint_save=True increments the checkpoint saves counter."""
        before = _counter(
            "inatinq_ingestion_checkpoint_saves_total",
            {"pipeline": "databricks ingestion pipeline"},
        )

        test_client.post(
            "/ingestion/metrics",
            json={"pipeline": "databricks ingestion pipeline", "checkpoint_save": True},
        )

        after = _counter(
            "inatinq_ingestion_checkpoint_saves_total",
            {"pipeline": "databricks ingestion pipeline"},
        )
        assert after - before == 1

    def test_endpoint_returns_ok(self, test_client: TestClient) -> None:
        """Endpoint returns 200 with {"status": "ok"}."""
        response = test_client.post(
            "/ingestion/metrics",
            json={"pipeline": "local ray ingestion pipeline", "successful": 1},
        )

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_missing_pipeline_returns_422(self, test_client: TestClient) -> None:
        """Missing pipeline field returns HTTP 422."""
        response = test_client.post("/ingestion/metrics", json={"successful": 5})

        assert response.status_code == 422


# =============================================================================
# Auto-Detect Image Provider Tests
# =============================================================================


class TestAutoDetectImageProvider:
    """Test suite for auto_detect_image_provider helper."""

    def test_auto_detect_siglip_returns_infinity(self) -> None:
        """SigLIP model names should resolve to INFINITY provider."""
        from api.routes import auto_detect_image_provider
        from config import ProviderType

        assert auto_detect_image_provider("google/siglip-so400m-patch14-384") == ProviderType.INFINITY

    def test_auto_detect_siglip2_returns_infinity(self) -> None:
        """SigLIP2 model names should also resolve to INFINITY provider."""
        from api.routes import auto_detect_image_provider
        from config import ProviderType

        assert auto_detect_image_provider("google/siglip2-base-patch16-224") == ProviderType.INFINITY

    def test_auto_detect_llava_returns_clip(self) -> None:
        """LLaVA model names should resolve to LOCAL_CLIP provider."""
        from api.routes import auto_detect_image_provider
        from config import ProviderType

        assert auto_detect_image_provider("llava") == ProviderType.LOCAL_CLIP

    def test_auto_detect_unknown_returns_clip(self) -> None:
        """Unknown model names should fall back to LOCAL_CLIP provider."""
        from api.routes import auto_detect_image_provider
        from config import ProviderType

        assert auto_detect_image_provider("some-unknown-model") == ProviderType.LOCAL_CLIP


# =============================================================================
# Image Search Model Override Tests
# =============================================================================


class TestImageSearchModelOverride:
    """Test suite for per-request model override on /search/images."""

    def test_search_with_model_override(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Passing ?model=google/siglip-so400m-patch14-384 should use Infinity provider."""
        from config import ProviderType

        with (
            patch("api.routes.EmbeddingConfig.from_env") as mock_from_env,
            patch("api.routes.create_embedding_provider") as mock_create_embed,
            patch("api.routes.create_vector_db_provider", return_value=mock_image_vector_db_provider),
        ):
            from config import EmbeddingConfig

            base_config = EmbeddingConfig(provider_type=ProviderType.LOCAL_CLIP, clip_model="ViT-B/32")
            mock_from_env.return_value = base_config

            mock_provider = MagicMock()
            mock_provider.embed_text = AsyncMock(return_value=[0.1] * 768)
            mock_provider.vector_size = 768
            mock_provider.model_name = "google/siglip-so400m-patch14-384"
            mock_create_embed.return_value = mock_provider

            response = test_client.get("/search/images?q=test&model=google/siglip-so400m-patch14-384")

        assert response.status_code == 200
        # Verify create_embedding_provider was called with Infinity provider type
        called_config = mock_create_embed.call_args[0][0]
        assert called_config.provider_type == ProviderType.INFINITY
        assert called_config.infinity_model == "google/siglip-so400m-patch14-384"

    def test_search_with_model_and_provider(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Explicit image_provider should override auto-detection."""
        from config import ProviderType

        with (
            patch("api.routes.EmbeddingConfig.from_env") as mock_from_env,
            patch("api.routes.create_embedding_provider") as mock_create_embed,
            patch("api.routes.create_vector_db_provider", return_value=mock_image_vector_db_provider),
        ):
            from config import EmbeddingConfig

            base_config = EmbeddingConfig(provider_type=ProviderType.LOCAL_CLIP, clip_model="ViT-B/32")
            mock_from_env.return_value = base_config

            mock_provider = MagicMock()
            mock_provider.embed_text = AsyncMock(return_value=[0.1] * 768)
            mock_provider.vector_size = 768
            mock_provider.model_name = "custom-model"
            mock_create_embed.return_value = mock_provider

            response = test_client.get("/search/images?q=test&model=custom-model&image_provider=clip")

        assert response.status_code == 200
        called_config = mock_create_embed.call_args[0][0]
        assert called_config.provider_type == ProviderType.LOCAL_CLIP
        assert called_config.clip_model == "custom-model"

    def test_search_without_model_uses_default(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        patch_embedding_config: MagicMock,
        patch_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Without ?model param, embedding config should be used as-is (no override)."""
        with patch(
            "api.routes.create_vector_db_provider",
            return_value=mock_image_vector_db_provider,
        ):
            response = test_client.get("/search/images?q=test")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "ViT-B/32"  # Default from mock fixture

    def test_response_reflects_override_model(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Response 'model' field should reflect the overridden model name."""
        from config import ProviderType

        with (
            patch("api.routes.EmbeddingConfig.from_env") as mock_from_env,
            patch("api.routes.create_embedding_provider") as mock_create_embed,
            patch("api.routes.create_vector_db_provider", return_value=mock_image_vector_db_provider),
        ):
            from config import EmbeddingConfig

            base_config = EmbeddingConfig(provider_type=ProviderType.LOCAL_CLIP, clip_model="ViT-B/32")
            mock_from_env.return_value = base_config

            mock_provider = MagicMock()
            mock_provider.embed_text = AsyncMock(return_value=[0.1] * 768)
            mock_provider.vector_size = 768
            mock_provider.model_name = "google/siglip-so400m-patch14-384"
            mock_create_embed.return_value = mock_provider

            response = test_client.get("/search/images?q=test&model=google/siglip-so400m-patch14-384")

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "google/siglip-so400m-patch14-384"

    def test_vector_size_mismatch_returns_400(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
    ) -> None:
        """When model vector size doesn't match collection, should return 400."""
        from config import ProviderType

        from core.models import CollectionInfo

        mock_vector_db = MagicMock()
        mock_vector_db.search_async = AsyncMock()
        # Collection expects 512-d vectors
        mock_vector_db.get_collection_info_async = AsyncMock(
            return_value=CollectionInfo(name="documents", vector_size=512)
        )
        mock_vector_db.close = MagicMock()

        with (
            patch("api.routes.EmbeddingConfig.from_env") as mock_from_env,
            patch("api.routes.create_embedding_provider") as mock_create_embed,
            patch("api.routes.create_vector_db_provider", return_value=mock_vector_db),
        ):
            from config import EmbeddingConfig

            base_config = EmbeddingConfig(provider_type=ProviderType.LOCAL_CLIP, clip_model="ViT-B/32")
            mock_from_env.return_value = base_config

            mock_provider = MagicMock()
            mock_provider.embed_text = AsyncMock(return_value=[0.1] * 768)
            # Model produces 768-d but collection has 512-d
            mock_provider.vector_size = 768
            mock_provider.model_name = "google/siglip-so400m-patch14-384"
            mock_create_embed.return_value = mock_provider

            response = test_client.get("/search/images?q=test&model=google/siglip-so400m-patch14-384")

        assert response.status_code == 400
        data = response.json()
        assert "mismatch" in data["message"].lower()

    def test_model_override_missing_provider_config_returns_400(
        self,
        test_client: TestClient,
        patch_get_settings: MagicMock,
    ) -> None:
        """Override to a provider whose URL is not configured should return 400."""
        from config import ProviderType

        with (
            patch("api.routes.EmbeddingConfig.from_env") as mock_from_env,
            patch(
                "api.routes.create_embedding_provider",
                side_effect=ValueError("clip_url is required in EmbeddingConfig"),
            ),
        ):
            from config import EmbeddingConfig

            # Base config is Infinity — clip_url is not set
            base_config = EmbeddingConfig(
                provider_type=ProviderType.INFINITY,
                infinity_url="http://localhost:7997",
                infinity_model="google/siglip-so400m-patch14-384",
            )
            mock_from_env.return_value = base_config

            response = test_client.get("/search/images?q=test&model=ViT-B/32&image_provider=clip")

        assert response.status_code == 400
        data = response.json()
        assert "clip_url" in data["message"]
