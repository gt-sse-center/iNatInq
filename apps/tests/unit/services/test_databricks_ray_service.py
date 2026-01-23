"""Unit tests for core.services.databricks_ray_service module.

This file tests the DatabricksRayService class which submits Ray ingestion jobs
via the Databricks Jobs API.

# Test Coverage

The tests cover:
  - Job Submission: Parameter construction and API invocation
  - Optional Config: Embedding and vector DB env propagation
  - Error Handling: UpstreamError wrapping and config errors

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.

# Running Tests

Run with: pytest tests/unit/services/test_databricks_ray_service.py
"""

from unittest.mock import MagicMock, patch

import pytest

from config import EmbeddingConfig, VectorDBConfig
from core.exceptions import UpstreamError
from core.services.databricks_ray_service import DatabricksRayService


# =============================================================================
# Job Submission Tests
# =============================================================================


class TestDatabricksRayServiceSubmitJob:
    """Test suite for DatabricksRayService.submit_s3_to_qdrant."""

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that submit_s3_to_qdrant submits a job successfully."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.job_id = 123
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.run_id = 456
        mock_client.jobs.run_now.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        run_id = service.submit_s3_to_qdrant(
            namespace="test-namespace",
            s3_endpoint="http://minio.test:9000",
            s3_access_key_id="test-key",
            s3_secret_access_key="test-secret",
            s3_bucket="test-bucket",
            s3_prefix="inputs/",
            embedding_config=embedding_config,
            vector_db_config=vector_db_config,
            collection="test-collection",
        )

        assert run_id == 456
        mock_client_cls.assert_called_once_with(host="https://dbc.example.cloud", token="databricks-token")

        call_kwargs = mock_client.jobs.run_now.call_args.kwargs
        assert call_kwargs["job_id"] == 123

        params = call_kwargs["python_params"]
        assert "K8S_NAMESPACE=test-namespace" in params
        assert "S3_PREFIX=inputs/" in params
        assert "S3_ENDPOINT=http://minio.test:9000" in params
        assert "S3_ACCESS_KEY_ID=test-key" in params
        assert "S3_SECRET_ACCESS_KEY=test-secret" in params
        assert "S3_BUCKET=test-bucket" in params
        assert "VECTOR_DB_COLLECTION=test-collection" in params
        assert "EMBEDDING_PROVIDER_TYPE=ollama" in params

    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "http://qdrant.test:6333",
            "QDRANT_API_KEY": "qdrant-key",
            "WEAVIATE_URL": "http://weaviate.test:8080",
            "WEAVIATE_API_KEY": "weaviate-key",
            "WEAVIATE_GRPC_HOST": "grpc.weaviate.test",
        },
        clear=False,
    )
    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_includes_optional_config(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that submit includes optional embedding and vector DB env vars."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.job_id = 123
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.run_id = 456
        mock_client.jobs.run_now.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedding_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama.test:11434",
            ollama_model="nomic-embed-text",
            vector_size=768,
        )
        vector_db_config = VectorDBConfig(
            provider_type="qdrant",
            qdrant_url="http://qdrant.test:6333",
            collection="test_documents",
        )

        service = DatabricksRayService()
        service.submit_s3_to_qdrant(
            namespace="test-namespace",
            s3_endpoint="http://minio.test:9000",
            s3_access_key_id="test-key",
            s3_secret_access_key="test-secret",
            s3_bucket="test-bucket",
            s3_prefix="inputs/",
            embedding_config=embedding_config,
            vector_db_config=vector_db_config,
            collection="test-collection",
        )

        params = mock_client.jobs.run_now.call_args.kwargs["python_params"]
        assert "EMBEDDING_VECTOR_SIZE=768" in params
        assert "OLLAMA_BASE_URL=http://ollama.test:11434" in params
        assert "OLLAMA_MODEL=nomic-embed-text" in params
        assert "QDRANT_URL=http://qdrant.test:6333" in params
        assert "QDRANT_API_KEY=qdrant-key" in params
        assert "WEAVIATE_URL=http://weaviate.test:8080" in params
        assert "WEAVIATE_API_KEY=weaviate-key" in params
        assert "WEAVIATE_GRPC_HOST=grpc.weaviate.test" in params

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    def test_submit_raises_on_missing_config(
        self,
        mock_config: MagicMock,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that submit propagates configuration errors."""
        mock_config.side_effect = ValueError("Missing required Databricks config")
        service = DatabricksRayService()

        with pytest.raises(ValueError, match="Missing required Databricks config"):
            service.submit_s3_to_qdrant(
                namespace="test-namespace",
                s3_endpoint="http://minio.test:9000",
                s3_access_key_id="test-key",
                s3_secret_access_key="test-secret",
                s3_bucket="test-bucket",
                s3_prefix="inputs/",
                embedding_config=embedding_config,
                vector_db_config=vector_db_config,
                collection="test-collection",
            )

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_raises_on_client_error(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that submit wraps Databricks SDK errors in UpstreamError."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.job_id = 123
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_client.jobs.run_now.side_effect = Exception("boom")
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        with pytest.raises(UpstreamError, match="Failed to submit Databricks job"):
            service.submit_s3_to_qdrant(
                namespace="test-namespace",
                s3_endpoint="http://minio.test:9000",
                s3_access_key_id="test-key",
                s3_secret_access_key="test-secret",
                s3_bucket="test-bucket",
                s3_prefix="inputs/",
                embedding_config=embedding_config,
                vector_db_config=vector_db_config,
                collection="test-collection",
            )
