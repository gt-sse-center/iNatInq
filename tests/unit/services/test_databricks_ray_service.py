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

from config import EmbeddingConfig, EmbeddingConfig, ProviderType
from core.exceptions import UpstreamError
from core.services.databricks_ray_service import DatabricksRayService

# =============================================================================
# Image Job Submission Tests
# =============================================================================


class TestDatabricksRayServiceSubmitImageJob:
    """Test suite for DatabricksRayService.submit_image_job."""

    @patch.dict(
        "os.environ",
        {
            "CLIP_API_KEY": "clip-key",
            "INAT_MAX_ROWS": "250",
            "INAT_METADATA_URL": "s3://inaturalist-open-data/photos.csv.gz",
            "INATINQ_SRC_DIR": "/Workspace/Users/test/iNatInq/src",
            "VECTOR_DB_PROVIDER": "qdrant",
        },
        clear=False,
    )
    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_image_job_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that submit_image_job submits a Databricks run with image params."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.job_id = 123
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.run_id = 789
        mock_client.jobs.run_now.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedding_config = EmbeddingConfig(
            provider_type=ProviderType.HOSTED_CLIP,
            clip_url="http://clip.test:8000",
            clip_model="ViT-B/32",
            clip_timeout=90,
            clip_circuit_breaker_threshold=7,
            clip_circuit_breaker_timeout=45,
            clip_max_batch_size=16,
            clip_vector_size=512,
            image_batch_size=12,
            image_max_size_mb=8.5,
            image_target_size=336,
        )

        service = DatabricksRayService()
        run_id = service.submit_image_job(
            namespace="test-namespace",
            s3_endpoint="http://minio.test:9000",
            s3_access_key_id="test-key",
            s3_secret_access_key="test-secret",
            s3_bucket="test-bucket",
            s3_prefix="images/",
            embedding_config=embedding_config,
            collection="test-image-collection",
        )

        assert run_id == 789
        mock_client_cls.assert_called_once_with(host="https://dbc.example.cloud", token="databricks-token")

        call_kwargs = mock_client.jobs.run_now.call_args.kwargs
        assert call_kwargs["job_id"] == 123
        params = call_kwargs["python_params"]
        assert "K8S_NAMESPACE=test-namespace" in params
        assert "S3_PREFIX=images/" in params
        assert "S3_ENDPOINT=http://minio.test:9000" in params
        assert "S3_ACCESS_KEY_ID=test-key" in params
        assert "S3_SECRET_ACCESS_KEY=test-secret" in params
        assert "S3_BUCKET=test-bucket" in params
        assert "VECTOR_DB_COLLECTION=test-image-collection" in params
        assert "CLIP_URL=http://clip.test:8000" in params
        assert "CLIP_MODEL=ViT-B/32" in params
        assert "EMBEDDING_PROVIDER=hosted_clip" in params
        assert "CLIP_TIMEOUT=90" in params
        assert "CLIP_CIRCUIT_BREAKER_THRESHOLD=7" in params
        assert "CLIP_CIRCUIT_BREAKER_TIMEOUT=45" in params
        assert "CLIP_MAX_BATCH_SIZE=16" in params
        assert "CLIP_VECTOR_SIZE=512" in params
        assert "IMAGE_BATCH_SIZE=12" in params
        assert "IMAGE_MAX_SIZE_MB=8.5" in params
        assert "IMAGE_TARGET_SIZE=336" in params
        assert "CLIP_API_KEY=clip-key" in params
        assert "INAT_MAX_ROWS=250" in params
        assert "INAT_METADATA_URL=s3://inaturalist-open-data/photos.csv.gz" in params
        assert "INATINQ_SRC_DIR=/Workspace/Users/test/iNatInq/src" in params
        assert "VECTOR_DB_PROVIDER=qdrant" in params

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_image_job_raises_on_client_error(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that submit_image_job wraps Databricks SDK errors in UpstreamError."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.job_id = 123
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_client.jobs.run_now.side_effect = Exception("boom")
        mock_client_cls.return_value = mock_client

        embedding_config = EmbeddingConfig(
            provider_type=ProviderType.LOCAL_CLIP,
            clip_url="http://clip.test:8000",
            clip_model="ViT-B/32",
        )

        service = DatabricksRayService()
        with pytest.raises(UpstreamError, match="Failed to submit Databricks image job"):
            service.submit_image_job(
                namespace="test-namespace",
                s3_endpoint="http://minio.test:9000",
                s3_access_key_id="test-key",
                s3_secret_access_key="test-secret",
                s3_bucket="test-bucket",
                s3_prefix="images/",
                embedding_config=embedding_config,
                collection="test-image-collection",
            )


class TestDatabricksRayServiceSubmitINatImageJob:
    """Test suite for DatabricksRayService.submit_inat_image_job."""

    @patch.dict(
        "os.environ",
        {
            "INAT_MAX_ROWS": "250",
            "INAT_METADATA_URL": "s3://inaturalist-open-data/photos.csv.gz",
            "INATINQ_SRC_DIR": "/Workspace/Users/test/iNatInq/src",
        },
        clear=False,
    )
    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_inat_image_job_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that submit_inat_image_job uses DATABRICKS_INAT_JOB_ID."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.job_id = 123
        mock_databricks_config.inat_job_id = 987
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.run_id = 456
        mock_client.jobs.run_now.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedding_config = EmbeddingConfig(
            provider_type=ProviderType.LOCAL_CLIP,
            clip_url="http://clip.test:8000",
            clip_model="ViT-B/32",
        )

        service = DatabricksRayService()
        run_id = service.submit_inat_image_job(
            namespace="test-namespace",
            s3_prefix="images/",
            embedding_config=embedding_config,
            collection="test-image-collection",
        )

        assert run_id == 456
        call_kwargs = mock_client.jobs.run_now.call_args.kwargs
        assert call_kwargs["job_id"] == 987
        params = call_kwargs["python_params"]
        assert "INAT_MAX_ROWS=250" in params
        assert "INAT_METADATA_URL=s3://inaturalist-open-data/photos.csv.gz" in params
        assert not any(param.startswith("S3_ENDPOINT=") for param in params)
        assert not any(param.startswith("S3_ACCESS_KEY_ID=") for param in params)
        assert not any(param.startswith("S3_SECRET_ACCESS_KEY=") for param in params)
        assert not any(param.startswith("S3_BUCKET=") for param in params)

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    def test_submit_inat_image_job_raises_without_inat_job_id(self, mock_config: MagicMock) -> None:
        """Test dedicated iNat job id is required for iNat submission method."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.job_id = 123
        mock_databricks_config.inat_job_id = None
        mock_config.return_value = mock_databricks_config

        embedding_config = EmbeddingConfig(
            provider_type=ProviderType.LOCAL_CLIP,
            clip_url="http://clip.test:8000",
            clip_model="ViT-B/32",
        )

        service = DatabricksRayService()
        with pytest.raises(ValueError, match="DATABRICKS_INAT_JOB_ID"):
            service.submit_inat_image_job(
                namespace="test-namespace",
                s3_prefix="images/",
                embedding_config=embedding_config,
                collection="test-image-collection",
            )


class TestDatabricksRayServiceSubmitS3CDCJobs:
    """Test suite for Databricks Auto Loader producer submission."""

    @patch.dict(
        "os.environ",
        {
            "S3_BUCKET": "pipeline",
            "S3_PREFIX": "images",
            "AUTOLOADER_BRONZE_TABLE": "main.default.images_bronze",
            "AUTOLOADER_SCHEMA_LOCATION": "dbfs:/pipelines/schema",
            "AUTOLOADER_CHECKPOINT_LOCATION": "dbfs:/pipelines/checkpoints",
            "S3_ENDPOINT": "http://minio.test:9000",
            "S3_ACCESS_KEY_ID": "test-key",
            "S3_SECRET_ACCESS_KEY": "test-secret",
            "S3_USE_SSL": "false",
            "S3_PATH_STYLE": "true",
            "INATINQ_SRC_DIR": "/Workspace/Users/test/iNatInq/src",
        },
        clear=False,
    )
    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_s3_autoloader_job_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Auto Loader submission should target dedicated job id."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_databricks_config.s3_autoloader_job_id = 321
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.run_id = 654
        mock_client.jobs.run_now.return_value = mock_response
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        run_id = service.submit_s3_autoloader_job(namespace="test-namespace")

        assert run_id == 654
        mock_config.assert_called_once_with(require_job_id=False)
        call_kwargs = mock_client.jobs.run_now.call_args.kwargs
        assert call_kwargs["job_id"] == 321
        params = call_kwargs["python_params"]
        assert "K8S_NAMESPACE=test-namespace" in params
        assert "S3_BUCKET=pipeline" in params
        assert "S3_PREFIX=images" in params
        assert "AUTOLOADER_BRONZE_TABLE=main.default.images_bronze" in params
        assert "AUTOLOADER_SCHEMA_LOCATION=dbfs:/pipelines/schema" in params
        assert "AUTOLOADER_CHECKPOINT_LOCATION=dbfs:/pipelines/checkpoints" in params
        assert "S3_ENDPOINT=http://minio.test:9000" in params
        assert "S3_ACCESS_KEY_ID=test-key" in params
        assert "S3_SECRET_ACCESS_KEY=test-secret" in params
        assert "S3_USE_SSL=false" in params
        assert "S3_PATH_STYLE=true" in params
        assert "INATINQ_SRC_DIR=/Workspace/Users/test/iNatInq/src" in params

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    def test_submit_s3_autoloader_job_raises_without_dedicated_job_id(
        self,
        mock_config: MagicMock,
    ) -> None:
        """Dedicated Auto Loader job id should be required."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.s3_autoloader_job_id = None
        mock_config.return_value = mock_databricks_config

        service = DatabricksRayService()
        with pytest.raises(ValueError, match="DATABRICKS_S3_AUTOLOADER_JOB_ID"):
            service.submit_s3_autoloader_job(namespace="test-namespace")
        mock_config.assert_called_once_with(require_job_id=False)


# =============================================================================
# Job Stop Tests
# =============================================================================


class TestDatabricksRayServiceStopRun:
    """Test suite for DatabricksRayService.stop_run."""

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_stop_run_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that stop_run cancels a Databricks job run."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        service.stop_run(123)

        mock_client_cls.assert_called_once_with(host="https://dbc.example.cloud", token="databricks-token")
        mock_client.jobs.cancel_run.assert_called_once_with(run_id=123)

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_stop_run_raises_on_client_error(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that stop_run wraps Databricks SDK errors in UpstreamError."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_client.jobs.cancel_run.side_effect = Exception("boom")
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        with pytest.raises(UpstreamError, match="Failed to stop Databricks job"):
            service.stop_run("456")


# =============================================================================
# Job Status/Logs Tests
# =============================================================================


class TestDatabricksRayServiceStatusLogs:
    """Test suite for DatabricksRayService status/log retrieval."""

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_get_run_status_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that get_run_status returns run state fields."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_config.return_value = mock_databricks_config

        mock_state = MagicMock()
        mock_state.life_cycle_state = "RUNNING"
        mock_state.result_state = None
        mock_state.state_message = "Job is running"
        mock_run = MagicMock()
        mock_run.state = mock_state

        mock_client = MagicMock()
        mock_client.jobs.get_run.return_value = mock_run
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        status = service.get_run_status(123)

        assert status["life_cycle_state"] == "RUNNING"
        assert status["result_state"] is None
        assert status["state_message"] == "Job is running"

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_get_run_status_raises_on_error(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that get_run_status wraps SDK errors in UpstreamError."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_client.jobs.get_run.side_effect = Exception("boom")
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        with pytest.raises(UpstreamError, match="Failed to get Databricks run status"):
            service.get_run_status(123)

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_get_run_output_prefers_logs(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that get_run_output returns logs when available."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_config.return_value = mock_databricks_config

        mock_output = MagicMock()
        mock_output.logs = "log output"

        mock_client = MagicMock()
        mock_client.jobs.get_run_output.return_value = mock_output
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        assert service.get_run_output(123) == "log output"

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_get_run_output_raises_on_error(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test that get_run_output wraps SDK errors in UpstreamError."""
        mock_databricks_config = MagicMock()
        mock_databricks_config.host = "https://dbc.example.cloud"
        mock_databricks_config.token = "databricks-token"
        mock_config.return_value = mock_databricks_config

        mock_client = MagicMock()
        mock_client.jobs.get_run_output.side_effect = Exception("boom")
        mock_client_cls.return_value = mock_client

        service = DatabricksRayService()
        with pytest.raises(UpstreamError, match="Failed to get Databricks run output"):
            service.get_run_output(123)

    @patch("core.services.databricks_ray_service.DatabricksRayJobConfig.from_env")
    @patch("core.services.databricks_ray_service.WorkspaceClient")
    def test_submit_raises_on_client_error(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
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

        embedding_config = EmbeddingConfig(
            provider_type=ProviderType.HOSTED_CLIP,
            clip_url="http://clip.test:8000",
            clip_model="ViT-B/32",
            clip_timeout=90,
            clip_circuit_breaker_threshold=7,
            clip_circuit_breaker_timeout=45,
            clip_max_batch_size=16,
            clip_vector_size=512,
            image_batch_size=12,
            image_max_size_mb=8.5,
            image_target_size=336,
        )
        service = DatabricksRayService()
        with pytest.raises(UpstreamError, match="Failed to submit Databricks image job"):
            service.submit_image_job(
                namespace="test-namespace",
                s3_endpoint="http://minio.test:9000",
                s3_access_key_id="test-key",
                s3_secret_access_key="test-secret",
                s3_bucket="test-bucket",
                s3_prefix="inputs/",
                embedding_config=embedding_config,
                collection="test-collection",
            )
