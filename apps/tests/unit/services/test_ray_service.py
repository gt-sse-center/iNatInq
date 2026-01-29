"""Unit tests for core.services.ray_service module.

This file tests the RayService class which provides a service layer for submitting
and managing Ray jobs via the Ray Jobs API.

# Test Coverage

The tests cover:
  - Job Submission: Environment variable construction, client interaction, error handling
  - Job Status: Status extraction, info parsing, error handling
  - Job Logs: Log retrieval, error handling
  - Job Stopping: Stop operation, error handling
  - Configuration: Ray address validation, dashboard address construction
  - Error Handling: UpstreamError on failures, missing configuration

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The Ray JobSubmissionClient is mocked to isolate service logic.

# Running Tests

Run with: pytest tests/unit/services/test_ray_service.py
"""

from unittest.mock import MagicMock, patch

import pytest

from config import EmbeddingConfig, VectorDBConfig
from core.exceptions import UpstreamError
from core.services.ray_service import RayService

# =============================================================================
# Job Submission Tests
# =============================================================================


class TestRayServiceSubmitJob:
    """Test suite for RayService.submit_s3_to_qdrant method."""

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_submit_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        ray_service: RayService,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that submit_s3_to_qdrant submits job successfully.

        **Why this test is important:**
          - Job submission is the core functionality
          - Validates Ray API interaction
          - Ensures environment variables are constructed correctly
          - Critical for job execution

        **What it tests:**
          - JobSubmissionClient is created with correct dashboard address
          - Job is submitted with correct entrypoint
          - Environment variables include all required config
          - Job ID is returned
        """
        # Mock config with dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "raysubmit_test123"
        mock_client_cls.return_value = mock_client

        job_id = ray_service.submit_s3_to_qdrant(
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

        assert job_id == "raysubmit_test123"

        # Verify client was created with correct dashboard address from config
        mock_client_cls.assert_called_once_with("http://ray-head.test-namespace:8265")

        # Verify job was submitted with correct entrypoint
        mock_client.submit_job.assert_called_once()
        call_kwargs = mock_client.submit_job.call_args[1]
        assert call_kwargs["entrypoint"] == "python -m core.ingestion.ray.process_s3_to_qdrant"

        # Verify environment variables
        env_vars = call_kwargs["runtime_env"]["env_vars"]
        assert env_vars["K8S_NAMESPACE"] == "test-namespace"
        assert env_vars["S3_PREFIX"] == "inputs/"
        assert env_vars["S3_ENDPOINT"] == "http://minio.test:9000"
        assert env_vars["S3_ACCESS_KEY_ID"] == "test-key"
        assert env_vars["S3_SECRET_ACCESS_KEY"] == "test-secret"
        assert env_vars["S3_BUCKET"] == "test-bucket"
        assert env_vars["VECTOR_DB_COLLECTION"] == "test-collection"
        assert env_vars["EMBEDDING_PROVIDER_TYPE"] == "ollama"

    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "http://qdrant.test:6333",
            "WEAVIATE_URL": "http://weaviate.test:8080",
        },
        clear=False,
    )
    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_submit_includes_optional_config(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        ray_service: RayService,
    ) -> None:
        """Test that submit includes optional configuration in env vars.

        **Why this test is important:**
          - Optional config enables flexibility
          - Validates conditional env var inclusion
          - Critical for advanced configuration
          - Validates config forwarding

        **What it tests:**
          - Optional embedding config is included
          - Vector DB env vars from environment are passed
        """
        # Mock config with dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.ml-system:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "raysubmit_test123"
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
            vector_size=768,
        )

        ray_service.submit_s3_to_qdrant(
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

        call_kwargs = mock_client.submit_job.call_args[1]
        env_vars = call_kwargs["runtime_env"]["env_vars"]
        assert env_vars["EMBEDDING_VECTOR_SIZE"] == "768"
        assert env_vars["OLLAMA_BASE_URL"] == "http://ollama.test:11434"
        assert env_vars["OLLAMA_MODEL"] == "nomic-embed-text"
        # Vector DB env vars come from environment (os.environ)
        assert env_vars["QDRANT_URL"] == "http://qdrant.test:6333"
        assert env_vars["WEAVIATE_URL"] == "http://weaviate.test:8080"

    @patch.dict(
        "os.environ",
        {
            "QDRANT_URL": "http://qdrant.test:6333",
            "QDRANT_API_KEY": "qdrant-key",
            "WEAVIATE_URL": "https://my-cluster.weaviate.cloud",
            "WEAVIATE_API_KEY": "weaviate-key",
            "WEAVIATE_GRPC_HOST": "grpc-my-cluster.weaviate.cloud",
        },
        clear=False,
    )
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_submit_passes_both_vector_db_env_vars(
        self,
        mock_client_class: MagicMock,
        ray_service: RayService,
    ) -> None:
        """Test that both Qdrant and Weaviate env vars are passed to Ray jobs.

        **Why this test is important:**
          - Ray jobs index BOTH databases simultaneously via create_both()
          - All database credentials must be passed to Ray workers
          - Critical for dual-write architecture

        **What it tests:**
          - QDRANT_URL, QDRANT_API_KEY are passed
          - WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_GRPC_HOST are passed
        """
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.submit_job.return_value = "job-123"

        embedding_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama.test:11434",
            ollama_model="nomic-embed-text",
            vector_size=768,
        )
        vector_db_config = VectorDBConfig(
            provider_type="qdrant",  # Provider type doesn't matter - both are indexed
            qdrant_url="http://qdrant.test:6333",
            collection="test_documents",
            vector_size=768,
        )

        ray_service.submit_s3_to_qdrant(
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

        call_kwargs = mock_client.submit_job.call_args[1]
        env_vars = call_kwargs["runtime_env"]["env_vars"]
        # Both Qdrant and Weaviate env vars should be passed
        assert env_vars["QDRANT_URL"] == "http://qdrant.test:6333"
        assert env_vars["QDRANT_API_KEY"] == "qdrant-key"
        assert env_vars["WEAVIATE_URL"] == "https://my-cluster.weaviate.cloud"
        assert env_vars["WEAVIATE_API_KEY"] == "weaviate-key"
        assert env_vars["WEAVIATE_GRPC_HOST"] == "grpc-my-cluster.weaviate.cloud"

    @patch("core.services.ray_service.RayJobConfig.from_env")
    def test_submit_raises_on_missing_dashboard_address(
        self,
        mock_config: MagicMock,
        ray_service: RayService,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that submit raises UpstreamError when dashboard_address is missing.

        **Why this test is important:**
          - Configuration validation prevents runtime errors
          - Clear error messages aid debugging
          - Critical for error handling
          - Validates config validation

        **What it tests:**
          - UpstreamError is raised when dashboard_address is None
          - Error message is descriptive
        """
        # Mock config with missing dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = None
        mock_config.return_value = mock_ray_config

        with pytest.raises(UpstreamError, match="RAY_DASHBOARD_ADDRESS not configured"):
            ray_service.submit_s3_to_qdrant(
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

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_submit_raises_on_client_error(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        ray_service: RayService,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that submit raises UpstreamError on client errors.

        **Why this test is important:**
          - Client errors should be wrapped consistently
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation
          - Validates error wrapping

        **What it tests:**
          - Client exceptions are wrapped in UpstreamError
          - Error message includes context
        """
        # Mock config with dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.ml-system:8265"
        mock_config.return_value = mock_ray_config

        # Mock client with error
        mock_client = MagicMock()
        mock_client.submit_job.side_effect = Exception("Connection refused")
        mock_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Failed to submit Ray job"):
            ray_service.submit_s3_to_qdrant(
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


# =============================================================================
# Image Job Submission Tests
# =============================================================================


class TestRayServiceSubmitImageJob:
    """Test suite for RayService.submit_image_job method."""

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_submit_image_job_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        ray_service: RayService,
    ) -> None:
        """Test that submit_image_job submits job with correct entrypoint and env vars.

        **Why this test is important:**
          - Image job submission is the core of the image ingestion API
          - Validates Ray API interaction for image pipeline
          - Ensures S3 bucket/prefix and collection are passed to workers
        """
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        mock_client = MagicMock()
        mock_client.submit_job.return_value = "raysubmit_image123"
        mock_client_cls.return_value = mock_client

        job_id = ray_service.submit_image_job(
            namespace="test-namespace",
            s3_endpoint="http://minio.test:9000",
            s3_access_key_id="test-key",
            s3_secret_access_key="test-secret",
            s3_bucket="pipeline",
            s3_prefix="images/",
            collection="documents",
        )

        assert job_id == "raysubmit_image123"
        mock_client.submit_job.assert_called_once()
        call_kwargs = mock_client.submit_job.call_args[1]
        assert call_kwargs["entrypoint"] == "python -m core.ingestion.ray.process_s3_images"
        env_vars = call_kwargs["runtime_env"]["env_vars"]
        assert env_vars["K8S_NAMESPACE"] == "test-namespace"
        assert env_vars["S3_BUCKET"] == "pipeline"
        assert env_vars["S3_PREFIX"] == "images/"
        assert env_vars["VECTOR_DB_COLLECTION"] == "documents"
        assert "pillow" in call_kwargs["runtime_env"]["pip"]

    @patch("core.services.ray_service.RayJobConfig.from_env")
    def test_submit_image_job_raises_on_missing_dashboard_address(
        self,
        mock_config: MagicMock,
        ray_service: RayService,
    ) -> None:
        """Test that submit_image_job raises UpstreamError when dashboard_address is missing."""
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = None
        mock_config.return_value = mock_ray_config

        with pytest.raises(UpstreamError, match="RAY_DASHBOARD_ADDRESS not configured"):
            ray_service.submit_image_job(
                namespace="test-namespace",
                s3_endpoint="http://minio.test:9000",
                s3_access_key_id="test-key",
                s3_secret_access_key="test-secret",
                s3_bucket="pipeline",
                s3_prefix="images/",
                collection="documents",
            )


# =============================================================================
# Job Status Tests
# =============================================================================


class TestRayServiceGetJobStatus:
    """Test suite for RayService.get_job_status method."""

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_get_status_success(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_job_status returns status information.

        **Why this test is important:**
          - Status checking is needed for monitoring
          - Validates status extraction
          - Critical for job tracking
          - Validates data parsing

        **What it tests:**
          - Client get_job_status and get_job_info are called
          - Status is extracted correctly
          - Message is included if available
        """
        # Mock config with dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_status = MagicMock()
        mock_status.value = "RUNNING"
        mock_info = MagicMock()
        mock_info.message = "Job is running"
        mock_client.get_job_status.return_value = mock_status
        mock_client.get_job_info.return_value = mock_info
        mock_client_cls.return_value = mock_client

        result = ray_service.get_job_status("raysubmit_test123", "test-namespace")

        assert result["status"] == "RUNNING"
        assert result["message"] == "Job is running"

        mock_client_cls.assert_called_once_with("http://ray-head.test-namespace:8265")
        mock_client.get_job_status.assert_called_once_with("raysubmit_test123")
        mock_client.get_job_info.assert_called_once_with("raysubmit_test123")

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_get_status_handles_missing_message(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_status handles missing message gracefully.

        **Why this test is important:**
          - Message may not always be available
          - Graceful handling prevents crashes
          - Critical for robustness
          - Validates defensive programming

        **What it tests:**
          - Missing message returns None
          - No AttributeError is raised
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_status = MagicMock()
        mock_status.value = "PENDING"
        mock_client.get_job_status.return_value = mock_status
        mock_client.get_job_info.return_value = None
        mock_client_cls.return_value = mock_client

        result = ray_service.get_job_status("raysubmit_test123", "test-namespace")

        assert result["status"] == "PENDING"
        assert result["message"] is None

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_get_status_handles_string_status(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_status handles string status values.

        **Why this test is important:**
          - Status might be string or enum
          - Flexible handling supports both formats
          - Critical for compatibility
          - Validates type handling

        **What it tests:**
          - String status values are handled
          - Status without .value attribute works
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_client.get_job_status.return_value = "SUCCEEDED"
        mock_client.get_job_info.return_value = None
        mock_client_cls.return_value = mock_client

        result = ray_service.get_job_status("raysubmit_test123", "test-namespace")

        assert result["status"] == "SUCCEEDED"

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_get_status_raises_on_client_error(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_status raises UpstreamError on client errors.

        **Why this test is important:**
          - Client errors should be wrapped
          - Consistent error handling
          - Critical for error propagation
          - Validates error wrapping

        **What it tests:**
          - Client exceptions are wrapped in UpstreamError
          - Error message includes context
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client with error
        mock_client = MagicMock()
        mock_client.get_job_status.side_effect = Exception("Job not found")
        mock_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Failed to get job status"):
            ray_service.get_job_status("nonexistent-job", "test-namespace")

    @patch("core.services.ray_service.RayJobConfig.from_env")
    def test_get_status_raises_on_missing_dashboard_address(
        self, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_status raises UpstreamError when dashboard_address is missing.

        **Why this test is important:**
          - Configuration validation prevents runtime errors
          - Validates that dashboard address is required
          - Critical for error handling

        **What it tests:**
          - UpstreamError is raised when dashboard_address is None
        """
        # Mock config with missing dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = None
        mock_config.return_value = mock_ray_config

        with pytest.raises(UpstreamError, match="RAY_DASHBOARD_ADDRESS not configured"):
            ray_service.get_job_status("raysubmit_test123", "test-namespace")


# =============================================================================
# Job Logs Tests
# =============================================================================


class TestRayServiceGetJobLogs:
    """Test suite for RayService.get_job_logs method."""

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_get_logs_success(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_job_logs returns log content.

        **Why this test is important:**
          - Log retrieval is needed for debugging
          - Validates client interaction
          - Critical for troubleshooting
          - Validates log extraction

        **What it tests:**
          - Client get_job_logs is called
          - Log content is returned as string
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = "Log line 1\nLog line 2\n"
        mock_client_cls.return_value = mock_client

        result = ray_service.get_job_logs("raysubmit_test123", "test-namespace")

        assert result == "Log line 1\nLog line 2\n"

        mock_client_cls.assert_called_once_with("http://ray-head.test-namespace:8265")
        mock_client.get_job_logs.assert_called_once_with("raysubmit_test123")

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_get_logs_converts_to_string(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_job_logs converts result to string.

        **Why this test is important:**
          - Ray client may return different types
          - String conversion ensures consistent output
          - Critical for API compatibility
          - Validates type conversion

        **What it tests:**
          - Non-string results are converted to string
          - str() is called on result
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client with non-string result
        mock_client = MagicMock()
        mock_client.get_job_logs.return_value = 12345
        mock_client_cls.return_value = mock_client

        result = ray_service.get_job_logs("raysubmit_test123", "test-namespace")

        assert result == "12345"
        assert isinstance(result, str)

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_get_logs_raises_on_client_error(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_logs raises UpstreamError on client errors.

        **Why this test is important:**
          - Client errors should be wrapped
          - Consistent error handling
          - Critical for error propagation
          - Validates error wrapping

        **What it tests:**
          - Client exceptions are wrapped in UpstreamError
          - Error message includes context
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client with error
        mock_client = MagicMock()
        mock_client.get_job_logs.side_effect = Exception("Job not found")
        mock_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Failed to get job logs"):
            ray_service.get_job_logs("nonexistent-job", "test-namespace")

    @patch("core.services.ray_service.RayJobConfig.from_env")
    def test_get_logs_raises_on_missing_dashboard_address(
        self, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that get_logs raises UpstreamError when dashboard_address is missing.

        **Why this test is important:**
          - Configuration validation prevents runtime errors
          - Validates that dashboard address is required

        **What it tests:**
          - UpstreamError is raised when dashboard_address is None
        """
        # Mock config with missing dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = None
        mock_config.return_value = mock_ray_config

        with pytest.raises(UpstreamError, match="RAY_DASHBOARD_ADDRESS not configured"):
            ray_service.get_job_logs("raysubmit_test123", "test-namespace")


# =============================================================================
# Job Stop Tests
# =============================================================================


class TestRayServiceStopJob:
    """Test suite for RayService.stop_job method."""

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_stop_job_success(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that stop_job stops the job successfully.

        **Why this test is important:**
          - Job stopping is needed for cancellation
          - Validates client interaction
          - Critical for resource management
          - Validates stop operation

        **What it tests:**
          - Client stop_job is called with correct job ID
          - Method completes without errors
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_client.stop_job.return_value = None
        mock_client_cls.return_value = mock_client

        # Should not raise
        ray_service.stop_job("raysubmit_test123", "test-namespace")

        mock_client_cls.assert_called_once_with("http://ray-head.test-namespace:8265")
        mock_client.stop_job.assert_called_once_with("raysubmit_test123")

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_stop_job_raises_on_client_error(
        self, mock_client_cls: MagicMock, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that stop_job raises UpstreamError on client errors.

        **Why this test is important:**
          - Client errors should be wrapped
          - Consistent error handling
          - Critical for error propagation
          - Validates error wrapping

        **What it tests:**
          - Client exceptions are wrapped in UpstreamError
          - Error message includes context
        """
        # Mock config
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head.test-namespace:8265"
        mock_config.return_value = mock_ray_config

        # Mock client with error
        mock_client = MagicMock()
        mock_client.stop_job.side_effect = Exception("Job not found")
        mock_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Failed to stop job"):
            ray_service.stop_job("nonexistent-job", "test-namespace")

    @patch("core.services.ray_service.RayJobConfig.from_env")
    def test_stop_job_raises_on_missing_dashboard_address(
        self, mock_config: MagicMock, ray_service: RayService
    ) -> None:
        """Test that stop_job raises UpstreamError when dashboard_address is missing.

        **Why this test is important:**
          - Configuration validation prevents runtime errors
          - Validates that dashboard address is required

        **What it tests:**
          - UpstreamError is raised when dashboard_address is None
        """
        # Mock config with missing dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = None
        mock_config.return_value = mock_ray_config

        with pytest.raises(UpstreamError, match="RAY_DASHBOARD_ADDRESS not configured"):
            ray_service.stop_job("raysubmit_test123", "test-namespace")


# =============================================================================
# Dashboard Address Tests
# =============================================================================


class TestRayServiceDashboardAddress:
    """Test suite for dashboard address configuration."""

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_dashboard_address_uses_config(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        ray_service: RayService,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that dashboard address is taken from RayJobConfig.

        **Why this test is important:**
          - Dashboard address must be configurable for different environments
          - Docker Compose uses simple hostnames (ray-head:8265)
          - Kubernetes uses namespace DNS (ray-head.{namespace}:8265)
          - Critical for environment-agnostic deployment

        **What it tests:**
          - Dashboard address comes from config, not hardcoded
          - Client is created with address from config
        """
        # Mock config with custom dashboard_address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://custom-ray-head:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "raysubmit_test123"
        mock_client_cls.return_value = mock_client

        ray_service.submit_s3_to_qdrant(
            namespace="custom-namespace",
            s3_endpoint="http://minio.test:9000",
            s3_access_key_id="test-key",
            s3_secret_access_key="test-secret",
            s3_bucket="test-bucket",
            s3_prefix="inputs/",
            embedding_config=embedding_config,
            vector_db_config=vector_db_config,
            collection="test-collection",
        )

        # Verify client was created with address from config (not namespace-based)
        mock_client_cls.assert_called_once_with("http://custom-ray-head:8265")

    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_dashboard_address_docker_compose_style(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        ray_service: RayService,
        embedding_config: EmbeddingConfig,
        vector_db_config: VectorDBConfig,
    ) -> None:
        """Test that Docker Compose style addresses work correctly.

        **Why this test is important:**
          - Docker Compose uses simple service names without namespace
          - Validates that http://ray-head:8265 works
          - Critical for local development environment

        **What it tests:**
          - Simple hostname (no namespace) is used correctly
          - Works with Docker Compose naming convention
        """
        # Mock config with Docker Compose style address
        mock_ray_config = MagicMock()
        mock_ray_config.dashboard_address = "http://ray-head:8265"
        mock_config.return_value = mock_ray_config

        # Mock client
        mock_client = MagicMock()
        mock_client.submit_job.return_value = "raysubmit_docker123"
        mock_client_cls.return_value = mock_client

        job_id = ray_service.submit_s3_to_qdrant(
            namespace="ml-system",
            s3_endpoint="http://minio:9000",
            s3_access_key_id="minioadmin",
            s3_secret_access_key="minioadmin",
            s3_bucket="pipeline",
            s3_prefix="inputs/",
            embedding_config=embedding_config,
            vector_db_config=vector_db_config,
            collection="documents",
        )

        assert job_id == "raysubmit_docker123"
        mock_client_cls.assert_called_once_with("http://ray-head:8265")
