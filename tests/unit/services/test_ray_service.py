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

from config import EmbeddingConfig, ProviderType
from foundation.exceptions import UpstreamError
from core.services.ray_service import RayService

# =============================================================================
# Job Submission Tests
# =============================================================================


class TestRayServiceSubmitJob:
    """Test suite for RayService.submit_s3_to_vector_dbs method."""


# =============================================================================
# Image Job Submission Tests
# =============================================================================


class TestRayServiceSubmitImageJob:
    """Test suite for RayService.submit_image_job method."""

    @patch("core.services.ray_service.EmbeddingConfig.from_env")
    @patch("core.services.ray_service.RayJobConfig.from_env")
    @patch("core.services.ray_service.JobSubmissionClient")
    def test_submit_image_job_success(
        self,
        mock_client_cls: MagicMock,
        mock_config: MagicMock,
        mock_embed_config: MagicMock,
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

        mock_embed_config.return_value = EmbeddingConfig(
            provider_type=ProviderType.LOCAL_CLIP,
        )

        mock_client = MagicMock()
        mock_client.submit_job.return_value = "raysubmit_image123"
        mock_client_cls.return_value = mock_client

        job_id = ray_service.submit_image_job(
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

        result = ray_service.get_job_status("raysubmit_test123")

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

        result = ray_service.get_job_status("raysubmit_test123")

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

        result = ray_service.get_job_status("raysubmit_test123")

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
            ray_service.get_job_status("nonexistent-job")

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
            ray_service.get_job_status("raysubmit_test123")


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

        result = ray_service.get_job_logs("raysubmit_test123")

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

        result = ray_service.get_job_logs("raysubmit_test123")

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
            ray_service.get_job_logs("nonexistent-job")

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
            ray_service.get_job_logs("raysubmit_test123")


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
        ray_service.stop_job("raysubmit_test123")

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
            ray_service.stop_job("nonexistent-job")

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
            ray_service.stop_job("raysubmit_test123")


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
    ) -> None:
        """Test that dashboard address is taken from RayJobConfig.

        **Why this test is important:**
          - Dashboard address must be configurable for different environments
          - Docker Compose uses simple hostnames (ray-head:8265)
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

        ray_service.submit_image_job(
            s3_endpoint="http://minio.test:9000",
            s3_access_key_id="test-key",
            s3_secret_access_key="test-secret",
            s3_bucket="test-bucket",
            s3_prefix="inputs/",
            embedding_config=EmbeddingConfig(provider_type=ProviderType.LOCAL_CLIP),
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

        job_id = ray_service.submit_image_job(
            s3_endpoint="http://minio:9000",
            s3_access_key_id="minioadmin",
            s3_secret_access_key="minioadmin",
            s3_bucket="pipeline",
            s3_prefix="inputs/",
            embedding_config=EmbeddingConfig(provider_type=ProviderType.LOCAL_CLIP),
            collection="documents",
        )

        assert job_id == "raysubmit_docker123"
        mock_client_cls.assert_called_once_with("http://ray-head:8265")
