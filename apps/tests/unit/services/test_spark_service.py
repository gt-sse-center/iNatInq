"""Unit tests for core.services.spark_service module.

This file tests the SparkService class which provides a high-level service layer
for submitting and managing Spark jobs via the Kubernetes Spark Operator.

# Test Coverage

The tests cover:
  - Service Initialization: Namespace configuration, client creation
  - Job Submission: Auto-generated names, custom names, parameter passing
  - Job Status: Status extraction, error handling
  - Job Listing: Response parsing, empty lists
  - Job Deletion: Deletion confirmation, error handling
  - Error Handling: Client exceptions, validation errors
  - Integration: End-to-end job lifecycle

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The underlying SparkJobClient is mocked to isolate service logic.

# Running Tests

Run with: pytest tests/unit/services/test_spark_service.py
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.services.spark_service import SparkService

# =============================================================================
# Service Initialization Tests
# =============================================================================


class TestSparkServiceInit:
    """Test suite for SparkService initialization."""

    def test_creates_service_with_default_namespace(self, mock_spark_client: MagicMock) -> None:
        """Test that service is created with default namespace.

        **Why this test is important:**
          - Default namespace must work out of the box
          - Ensures sensible defaults for common use cases
          - Validates that service is created successfully
          - Critical for ease of use

        **What it tests:**
          - Service is created with default ml-system namespace
          - Client is created automatically
        """
        with patch(
            "core.services.spark_service.SparkJobClient",
            return_value=mock_spark_client,
        ):
            service = SparkService()

            assert service.namespace == "ml-system"
            assert service.client is not None

    def test_creates_service_with_custom_namespace(self, mock_spark_client: MagicMock) -> None:
        """Test that service accepts custom namespace.

        **Why this test is important:**
          - Custom namespace allows multi-tenant deployments
          - Different environments may need different namespaces
          - Critical for flexibility
          - Validates parameter passing

        **What it tests:**
          - Custom namespace value is applied
          - Client is initialized with custom namespace
        """
        with patch(
            "core.services.spark_service.SparkJobClient",
            return_value=mock_spark_client,
        ) as mock_client_cls:
            service = SparkService(namespace="custom-namespace")

            assert service.namespace == "custom-namespace"
            mock_client_cls.assert_called_once_with(namespace="custom-namespace")


# =============================================================================
# Job Submission Tests
# =============================================================================


class TestSparkServiceSubmitProcessingJob:
    """Test suite for SparkService.submit_processing_job method."""

    def test_submit_with_auto_generated_name(self, spark_service: SparkService) -> None:
        """Test that submit generates job name automatically.

        **Why this test is important:**
          - Auto-generated names prevent conflicts
          - Ensures unique job names across submissions
          - Critical for job tracking
          - Validates name generation logic

        **What it tests:**
          - Job name is auto-generated with timestamp and UUID
          - Job name format follows pattern: s3-to-vector-db-<timestamp>-<uuid>
          - Client submit_job is called with generated name
        """
        result = spark_service.submit_processing_job(
            s3_prefix="inputs/",
            collection="documents",
        )

        assert "job_name" in result
        assert result["job_name"].startswith("s3-to-vector-db-")
        assert result["status"] == "submitted"
        assert result["namespace"] == "test-namespace"
        assert result["s3_prefix"] == "inputs/"
        assert result["collection"] == "documents"
        assert "submitted_at" in result

        spark_service.client.submit_job.assert_called_once()
        call_kwargs = spark_service.client.submit_job.call_args[1]
        assert call_kwargs["s3_prefix"] == "inputs/"
        assert call_kwargs["collection"] == "documents"

    def test_submit_with_custom_name(self, spark_service: SparkService) -> None:
        """Test that submit accepts custom job name.

        **Why this test is important:**
          - Custom names allow explicit job tracking
          - Supports idempotent job submission
          - Critical for integration scenarios
          - Validates custom name handling

        **What it tests:**
          - Custom job name is used instead of auto-generated
          - Job name is passed to client
          - Response includes custom name
        """
        result = spark_service.submit_processing_job(
            s3_prefix="inputs/",
            collection="documents",
            job_name="custom-job-name",
        )

        assert result["job_name"] == "custom-job-name"
        spark_service.client.submit_job.assert_called_once()
        call_kwargs = spark_service.client.submit_job.call_args[1]
        assert call_kwargs["name"] == "custom-job-name"

    def test_submit_with_custom_executor_config(self, spark_service: SparkService) -> None:
        """Test that submit passes executor configuration.

        **Why this test is important:**
          - Executor config controls job parallelism
          - Different jobs need different resources
          - Critical for performance tuning
          - Validates parameter forwarding

        **What it tests:**
          - Executor instances and memory are passed to client
          - All configuration parameters are forwarded correctly
        """
        spark_service.submit_processing_job(
            s3_prefix="inputs/",
            collection="documents",
            executor_instances=5,
            executor_memory="2g",
        )

        spark_service.client.submit_job.assert_called_once()
        call_kwargs = spark_service.client.submit_job.call_args[1]
        assert call_kwargs["executor_instances"] == 5
        assert call_kwargs["executor_memory"] == "2g"

    def test_submit_returns_metadata(self, spark_service: SparkService) -> None:
        """Test that submit returns complete job metadata.

        **Why this test is important:**
          - Metadata is needed for job tracking
          - Response format is API contract
          - Critical for API consistency
          - Validates response structure

        **What it tests:**
          - Response includes all required fields
          - Timestamps are in ISO format
          - Status is set correctly
        """
        result = spark_service.submit_processing_job(
            s3_prefix="inputs/test/",
            collection="test_docs",
            job_name="test-job",
        )

        assert result["job_name"] == "test-job"
        assert result["status"] == "submitted"
        assert result["namespace"] == "test-namespace"
        assert result["s3_prefix"] == "inputs/test/"
        assert result["collection"] == "test_docs"
        assert "submitted_at" in result
        # Validate timestamp format
        datetime.fromisoformat(result["submitted_at"])

    def test_submit_raises_on_client_error(self, spark_service: SparkService) -> None:
        """Test that submit propagates client errors.

        **Why this test is important:**
          - Error propagation enables proper error handling
          - Clients can fail for various reasons
          - Critical for debugging
          - Validates error handling

        **What it tests:**
          - Client exceptions are propagated
          - Service does not swallow errors
        """
        spark_service.client.submit_job.side_effect = RuntimeError("K8s API error")

        with pytest.raises(RuntimeError, match="K8s API error"):
            spark_service.submit_processing_job(s3_prefix="inputs/", collection="documents")


# =============================================================================
# Job Status Tests
# =============================================================================


class TestSparkServiceGetJobStatus:
    """Test suite for SparkService.get_job_status method."""

    def test_get_status_success(self, spark_service: SparkService) -> None:
        """Test that get_status extracts status information.

        **Why this test is important:**
          - Status information is needed for monitoring
          - Validates response parsing
          - Critical for job tracking
          - Validates data extraction

        **What it tests:**
          - SparkApplication status is parsed correctly
          - All status fields are extracted
          - Response format matches expectations
        """
        mock_spark_app = {
            "status": {
                "applicationState": {"state": "RUNNING"},
                "sparkApplicationState": "RUNNING",
                "driverInfo": {"podName": "driver-pod"},
                "executionAttempts": 1,
                "lastSubmissionAttemptTime": "2026-01-12T15:30:45Z",
                "terminationTime": None,
            }
        }
        spark_service.client.get_job_status.return_value = mock_spark_app

        result = spark_service.get_job_status("test-job")

        assert result["job_name"] == "test-job"
        assert result["state"] == "RUNNING"
        assert result["spark_state"] == "RUNNING"
        assert result["driver_info"] == {"podName": "driver-pod"}
        assert result["execution_attempts"] == 1
        assert result["last_submission_attempt_time"] == "2026-01-12T15:30:45Z"
        assert result["termination_time"] is None

        spark_service.client.get_job_status.assert_called_once_with("test-job")

    def test_get_status_handles_missing_fields(self, spark_service: SparkService) -> None:
        """Test that get_status handles missing status fields gracefully.

        **Why this test is important:**
          - SparkApplication status may be incomplete
          - Graceful handling prevents crashes
          - Critical for robustness
          - Validates defensive programming

        **What it tests:**
          - Missing fields return default values
          - No KeyError is raised
          - UNKNOWN state is used as fallback
        """
        mock_spark_app = {"status": {}}
        spark_service.client.get_job_status.return_value = mock_spark_app

        result = spark_service.get_job_status("test-job")

        assert result["job_name"] == "test-job"
        assert result["state"] == "UNKNOWN"
        assert result["spark_state"] == "UNKNOWN"
        assert result["driver_info"] == {}
        assert result["execution_attempts"] == 0
        assert result["last_submission_attempt_time"] is None
        assert result["termination_time"] is None

    def test_get_status_propagates_client_error(self, spark_service: SparkService) -> None:
        """Test that get_status propagates client errors.

        **Why this test is important:**
          - Client errors need to propagate
          - Job not found should raise error
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - RuntimeError from client is propagated
          - Error message is preserved
        """
        spark_service.client.get_job_status.side_effect = RuntimeError("Job not found")

        with pytest.raises(RuntimeError, match="Job not found"):
            spark_service.get_job_status("nonexistent-job")


# =============================================================================
# Job Listing Tests
# =============================================================================


class TestSparkServiceListJobs:
    """Test suite for SparkService.list_jobs method."""

    def test_list_jobs_returns_summaries(self, spark_service: SparkService) -> None:
        """Test that list_jobs returns job summaries.

        **Why this test is important:**
          - Job listing is needed for monitoring
          - Validates response parsing
          - Critical for job management
          - Validates data extraction

        **What it tests:**
          - SparkApplication list is parsed correctly
          - Job summaries include required fields
          - Response format matches expectations
        """
        mock_response = {
            "items": [
                {
                    "metadata": {"name": "job1", "creationTimestamp": "2026-01-12T15:30:00Z"},
                    "status": {"applicationState": {"state": "COMPLETED"}},
                },
                {
                    "metadata": {"name": "job2", "creationTimestamp": "2026-01-12T15:35:00Z"},
                    "status": {"applicationState": {"state": "RUNNING"}},
                },
            ]
        }
        spark_service.client.list_jobs.return_value = mock_response

        result = spark_service.list_jobs()

        assert len(result) == 2
        assert result[0]["job_name"] == "job1"
        assert result[0]["state"] == "COMPLETED"
        assert result[0]["created_at"] == "2026-01-12T15:30:00Z"
        assert result[1]["job_name"] == "job2"
        assert result[1]["state"] == "RUNNING"
        assert result[1]["created_at"] == "2026-01-12T15:35:00Z"

    def test_list_jobs_handles_empty_list(self, spark_service: SparkService) -> None:
        """Test that list_jobs handles empty job list.

        **Why this test is important:**
          - Empty list is valid response
          - No errors should occur
          - Critical for robustness
          - Validates edge case handling

        **What it tests:**
          - Empty items list returns empty list
          - No errors are raised
        """
        mock_response = {"items": []}
        spark_service.client.list_jobs.return_value = mock_response

        result = spark_service.list_jobs()

        assert result == []

    def test_list_jobs_handles_missing_status(self, spark_service: SparkService) -> None:
        """Test that list_jobs handles jobs with missing status.

        **Why this test is important:**
          - Jobs may not have status yet
          - Graceful handling prevents crashes
          - Critical for robustness
          - Validates defensive programming

        **What it tests:**
          - Missing status returns UNKNOWN
          - No KeyError is raised
        """
        mock_response = {
            "items": [
                {
                    "metadata": {"name": "job1", "creationTimestamp": "2026-01-12T15:30:00Z"},
                }
            ]
        }
        spark_service.client.list_jobs.return_value = mock_response

        result = spark_service.list_jobs()

        assert len(result) == 1
        assert result[0]["state"] == "UNKNOWN"


# =============================================================================
# Job Deletion Tests
# =============================================================================


class TestSparkServiceDeleteJob:
    """Test suite for SparkService.delete_job method."""

    def test_delete_job_success(self, spark_service: SparkService) -> None:
        """Test that delete_job deletes and returns confirmation.

        **Why this test is important:**
          - Job deletion is needed for cleanup
          - Validates client interaction
          - Critical for resource management
          - Validates response format

        **What it tests:**
          - Client delete_job is called with correct name
          - Response includes job name and status
          - Confirmation status is "deleted"
        """
        result = spark_service.delete_job("test-job")

        assert result["job_name"] == "test-job"
        assert result["status"] == "deleted"
        spark_service.client.delete_job.assert_called_once_with("test-job")

    def test_delete_job_propagates_client_error(self, spark_service: SparkService) -> None:
        """Test that delete_job propagates client errors.

        **Why this test is important:**
          - Deletion can fail for various reasons
          - Errors should propagate to caller
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - RuntimeError from client is propagated
          - Error message is preserved
        """
        spark_service.client.delete_job.side_effect = RuntimeError("Job not found")

        with pytest.raises(RuntimeError, match="Job not found"):
            spark_service.delete_job("nonexistent-job")


# =============================================================================
# Job Name Generation Tests
# =============================================================================


class TestSparkServiceJobNameGeneration:
    """Test suite for job name generation."""

    def test_generated_names_are_unique(self, spark_service: SparkService) -> None:
        """Test that generated job names are unique.

        **Why this test is important:**
          - Unique names prevent job conflicts
          - Concurrent submissions must not collide
          - Critical for reliability
          - Validates uniqueness guarantee

        **What it tests:**
          - Multiple submissions generate different names
          - Names include timestamp and UUID
          - Name format is consistent
        """
        result1 = spark_service.submit_processing_job(s3_prefix="inputs/", collection="docs")
        result2 = spark_service.submit_processing_job(s3_prefix="inputs/", collection="docs")

        name1 = result1["job_name"]
        name2 = result2["job_name"]

        assert name1 != name2
        assert name1.startswith("s3-to-vector-db-")
        assert name2.startswith("s3-to-vector-db-")

    def test_generated_names_include_timestamp(self, spark_service: SparkService) -> None:
        """Test that generated names include timestamp.

        **Why this test is important:**
          - Timestamps enable chronological ordering
          - Helps with debugging and monitoring
          - Critical for job tracking
          - Validates timestamp inclusion

        **What it tests:**
          - Job name includes timestamp in YYYYMMDD-HHMMSS format
          - Timestamp is recent (within 60 seconds)
        """
        result = spark_service.submit_processing_job(s3_prefix="inputs/", collection="docs")
        job_name = result["job_name"]

        # Extract timestamp portion: s3-to-vector-db-20260112-153045-uuid
        # Splits to: ['s3', 'to', 'vector', 'db', '20260112', '153045', 'uuid']
        parts = job_name.split("-")
        assert len(parts) >= 7
        date_part = parts[4]  # YYYYMMDD
        time_part = parts[5]  # HHMMSS

        # Validate format
        assert len(date_part) == 8
        assert len(time_part) == 6
        assert date_part.isdigit()
        assert time_part.isdigit()

        # Validate timestamp is recent (within 60 seconds)
        timestamp_str = (
            f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} "
            f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        )
        job_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        time_diff = abs((now - job_timestamp).total_seconds())
        assert time_diff < 60  # Within 60 seconds


# =============================================================================
# Integration Tests
# =============================================================================


class TestSparkServiceIntegration:
    """Test suite for end-to-end service integration."""

    def test_full_job_lifecycle(self, spark_service: SparkService) -> None:
        """Test complete job lifecycle: submit -> status -> delete.

        **Why this test is important:**
          - Validates end-to-end workflow
          - Ensures all operations work together
          - Critical for real-world usage
          - Validates integration

        **What it tests:**
          - Job can be submitted, monitored, and deleted
          - All operations complete successfully
          - Data flows correctly between operations
        """
        # Submit job
        submit_result = spark_service.submit_processing_job(
            s3_prefix="inputs/",
            collection="documents",
            job_name="integration-test-job",
        )
        assert submit_result["job_name"] == "integration-test-job"
        assert submit_result["status"] == "submitted"

        # Mock status response
        spark_service.client.get_job_status.return_value = {
            "status": {
                "applicationState": {"state": "RUNNING"},
                "sparkApplicationState": "RUNNING",
                "driverInfo": {},
                "executionAttempts": 1,
            }
        }

        # Get status
        status_result = spark_service.get_job_status("integration-test-job")
        assert status_result["job_name"] == "integration-test-job"
        assert status_result["state"] == "RUNNING"

        # Delete job
        delete_result = spark_service.delete_job("integration-test-job")
        assert delete_result["job_name"] == "integration-test-job"
        assert delete_result["status"] == "deleted"
