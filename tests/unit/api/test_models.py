"""Unit tests for API Pydantic models.

This file tests all Pydantic request and response models used by the FastAPI endpoints.
All models use Pydantic for automatic validation, serialization, and OpenAPI schema generation.

# Test Coverage

The tests cover:
  - Request model validation (field types, constraints, required fields)
  - Response model serialization (field inclusion, defaults)
  - Model field constraints (min_length, ge, le, pattern)
  - Optional vs required fields
  - Model defaults and factories

# Test Structure

Tests use pytest class-based organization for grouping related tests.

# Running Tests

Run with: pytest tests/unit/api/test_models.py
"""

import pytest
from pydantic import ValidationError
from api import models


# =============================================================================
# Ray Job Models Tests
# =============================================================================


class TestRayJobStatusResponse:
    """Test suite for RayJobStatusResponse model."""

    def test_valid_status_without_message(self) -> None:
        """Test that RayJobStatusResponse handles successful status.

        **Why this test is important:**
          - Validates optional message field
          - Tests success case
        """
        resp = models.RayJobStatusResponse(
            job_id="raysubmit_123",
            status="RUNNING",
        )

        assert resp.job_id == "raysubmit_123"
        assert resp.status == "RUNNING"
        assert resp.message is None

    def test_valid_status_with_error_message(self) -> None:
        """Test that RayJobStatusResponse handles failure status.

        **Why this test is important:**
          - Validates message field with error
          - Tests failure case
        """
        resp = models.RayJobStatusResponse(
            job_id="raysubmit_123",
            status="FAILED",
            message="Job failed due to timeout",
        )

        assert resp.status == "FAILED"
        assert resp.message == "Job failed due to timeout"


# =============================================================================
# Ray Image Job Models Tests
# =============================================================================


class TestRayImageJobRequest:
    """Test suite for RayImageJobRequest model."""

    def test_valid_request_with_defaults(self) -> None:
        """Test that RayImageJobRequest works with default s3_prefix and no image_max_items."""
        req = models.RayImageJobRequest(
            s3_bucket="pipeline",
            collection="documents",
        )

        assert req.s3_bucket == "pipeline"
        assert req.s3_prefix == ""
        assert req.collection == "documents"
        assert req.image_max_items is None
        assert req.image_page_size is None

    def test_empty_prefix_means_bucket_root(self) -> None:
        """Test that s3_prefix defaults to empty string (bucket root)."""
        req = models.RayImageJobRequest(
            s3_bucket="pipeline",
            collection="documents",
        )
        assert req.s3_prefix == ""

    def test_explicit_prefix(self) -> None:
        """Test that an explicit s3_prefix is accepted."""
        req = models.RayImageJobRequest(
            s3_bucket="pipeline",
            s3_prefix="images/2026/",
            collection="documents",
        )
        assert req.s3_prefix == "images/2026/"

    def test_image_max_items_positive(self) -> None:
        """Test that image_max_items accepts positive integers."""
        req = models.RayImageJobRequest(
            s3_bucket="pipeline",
            collection="documents",
            image_max_items=500,
        )
        assert req.image_max_items == 500

    def test_image_max_items_rejects_zero(self) -> None:
        """Test that image_max_items rejects zero (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            models.RayImageJobRequest(
                s3_bucket="pipeline",
                collection="documents",
                image_max_items=0,
            )

    def test_image_max_items_rejects_negative(self) -> None:
        """Test that image_max_items rejects negative values."""
        with pytest.raises(ValidationError):
            models.RayImageJobRequest(
                s3_bucket="pipeline",
                collection="documents",
                image_max_items=-1,
            )

    def test_image_page_size_positive(self) -> None:
        """Test that image_page_size accepts positive integers."""
        req = models.RayImageJobRequest(
            s3_bucket="pipeline",
            collection="documents",
            image_page_size=500,
        )
        assert req.image_page_size == 500

    def test_image_page_size_rejects_zero(self) -> None:
        """Test that image_page_size rejects zero (gt=0 constraint)."""
        with pytest.raises(ValidationError):
            models.RayImageJobRequest(
                s3_bucket="pipeline",
                collection="documents",
                image_page_size=0,
            )

    def test_image_page_size_rejects_negative(self) -> None:
        """Test that image_page_size rejects negative values."""
        with pytest.raises(ValidationError):
            models.RayImageJobRequest(
                s3_bucket="pipeline",
                collection="documents",
                image_page_size=-1,
            )


class TestRayImageJobResponse:
    """Test suite for RayImageJobResponse model."""

    def test_response_includes_image_max_items(self) -> None:
        """Test that RayImageJobResponse includes image_max_items when provided."""
        resp = models.RayImageJobResponse(
            job_id="raysubmit_123",
            status="submitted",
            namespace="ml-system",
            s3_bucket="pipeline",
            s3_prefix="images/",
            collection="documents",
            image_max_items=100,
            submitted_at="2026-01-12T15:30:45Z",
        )
        assert resp.image_max_items == 100

    def test_response_image_max_items_defaults_none(self) -> None:
        """Test that RayImageJobResponse defaults image_max_items to None."""
        resp = models.RayImageJobResponse(
            job_id="raysubmit_123",
            status="submitted",
            namespace="ml-system",
            s3_bucket="pipeline",
            s3_prefix="",
            collection="documents",
            submitted_at="2026-01-12T15:30:45Z",
        )
        assert resp.image_max_items is None


# =============================================================================
# Databricks Image Job Models Tests
# =============================================================================


class TestDatabricksImageJobRequest:
    """Test suite for DatabricksImageJobRequest model."""

    def test_s3_prefix_defaults_empty_string(self) -> None:
        """Test that s3_prefix defaults to bucket root when omitted."""
        req = models.DatabricksImageJobRequest(
            source="s3",
            collection="documents",
        )
        assert req.s3_prefix == ""

    def test_image_max_items_field(self) -> None:
        """Test that DatabricksImageJobRequest accepts image_max_items."""
        req = models.DatabricksImageJobRequest(
            source="s3",
            s3_prefix="images/",
            collection="documents",
            image_max_items=200,
        )
        assert req.image_max_items == 200

    def test_image_max_items_defaults_none(self) -> None:
        """Test that image_max_items defaults to None."""
        req = models.DatabricksImageJobRequest(
            source="s3",
            s3_prefix="images/",
            collection="documents",
        )
        assert req.image_max_items is None
        assert req.image_page_size is None

    def test_image_max_items_rejects_zero(self) -> None:
        """Test that image_max_items rejects zero."""
        with pytest.raises(ValidationError):
            models.DatabricksImageJobRequest(
                source="s3",
                s3_prefix="images/",
                collection="documents",
                image_max_items=0,
            )

    def test_image_page_size_positive(self) -> None:
        """Test that image_page_size accepts positive integers."""
        req = models.DatabricksImageJobRequest(
            source="s3",
            s3_prefix="images/",
            collection="documents",
            image_page_size=2000,
        )
        assert req.image_page_size == 2000

    def test_image_page_size_rejects_zero(self) -> None:
        """Test that image_page_size rejects zero."""
        with pytest.raises(ValidationError):
            models.DatabricksImageJobRequest(
                source="s3",
                s3_prefix="images/",
                collection="documents",
                image_page_size=0,
            )


class TestDatabricksCdcJobResponses:
    """Test suite for Databricks CDC job response models."""

    def test_cdc_producer_response_fields(self) -> None:
        """Producer response should expose run metadata fields."""
        resp = models.DatabricksCdcProducerJobResponse(
            run_id="123",
            status="submitted",
            namespace="ml-system",
            submitted_at="2026-03-09T10:59:54Z",
        )
        assert resp.run_id == "123"
        assert resp.status == "submitted"
        assert resp.namespace == "ml-system"

    def test_cdc_consumer_response_fields(self) -> None:
        """Consumer response should expose run metadata fields."""
        resp = models.DatabricksCdcConsumerJobResponse(
            run_id="456",
            status="submitted",
            namespace="ml-system",
            submitted_at="2026-03-09T11:05:00Z",
        )
        assert resp.run_id == "456"
        assert resp.status == "submitted"
        assert resp.namespace == "ml-system"
