"""Unit tests for clients.s3 module.

This file tests the S3ClientWrapper class which provides object storage operations
via boto3 S3 client.

# Test Coverage

The tests cover:
  - Client Initialization: Default and custom configuration
  - Bucket Operations: ensure_bucket, bucket existence checking
  - Object Operations: put_object, get_object, list_objects, exists
  - Circuit Breaker Integration: Circuit breaker usage, error handling
  - Async Operations: Async object retrieval
  - Error Handling: UpstreamError on failures, circuit breaker errors

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The underlying boto3 S3 client and circuit breaker are mocked to isolate client logic.

# Running Tests

Run with: pytest tests/unit/clients/test_s3.py
"""

from unittest.mock import MagicMock, patch

import pybreaker
import pytest
from botocore.exceptions import ClientError

from clients.s3 import S3ClientWrapper
from core.exceptions import UpstreamError

# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestS3ClientWrapperInit:
    """Test suite for S3ClientWrapper initialization."""

    @patch("clients.s3.boto3.client")
    def test_creates_client_with_config(self, mock_boto3: MagicMock) -> None:
        """Test that client is created with configuration.

        **Why this test is important:**
          - Client initialization is the foundation for all operations
          - Ensures configuration is applied correctly
          - Validates that boto3 client is created with correct parameters
          - Critical for basic functionality

        **What it tests:**
          - boto3.client is called with correct parameters
          - Client attributes are set correctly
          - Circuit breaker is created
        """
        mock_boto3_client = MagicMock()
        mock_boto3.return_value = mock_boto3_client

        client = S3ClientWrapper(
            endpoint_url="http://minio.example.com:9000",
            access_key_id="test-key",
            secret_access_key="test-secret",
            region_name="us-east-1",
        )

        # Verify boto3.client was called with expected params
        mock_boto3.assert_called_once()
        call_kwargs = mock_boto3.call_args.kwargs
        assert call_kwargs["endpoint_url"] == "http://minio.example.com:9000"
        assert call_kwargs["aws_access_key_id"] == "test-key"
        assert call_kwargs["aws_secret_access_key"] == "test-secret"
        assert call_kwargs["region_name"] == "us-east-1"
        # Config is now also passed for timeout settings
        assert "config" in call_kwargs

        assert client.endpoint_url == "http://minio.example.com:9000"
        assert client.access_key_id == "test-key"
        assert client.secret_access_key == "test-secret"
        assert client.region_name == "us-east-1"
        assert client._client == mock_boto3_client

    def test_creates_circuit_breaker(self) -> None:
        """Test that circuit breaker is created during initialization.

        **Why this test is important:**
          - Circuit breaker provides fault tolerance
          - Ensures circuit breaker is configured with correct parameters
          - Critical for production reliability
          - Validates circuit breaker integration

        **What it tests:**
          - Circuit breaker is created with correct name
          - Failure threshold and recovery timeout are set correctly
        """
        client = S3ClientWrapper(
            endpoint_url="http://minio.example.com:9000",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )

        # Verify circuit breaker was created
        assert client._breaker is not None
        assert isinstance(client._breaker, pybreaker.CircuitBreaker)
        assert client._breaker.name == "s3"
        assert client._breaker.fail_max == 5
        assert client._breaker.reset_timeout == 120


# =============================================================================
# Bucket Operations Tests
# =============================================================================


class TestS3ClientWrapperBucket:
    """Test suite for S3ClientWrapper bucket operations."""

    def test_ensure_bucket_creates_if_missing(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that ensure_bucket creates bucket if missing.

        **Why this test is important:**
          - Bucket creation is essential for storage operations
          - Ensures buckets exist before use
          - Critical for dev convenience functions
          - Validates bucket creation logic

        **What it tests:**
          - head_bucket is called to check existence
          - create_bucket is called if bucket doesn't exist
        """
        mock_boto3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )

        s3_client.ensure_bucket("test-bucket")

        mock_boto3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_boto3_client.create_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_ensure_bucket_skips_if_exists(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that ensure_bucket skips creation if bucket exists.

        **Why this test is important:**
          - Idempotent operations prevent errors
          - Avoids unnecessary API calls
          - Critical for efficiency
          - Validates existence checking

        **What it tests:**
          - head_bucket is called to check existence
          - create_bucket is not called if bucket exists
        """
        mock_boto3_client.head_bucket.return_value = None

        s3_client.ensure_bucket("test-bucket")

        mock_boto3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_boto3_client.create_bucket.assert_not_called()


# =============================================================================
# Object Operations Tests
# =============================================================================


class TestS3ClientWrapperPutObject:
    """Test suite for S3ClientWrapper.put_object method."""

    def test_put_object_success(self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock) -> None:
        """Test that put_object succeeds on valid input.

        **Why this test is important:**
          - Object storage is the core functionality
          - Validates successful API interaction
          - Ensures data is stored correctly
          - Critical for basic functionality

        **What it tests:**
          - put_object is called with correct parameters
          - No exceptions are raised on success
        """
        mock_boto3_client.put_object.return_value = {}

        s3_client.put_object(bucket="test-bucket", key="test-key", body=b"test-data")

        mock_boto3_client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-key", Body=b"test-data"
        )

    def test_put_object_raises_upstream_error_on_client_error(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that put_object raises UpstreamError on ClientError.

        **Why this test is important:**
          - Error handling ensures consistent error types
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation and debugging
          - Validates error wrapping

        **What it tests:**
          - ClientError is wrapped in UpstreamError
          - Error message includes context
        """
        # AccessDenied is a non-retriable 4xx error, should fail immediately
        mock_boto3_client.put_object.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "PutObject",
        )

        with pytest.raises(UpstreamError, match="S3 put_object failed"):
            s3_client.put_object(bucket="test-bucket", key="test-key", body=b"test-data")

    def test_put_object_handles_circuit_breaker_error(
        self,
        s3_client: S3ClientWrapper,
    ) -> None:
        """Test that put_object handles circuit breaker errors.

        **Why this test is important:**
          - Circuit breaker errors need special handling
          - UpstreamError conversion ensures consistent error types
          - Critical for fault tolerance
          - Validates circuit breaker integration

        **What it tests:**
          - CircuitBreakerError is handled correctly by the decorator
          - UpstreamError is raised when circuit is open
        """
        # Replace the circuit breaker with a mock in OPEN state
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        mock_breaker.current_state = pybreaker.STATE_OPEN
        object.__setattr__(s3_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="s3 service is currently unavailable"):
            s3_client.put_object(bucket="test-bucket", key="test-key", body=b"test-data")


class TestS3ClientWrapperGetObject:
    """Test suite for S3ClientWrapper.get_object method."""

    def test_get_object_success(self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock) -> None:
        """Test that get_object returns object content on success.

        **Why this test is important:**
          - Object retrieval is essential functionality
          - Validates successful API interaction
          - Ensures data is retrieved correctly
          - Critical for basic functionality

        **What it tests:**
          - get_object is called with correct parameters
          - Response body is read and returned
        """
        mock_body = MagicMock()
        mock_body.read.return_value = b"test-data"
        mock_boto3_client.get_object.return_value = {"Body": mock_body}

        result = s3_client.get_object(bucket="test-bucket", key="test-key")

        assert result == b"test-data"
        mock_boto3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test-key")
        mock_body.read.assert_called_once()

    def test_get_object_raises_upstream_error_on_client_error(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that get_object raises UpstreamError on ClientError.

        **Why this test is important:**
          - Error handling ensures consistent error types
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation and debugging
          - Validates error wrapping

        **What it tests:**
          - ClientError is wrapped in UpstreamError
          - Error message includes context
        """
        # NoSuchKey is a non-retriable 404 error, should fail immediately
        mock_boto3_client.get_object.side_effect = ClientError(
            {
                "Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )

        with pytest.raises(UpstreamError, match="S3 get_object failed"):
            s3_client.get_object(bucket="test-bucket", key="test-key")

    def test_get_object_handles_circuit_breaker_error(
        self,
        s3_client: S3ClientWrapper,
    ) -> None:
        """Test that get_object handles circuit breaker errors.

        **Why this test is important:**
          - Circuit breaker errors need special handling
          - UpstreamError conversion ensures consistent error types
          - Critical for fault tolerance
          - Validates circuit breaker integration

        **What it tests:**
          - CircuitBreakerError is handled correctly by the decorator
          - UpstreamError is raised when circuit is open
        """
        # Replace the circuit breaker with a mock in OPEN state
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        mock_breaker.current_state = pybreaker.STATE_OPEN
        object.__setattr__(s3_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="s3 service is currently unavailable"):
            s3_client.get_object(bucket="test-bucket", key="test-key")


class TestS3ClientWrapperListObjects:
    """Test suite for S3ClientWrapper.list_objects method."""

    def test_list_objects_success(self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock) -> None:
        """Test that list_objects returns object keys on success.

        **Why this test is important:**
          - Object listing is essential for discovery
          - Validates successful API interaction
          - Ensures pagination is handled correctly
          - Critical for batch operations

        **What it tests:**
          - Paginator is created and used correctly
          - Object keys are collected from all pages
          - List of keys is returned
        """
        mock_paginator = MagicMock()
        mock_page1 = {"Contents": [{"Key": "key1"}, {"Key": "key2"}]}
        mock_page2 = {"Contents": [{"Key": "key3"}]}
        mock_paginator.paginate.return_value = [mock_page1, mock_page2]
        mock_boto3_client.get_paginator.return_value = mock_paginator

        result = s3_client.list_objects(bucket="test-bucket", prefix="test-prefix/")

        assert result == ["key1", "key2", "key3"]
        mock_boto3_client.get_paginator.assert_called_once_with("list_objects_v2")
        mock_paginator.paginate.assert_called_once_with(Bucket="test-bucket", Prefix="test-prefix/")

    def test_list_objects_with_empty_prefix(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that list_objects works with empty prefix.

        **Why this test is important:**
          - Empty prefix lists all objects
          - Validates default parameter handling
          - Critical for listing all objects in bucket
          - Validates parameter handling

        **What it tests:**
          - Empty prefix is handled correctly
          - All objects are returned
        """
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Contents": [{"Key": "key1"}]}]
        mock_boto3_client.get_paginator.return_value = mock_paginator

        result = s3_client.list_objects(bucket="test-bucket")

        assert result == ["key1"]
        mock_paginator.paginate.assert_called_once_with(Bucket="test-bucket", Prefix="")

    def test_list_objects_handles_empty_results(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that list_objects handles empty results.

        **Why this test is important:**
          - Empty buckets should return empty lists
          - Validates edge case handling
          - Critical for robustness
          - Validates empty result handling

        **What it tests:**
          - Empty Contents list returns empty list
          - Missing Contents key returns empty list
        """
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{}]  # No Contents key
        mock_boto3_client.get_paginator.return_value = mock_paginator

        result = s3_client.list_objects(bucket="test-bucket")

        assert result == []

    def test_list_objects_raises_upstream_error_on_client_error(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that list_objects raises UpstreamError on ClientError.

        **Why this test is important:**
          - Error handling ensures consistent error types
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation and debugging
          - Validates error wrapping

        **What it tests:**
          - ClientError is wrapped in UpstreamError
          - Error message includes context
        """
        # AccessDenied is a non-retriable 4xx error
        mock_boto3_client.get_paginator.side_effect = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "ListObjects",
        )

        with pytest.raises(UpstreamError, match="S3 list_objects failed"):
            s3_client.list_objects(bucket="test-bucket")

    def test_list_objects_handles_circuit_breaker_error(
        self,
        s3_client: S3ClientWrapper,
    ) -> None:
        """Test that list_objects handles circuit breaker errors.

        **Why this test is important:**
          - Circuit breaker errors need special handling
          - UpstreamError conversion ensures consistent error types
          - Critical for fault tolerance
          - Validates circuit breaker integration

        **What it tests:**
          - CircuitBreakerError is handled correctly by the decorator
          - UpstreamError is raised when circuit is open
        """
        # Replace the circuit breaker with a mock in OPEN state
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        mock_breaker.current_state = pybreaker.STATE_OPEN
        object.__setattr__(s3_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="s3 service is currently unavailable"):
            s3_client.list_objects(bucket="test-bucket")


class TestS3ClientWrapperExists:
    """Test suite for S3ClientWrapper.exists method."""

    def test_exists_returns_true_when_object_exists(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that exists returns True when object exists.

        **Why this test is important:**
          - Object existence checking is essential
          - Validates successful API interaction
          - Critical for conditional operations
          - Validates existence checking

        **What it tests:**
          - head_object is called with correct parameters
          - True is returned when object exists
        """
        mock_boto3_client.head_object.return_value = {}

        result = s3_client.exists(bucket="test-bucket", key="test-key")

        assert result is True
        mock_boto3_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="test-key")

    def test_exists_returns_false_on_404(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that exists returns False on 404 error.

        **Why this test is important:**
          - 404 errors indicate object doesn't exist
          - False return allows graceful handling
          - Critical for conditional logic
          - Validates error code handling

        **What it tests:**
          - 404 error code returns False
          - Other errors are re-raised
        """
        mock_boto3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )

        result = s3_client.exists(bucket="test-bucket", key="test-key")

        assert result is False

    def test_exists_raises_on_other_errors(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that exists raises on non-404 errors.

        **Why this test is important:**
          - Non-404 errors should be propagated
          - Allows proper error handling for access issues
          - Critical for error visibility
          - Validates error propagation

        **What it tests:**
          - Non-404 ClientError is re-raised
          - Error is not wrapped
        """
        mock_boto3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "HeadObject"
        )

        with pytest.raises(ClientError):
            s3_client.exists(bucket="test-bucket", key="test-key")


# =============================================================================
# Async Operations Tests
# =============================================================================


class TestS3ClientWrapperAsync:
    """Test suite for S3ClientWrapper async operations."""

    @pytest.mark.asyncio
    async def test_get_object_async_calls_sync_method(
        self, s3_client: S3ClientWrapper, mock_boto3_client: MagicMock
    ) -> None:
        """Test that get_object_async calls sync get_object.

        **Why this test is important:**
          - Async wrapper enables concurrent operations
          - Validates async method implementation
          - Critical for performance optimization
          - Validates async wrapper

        **What it tests:**
          - get_object_async calls get_object in executor
          - Result is returned correctly
        """
        mock_body = MagicMock()
        mock_body.read.return_value = b"test-data"
        mock_boto3_client.get_object.return_value = {"Body": mock_body}

        result = await s3_client.get_object_async(bucket="test-bucket", key="test-key")

        assert result == b"test-data"
        mock_boto3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test-key")
