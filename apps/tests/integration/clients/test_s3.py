"""Integration tests for S3ClientWrapper using real MinIO container.

This module tests the S3ClientWrapper against a real MinIO instance to validate:
1. Basic CRUD operations work correctly
2. Retry logic handles transient failures
3. Circuit breaker opens/closes appropriately
4. Error classification distinguishes retriable from non-retriable errors
5. Resource cleanup happens properly

## Test Categories

Following the integration test strategy, we cover:

1. **Happy Path**: Baseline correctness with valid operations
2. **Transient Failure → Retry Succeeds**: Retry logic for temporary errors
3. **Retry Exhaustion → Proper Failure**: All retries fail
4. **Non-Retriable Errors (Fail Fast)**: No retries for 4xx errors
5. **Circuit Breaker Opens**: Threshold-based circuit opening
6. **Circuit Breaker Recovery**: Half-open to closed transition
7. **Rate Limiting**: (Not applicable - boto3 handles internally)
8. **Timeout Handling**: Slow operations trigger timeouts
9. **Resource Cleanup**: No leaked connections
10. **Observability & Logging**: Errors are logged correctly

## Running Tests

```bash
# Run only S3 integration tests
pytest tests/integration/clients/test_s3.py -v

# Run with logging output
pytest tests/integration/clients/test_s3.py -v --log-cli-level=INFO
```
"""

import asyncio
import contextlib
import logging
import time
from unittest.mock import MagicMock, patch

import pybreaker
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

from clients.s3 import S3ClientWrapper, _is_retriable_error
from core.exceptions import UpstreamError
from foundation.circuit_breaker import create_circuit_breaker

logger = logging.getLogger(__name__)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# 1. Happy Path Tests (Baseline Correctness)
# =============================================================================


class TestHappyPath:
    """Test baseline correctness with valid operations."""

    def test_put_and_get_object(
        self, minio_client: S3ClientWrapper, test_bucket: str, unique_key: str, sample_data: bytes
    ) -> None:
        """Test basic put and get operations work correctly.

        **Why this test is important:**
          - Validates the fundamental S3 operations work end-to-end
          - Ensures data integrity is maintained through upload/download cycle
          - Establishes baseline correctness before testing failure scenarios

        **What it tests:**
          - Object can be uploaded via put_object
          - Object can be retrieved via get_object
          - Data integrity is maintained (bytes in == bytes out)
        """
        # Put object
        minio_client.put_object(bucket=test_bucket, key=unique_key, body=sample_data)

        # Get object
        result = minio_client.get_object(bucket=test_bucket, key=unique_key)

        # Verify data integrity
        assert result == sample_data

    def test_list_objects(self, minio_client: S3ClientWrapper, test_bucket: str, sample_data: bytes) -> None:
        """Test listing objects in a bucket.

        **Why this test is important:**
          - List operations are critical for batch processing and discovery
          - Prefix filtering is essential for organizing objects by path
          - Ensures the client correctly handles multi-object responses

        **What it tests:**
          - Multiple objects can be listed with list_objects
          - Prefix filtering returns only matching keys
          - All created objects are returned in the list
        """
        # Create multiple objects
        keys = [f"test-prefix/{i}.txt" for i in range(5)]
        for key in keys:
            minio_client.put_object(bucket=test_bucket, key=key, body=sample_data)

        # List all with prefix
        result = minio_client.list_objects(bucket=test_bucket, prefix="test-prefix/")

        assert len(result) == 5
        assert set(result) == set(keys)

    def test_exists_returns_true_for_existing_object(
        self, minio_client: S3ClientWrapper, test_bucket: str, unique_key: str, sample_data: bytes
    ) -> None:
        """Test exists() returns True for existing objects.

        **Why this test is important:**
          - Existence checks are used to avoid duplicate processing
          - Critical for idempotent operations and checkpointing
          - False negatives would cause unnecessary reprocessing

        **What it tests:**
          - exists() returns True for an object that was just uploaded
          - No false negatives occur for existing objects
        """
        minio_client.put_object(bucket=test_bucket, key=unique_key, body=sample_data)

        assert minio_client.exists(bucket=test_bucket, key=unique_key) is True

    def test_exists_returns_false_for_missing_object(
        self, minio_client: S3ClientWrapper, test_bucket: str
    ) -> None:
        """Test exists() returns False for non-existent objects.

        **Why this test is important:**
          - 404 errors must be handled gracefully, not raised as exceptions
          - False positives would cause skipped processing of new objects
          - Validates the exists() method correctly interprets S3 404 responses

        **What it tests:**
          - exists() returns False for a key that doesn't exist
          - 404 ClientError is caught and converted to False
        """
        assert minio_client.exists(bucket=test_bucket, key="nonexistent-key") is False

    def test_delete_object(
        self, minio_client: S3ClientWrapper, test_bucket: str, unique_key: str, sample_data: bytes
    ) -> None:
        """Test object deletion.

        **Why this test is important:**
          - Deletion is required for cleanup and data lifecycle management
          - Ensures deleted objects are actually removed from storage
          - Validates delete_object integrates correctly with exists()

        **What it tests:**
          - delete_object removes an existing object
          - exists() returns False after deletion
        """
        minio_client.put_object(bucket=test_bucket, key=unique_key, body=sample_data)
        assert minio_client.exists(bucket=test_bucket, key=unique_key) is True

        minio_client.delete_object(bucket=test_bucket, key=unique_key)

        assert minio_client.exists(bucket=test_bucket, key=unique_key) is False

    @pytest.mark.asyncio
    async def test_get_object_async(
        self, minio_client: S3ClientWrapper, test_bucket: str, unique_key: str, sample_data: bytes
    ) -> None:
        """Test async object retrieval.

        **Why this test is important:**
          - Async operations are used in Ray jobs and concurrent processing
          - Ensures the async wrapper correctly delegates to sync implementation
          - Validates data integrity through the async code path

        **What it tests:**
          - get_object_async retrieves data correctly
          - Data integrity is maintained in async operations
        """
        minio_client.put_object(bucket=test_bucket, key=unique_key, body=sample_data)

        result = await minio_client.get_object_async(bucket=test_bucket, key=unique_key)

        assert result == sample_data


# =============================================================================
# 2. Transient Failure → Retry Succeeds
# =============================================================================


class TestRetrySuccess:
    """Test that transient failures trigger retries that eventually succeed."""

    def test_retry_succeeds_after_connection_error(
        self, minio_config: dict[str, str], test_bucket: str, sample_data: bytes
    ) -> None:
        """Test that connection errors trigger retries.

        **Why this test is important:**
          - Connection errors are common in distributed systems
          - Retry logic is essential for resilience against transient network issues
          - Ensures the client doesn't fail on first transient error

        **What it tests:**
          - EndpointConnectionError triggers retry behavior
          - Operation succeeds after transient failures resolve
          - Correct number of retry attempts (3 in this case)
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
            max_retries=3,
            retry_min_wait=0.01,  # Fast for tests
            retry_max_wait=0.1,
        )

        # Ensure bucket exists
        client.ensure_bucket(test_bucket)

        call_count = 0
        original_put = client._client.put_object

        def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate connection error on first 2 attempts
                raise EndpointConnectionError(endpoint_url="http://test")
            return original_put(*args, **kwargs)

        with patch.object(client._client, "put_object", side_effect=failing_then_succeeding):
            client.put_object(bucket=test_bucket, key="retry-test", body=sample_data)

        # Should have retried twice before succeeding
        assert call_count == 3

        # Verify data was actually written
        result = client.get_object(bucket=test_bucket, key="retry-test")
        assert result == sample_data

    def test_retry_succeeds_after_500_error(
        self, minio_config: dict[str, str], test_bucket: str, sample_data: bytes
    ) -> None:
        """Test that 500 errors trigger retries.

        **Why this test is important:**
          - Server errors (5xx) indicate transient service issues
          - These should be retried as the service may recover quickly
          - Validates error classification for HTTP status codes

        **What it tests:**
          - HTTP 500 InternalError triggers retry behavior
          - Operation succeeds when service recovers
          - Error classification correctly identifies 5xx as retriable
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
            max_retries=3,
            retry_min_wait=0.01,
            retry_max_wait=0.1,
        )

        client.ensure_bucket(test_bucket)

        call_count = 0
        original_get = client._client.get_object

        def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ClientError(
                    {
                        "Error": {"Code": "InternalError", "Message": "Internal Error"},
                        "ResponseMetadata": {"HTTPStatusCode": 500},
                    },
                    "GetObject",
                )
            return original_get(*args, **kwargs)

        # First put the object
        client.put_object(bucket=test_bucket, key="server-error-test", body=sample_data)

        # Then test get with simulated failures
        with patch.object(client._client, "get_object", side_effect=failing_then_succeeding):
            result = client.get_object(bucket=test_bucket, key="server-error-test")

        assert result == sample_data
        assert call_count == 2


# =============================================================================
# 3. Retry Exhaustion → Proper Failure
# =============================================================================


class TestRetryExhaustion:
    """Test behavior when all retries are exhausted."""

    def test_raises_upstream_error_after_max_retries(
        self, minio_config: dict[str, str], test_bucket: str
    ) -> None:
        """Test that UpstreamError is raised when retries are exhausted.

        **Why this test is important:**
          - Prevents infinite retry loops that would hang jobs
          - Ensures proper error propagation for caller handling
          - Validates the retry count limit is respected

        **What it tests:**
          - UpstreamError is raised after max_retries attempts
          - Error message includes attempt count for debugging
          - Exactly max_retries attempts are made (no more)
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
            max_retries=2,  # Low for fast test
            retry_min_wait=0.01,
            retry_max_wait=0.05,
        )

        client.ensure_bucket(test_bucket)

        call_count = 0

        def always_fail(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise EndpointConnectionError(endpoint_url="http://test")

        with (
            patch.object(client._client, "get_object", side_effect=always_fail),
            pytest.raises(UpstreamError, match="failed after 2 attempts"),
        ):
            client.get_object(bucket=test_bucket, key="exhaustion-test")

        # Should have tried max_retries times
        assert call_count == 2


# =============================================================================
# 4. Non-Retriable Errors (Fail Fast)
# =============================================================================


class TestNonRetriableErrors:
    """Test that non-retriable errors fail immediately without retries."""

    def test_access_denied_not_retried(self, minio_config: dict[str, str], test_bucket: str) -> None:
        """Test that AccessDenied (403) is not retried.

        **Why this test is important:**
          - Authorization errors won't resolve with retries
          - Retrying 403s wastes time and resources
          - Fast failure allows callers to handle auth issues promptly

        **What it tests:**
          - AccessDenied (403) is classified as non-retriable
          - _is_retriable_error returns False for 403 status
        """
        # Test that AccessDenied is classified as non-retriable
        access_denied_error = ClientError(
            {
                "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            "GetObject",
        )

        # This should NOT be retriable
        assert _is_retriable_error(access_denied_error) is False

    def test_invalid_bucket_not_retried(self, minio_config: dict[str, str]) -> None:
        """Test that NoSuchBucket is not retried.

        **Why this test is important:**
          - Missing buckets won't appear with retries
          - 404 errors indicate configuration issues, not transient failures
          - Fast failure enables quick diagnosis of bucket misconfiguration

        **What it tests:**
          - NoSuchBucket (404) is classified as non-retriable
          - _is_retriable_error returns False for 404 status
        """
        # Test that NoSuchBucket is classified as non-retriable
        no_such_bucket_error = ClientError(
            {
                "Error": {"Code": "NoSuchBucket", "Message": "The specified bucket does not exist"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            "GetObject",
        )

        # This should NOT be retriable
        assert _is_retriable_error(no_such_bucket_error) is False

    def test_transient_error_is_retriable(self) -> None:
        """Test that transient server errors are classified as retriable.

        **Why this test is important:**
          - Correct error classification is fundamental to retry logic
          - Ensures transient errors trigger retries while permanent errors don't
          - Validates the error classification function works correctly

        **What it tests:**
          - HTTP 500 InternalError is classified as retriable
          - EndpointConnectionError is classified as retriable
          - _is_retriable_error returns True for both cases
        """
        # Internal server error should be retriable
        internal_error = ClientError(
            {
                "Error": {"Code": "InternalError", "Message": "Internal Error"},
                "ResponseMetadata": {"HTTPStatusCode": 500},
            },
            "GetObject",
        )
        assert _is_retriable_error(internal_error) is True

        # Connection error should be retriable
        connection_error = EndpointConnectionError(endpoint_url="http://test")
        assert _is_retriable_error(connection_error) is True


# =============================================================================
# 5. Circuit Breaker Opens After Threshold
# =============================================================================


class TestCircuitBreakerOpens:
    """Test that circuit breaker opens after repeated failures."""

    def test_circuit_opens_after_threshold(self, minio_config: dict[str, str], test_bucket: str) -> None:
        """Test circuit breaker opens after failure threshold.

        **Why this test is important:**
          - Circuit breakers prevent cascading failures to downstream services
          - Open circuit provides fail-fast behavior, reducing latency
          - Protects the system when S3/MinIO is experiencing issues

        **What it tests:**
          - Circuit transitions from CLOSED to OPEN after 5 failures
          - Subsequent calls fail fast without making network requests
          - No additional boto3 calls are made when circuit is open
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
            max_retries=1,  # Minimize retries
            retry_min_wait=0.001,
            retry_max_wait=0.01,
        )

        client.ensure_bucket(test_bucket)

        # Circuit breaker threshold is 5 failures
        failure_count = 0

        def always_fail(*_args, **_kwargs):
            nonlocal failure_count
            failure_count += 1
            raise EndpointConnectionError(endpoint_url="http://test")

        with patch.object(client._client, "get_object", side_effect=always_fail):
            # Trigger failures up to threshold
            for i in range(5):
                with contextlib.suppress(UpstreamError):
                    client.get_object(bucket=test_bucket, key=f"cb-test-{i}")

        # Circuit should now be OPEN
        assert client._breaker.current_state == pybreaker.STATE_OPEN

        # Next call should fail fast without making actual request
        pre_fail_count = failure_count
        with pytest.raises(UpstreamError, match="currently unavailable"):
            client.get_object(bucket=test_bucket, key="cb-fail-fast")

        # No additional calls should have been made
        assert failure_count == pre_fail_count


# =============================================================================
# 6. Circuit Breaker Recovery (Half-Open → Closed)
# =============================================================================


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery behavior."""

    def test_circuit_closes_after_successful_operations(
        self, minio_client: S3ClientWrapper, test_bucket: str, sample_data: bytes
    ) -> None:
        """Test that successful operations keep circuit closed.

        **Why this test is important:**
          - Successful operations should reset failure counters
          - Circuit should remain closed during normal healthy operation
          - Validates the circuit breaker doesn't open spuriously

        **What it tests:**
          - Circuit remains in CLOSED state after successful operations
          - fail_counter is 0 after successful operations
        """
        # Put some data
        minio_client.put_object(bucket=test_bucket, key="recovery-test", body=sample_data)

        # Successful operations keep circuit closed
        result = minio_client.get_object(bucket=test_bucket, key="recovery-test")
        assert result == sample_data

        # Circuit should still be closed
        assert minio_client._breaker.current_state == pybreaker.STATE_CLOSED
        assert minio_client._breaker.fail_counter == 0

    def test_circuit_breaker_state_transitions(self) -> None:
        """Test circuit breaker state transition logic.

        **Why this test is important:**
          - State transitions are the core of circuit breaker behavior
          - HALF_OPEN state enables graceful recovery testing
          - Validates the pybreaker library works as expected

        **What it tests:**
          - Circuit starts in CLOSED state
          - Transitions to OPEN after failure_threshold failures
          - Transitions to HALF_OPEN after recovery_timeout
          - Transitions back to CLOSED on successful call in HALF_OPEN
        """
        breaker = create_circuit_breaker(
            name="test-recovery",
            failure_threshold=2,
            recovery_timeout=0,  # Immediate recovery for testing
        )

        # Initially closed
        assert breaker.current_state == pybreaker.STATE_CLOSED

        # Fail twice to open circuit
        for _ in range(2):
            with contextlib.suppress(Exception):
                breaker.call(lambda: (_ for _ in ()).throw(Exception("test")))

        # Now open
        assert breaker.current_state == pybreaker.STATE_OPEN

        # Wait for recovery timeout (0 seconds)
        time.sleep(0.1)

        # After recovery timeout, next call attempt triggers HALF_OPEN
        # Success closes it, failure reopens it
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.current_state == pybreaker.STATE_CLOSED


# =============================================================================
# 8. Timeout Handling
# =============================================================================


class TestTimeoutHandling:
    """Test timeout behavior for slow operations."""

    def test_slow_operation_triggers_timeout(self, minio_config: dict[str, str]) -> None:
        """Test that slow operations trigger timeouts.

        **Why this test is important:**
          - Timeouts prevent indefinite hangs on slow/stalled connections
          - Critical for maintaining job throughput and responsiveness
          - Ensures resources aren't held indefinitely

        **What it tests:**
          - ReadTimeoutError is raised for slow operations
          - UpstreamError is raised after retries are exhausted
          - Client doesn't hang waiting for response
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
            max_retries=1,
            timeout_s=1,  # Very short timeout
            retry_min_wait=0.01,
            retry_max_wait=0.1,
        )

        def timeout_error(*_args, **_kwargs):
            raise ReadTimeoutError(endpoint_url="http://test")

        with (
            patch.object(client._client, "get_object", side_effect=timeout_error),
            pytest.raises(UpstreamError, match="failed"),
        ):
            client.get_object(bucket="test", key="timeout-test")


# =============================================================================
# 9. Resource Cleanup (No Leaks)
# =============================================================================


class TestResourceCleanup:
    """Test that resources are properly cleaned up."""

    def test_client_close_releases_resources(self, minio_config: dict[str, str]) -> None:
        """Test that close() releases client resources.

        **Why this test is important:**
          - Resource leaks cause memory issues in long-running jobs
          - Proper cleanup is essential for Ray/Spark worker lifecycle
          - Validates the close() method works correctly

        **What it tests:**
          - close() can be called on a client
          - _client reference is set to None after close
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
        )

        # Verify client is usable
        assert client._client is not None

        # Close client
        client.close()

        # Client reference should be cleared
        assert client._client is None

    @pytest.mark.asyncio
    async def test_concurrent_async_operations_cleanup(
        self, minio_client: S3ClientWrapper, test_bucket: str, sample_data: bytes
    ) -> None:
        """Test that concurrent async operations don't leak resources.

        **Why this test is important:**
          - Concurrent operations are common in parallel processing
          - Resource leaks under concurrency can cause pool exhaustion
          - Validates async operations work correctly with asyncio.gather

        **What it tests:**
          - Multiple concurrent get_object_async calls succeed
          - All results are returned correctly
          - No exceptions from resource contention
        """
        keys = [f"concurrent-{i}" for i in range(10)]

        # Upload objects
        for key in keys:
            minio_client.put_object(bucket=test_bucket, key=key, body=sample_data)

        # Fetch all concurrently
        tasks = [minio_client.get_object_async(bucket=test_bucket, key=key) for key in keys]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r == sample_data for r in results)


# =============================================================================
# 10. Observability & Logging
# =============================================================================


class TestObservability:
    """Test that errors are logged correctly for observability."""

    def test_retry_attempts_are_logged(self, minio_config: dict[str, str], test_bucket: str, caplog) -> None:
        """Test that retry attempts are logged.

        **Why this test is important:**
          - Observability is critical for debugging production issues
          - Retry logs help identify transient vs persistent failures
          - Enables operators to understand system behavior under stress

        **What it tests:**
          - Retry attempts produce WARNING level log messages
          - Log messages include "retrying" keyword for filtering
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
            max_retries=2,
            retry_min_wait=0.01,
            retry_max_wait=0.05,
        )

        client.ensure_bucket(test_bucket)

        call_count = 0

        def fail_once(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise EndpointConnectionError(endpoint_url="http://test")
            return MagicMock(Body=MagicMock(read=lambda: b"data"))

        with (
            caplog.at_level(logging.WARNING, logger="clients.s3"),
            patch.object(client._client, "get_object", side_effect=fail_once),
        ):
            client.get_object(bucket=test_bucket, key="log-test")

        # Check retry was logged
        assert any("retrying" in record.message.lower() for record in caplog.records)

    def test_final_failure_is_logged(self, minio_config: dict[str, str], test_bucket: str, caplog) -> None:
        """Test that final failures are logged with context.

        **Why this test is important:**
          - Final failure logs are essential for post-mortem analysis
          - Must include enough context to diagnose the issue
          - Enables alerting and monitoring based on error patterns

        **What it tests:**
          - Final failure produces ERROR level log message
          - Log message includes "failed" keyword for alerting
        """
        client = S3ClientWrapper(
            endpoint_url=minio_config["endpoint_url"],
            access_key_id=minio_config["access_key_id"],
            secret_access_key=minio_config["secret_access_key"],
            max_retries=2,
            retry_min_wait=0.01,
            retry_max_wait=0.05,
        )

        client.ensure_bucket(test_bucket)

        def always_fail(*_args, **_kwargs):
            raise EndpointConnectionError(endpoint_url="http://test")

        with (
            caplog.at_level(logging.ERROR, logger="clients.s3"),
            patch.object(client._client, "get_object", side_effect=always_fail),
            pytest.raises(UpstreamError),
        ):
            client.get_object(bucket=test_bucket, key="log-fail-test")

        # Check failure was logged
        assert any("failed" in record.message.lower() for record in caplog.records)
