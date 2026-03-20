"""S3 client wrapper class for object storage operations.

This module provides an S3 client wrapper class that encapsulates configuration
and provides methods for bucket and object operations. This replaces the functional
API with an object-oriented approach using attrs.

## Usage

```python
from clients.s3 import S3ClientWrapper

client = S3ClientWrapper(
    endpoint_url="http://minio.ml-system:9000",
    access_key_id="minioadmin",
    secret_access_key="minioadmin"
)
client.put_object(bucket="pipeline", key="data.txt", body=b"hello")
```

## Resilience Features

The client implements multi-layer resilience. We intentionally disable boto3's
built-in retry mechanism and implement our own via `tenacity` to provide consistent
behavior across all clients (S3, Qdrant, Ollama) and integrate with our
circuit breaker pattern.

1. **Retry with Exponential Backoff**: Transient errors (5xx, timeouts, connection
   errors) trigger automatic retries with exponential backoff.

2. **Circuit Breaker**: After repeated failures, the circuit opens to fail fast
   and prevent cascade failures.

3. **Fail-Fast for Non-Retriable Errors**: Since we disabled boto3's retry mechanism,
   we implement our own error classification. Client errors (4xx) like 403 Forbidden
   or 404 Not Found are NOT retried.

## Error Classification

Since boto3 retries are disabled, we classify errors ourselves:

**Retriable Errors** (trigger retry + circuit breaker):
- Connection errors (EndpointConnectionError, ConnectionClosedError)
- Timeouts (ReadTimeoutError, ConnectTimeoutError)
- Server errors (5xx status codes: InternalError, ServiceUnavailable)
- Throttling (SlowDown, RequestLimitExceeded)

**Non-Retriable Errors** (fail immediately, no retry):
- Authentication failures (InvalidAccessKeyId, SignatureDoesNotMatch)
- Authorization failures (AccessDenied, AllAccessDisabled)
- Not found (NoSuchBucket, NoSuchKey)
- Invalid requests (InvalidBucketName, MalformedPolicy)

## Design

The client wrapper:
- Encapsulates S3 connection configuration
- Provides a clean interface for bucket and object operations
- Handles errors consistently with proper classification
- Uses attrs for concise, correct class definition
"""

import asyncio
import logging
from collections.abc import Generator
from typing import Any, cast

import attrs
import boto3  # type: ignore[import-untyped]
import pybreaker
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.exceptions import (  # type: ignore[import-untyped]
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from typing_extensions import override

from foundation.circuit_breaker import with_circuit_breaker
from foundation.exceptions import UpstreamError
from foundation.metrics.decorators import with_client_metrics
from foundation.retry import HTTPErrorClassifier, RetryWithBackoff

from .mixins import CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin

logger = logging.getLogger("clients.s3")

# =============================================================================
# S3-Specific Error Codes
# =============================================================================

# S3 error codes that indicate throttling (always retriable)
S3_THROTTLING_CODES: frozenset[str] = frozenset(
    {
        "SlowDown",
        "RequestLimitExceeded",
        "ProvisionedThroughputExceededException",
        "Throttling",
        "ThrottlingException",
        "RequestThrottled",
        "BandwidthLimitExceeded",
    }
)

# S3 error codes that indicate server-side transient errors (retriable)
S3_TRANSIENT_ERROR_CODES: frozenset[str] = frozenset(
    {
        "InternalError",
        "InternalFailure",
        "ServiceUnavailable",
        "ServiceException",
        "RequestTimeout",
        "RequestExpired",
    }
)

# S3 error codes that indicate client errors (NOT retriable)
S3_NON_RETRIABLE_CODES: frozenset[str] = frozenset(
    {
        # Authentication/Authorization
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "AccessDenied",
        "AllAccessDisabled",
        "AccountProblem",
        "InvalidSecurity",
        # Not Found
        "NoSuchBucket",
        "NoSuchKey",
        "NoSuchUpload",
        "NoSuchVersion",
        "404",
        # Invalid Request
        "InvalidBucketName",
        "InvalidObjectName",
        "InvalidArgument",
        "MalformedPolicy",
        "MalformedXML",
        "InvalidRequest",
        "InvalidPart",
        "InvalidPartOrder",
        # Permission
        "UnauthorizedAccess",
    }
)


# =============================================================================
# S3 Error Classifier
# =============================================================================


class S3ErrorClassifier(HTTPErrorClassifier):
    """S3-specific error classification using foundation base class.

    Classifies boto3/botocore exceptions into retriable and non-retriable
    categories based on exception type and S3 error codes.
    """

    @override
    def is_retriable(self, exc: BaseException) -> bool:
        """Determine if an S3 exception should trigger a retry.

        Args:
            exc: Exception to classify.

        Returns:
            True if the error should trigger a retry, False otherwise.
        """
        # Connection-level errors are always retriable
        if isinstance(exc, (EndpointConnectionError, ConnectionClosedError, ReadTimeoutError)):
            logger.debug("Retriable connection error: %s", type(exc).__name__)
            return True

        # Generic BotoCoreError (not ClientError) - usually infrastructure issues
        if isinstance(exc, BotoCoreError) and not isinstance(exc, ClientError):
            logger.debug("Retriable BotoCoreError: %s", type(exc).__name__)
            return True

        # ClientError needs code-based classification
        if isinstance(exc, ClientError):
            error_response = exc.response.get("Error", {})
            error_code = error_response.get("Code", "")
            http_status = str(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))

            # Explicit non-retriable codes
            if error_code in S3_NON_RETRIABLE_CODES:
                logger.debug("Non-retriable error code: %s", error_code)
                return False

            # Throttling is always retriable
            if error_code in S3_THROTTLING_CODES:
                logger.debug("Retriable throttling error: %s", error_code)
                return True

            # Transient server errors are retriable
            if error_code in S3_TRANSIENT_ERROR_CODES:
                logger.debug("Retriable transient error: %s", error_code)
                return True

            # Use base class HTTP status classification
            if http_status:
                return self.is_retriable_http_status(http_status)

            # Unknown error - default to not retriable to fail fast
            logger.warning("Unknown S3 error code %s - not retrying", error_code)
            return False

        # UpstreamError from circuit breaker - not retriable at this level
        if isinstance(exc, UpstreamError):
            return False

        # Unknown exception type - don't retry
        return False

    @override
    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        """Extract S3-specific error details for logging.

        Args:
            exc: Exception to extract details from.

        Returns:
            Dictionary with error_code and http_status if available.
        """
        if isinstance(exc, ClientError):
            return {
                "error_code": exc.response.get("Error", {}).get("Code", ""),
                "http_status": exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
            }
        return {}


# Create singleton classifier
_s3_classifier = S3ErrorClassifier()


# =============================================================================
# S3 Client Wrapper
# =============================================================================


@attrs.define(frozen=False, slots=True)
class S3ClientWrapper(CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin):
    """Wrapper for boto3 S3 client with resilience features.

    Attributes:
        endpoint_url: S3 service endpoint URL. For MinIO: `http://minio.ml-system:9000`.
            For AWS S3: use regional endpoints like `https://s3.us-east-1.amazonaws.com`
            or omit to use default AWS endpoints.
        access_key_id: S3 access key for authentication.
        secret_access_key: S3 secret key for authentication.
        region_name: AWS region name (default: `us-east-1`). For MinIO, this is ignored
            but required by boto3.
        max_retries: Maximum number of retry attempts for transient errors (default: 3).
        retry_min_wait: Minimum wait between retries in seconds (default: 1.0).
        retry_max_wait: Maximum wait between retries in seconds (default: 10.0).
        timeout_s: Timeout for individual S3 operations in seconds (default: 30).
        circuit_breaker_threshold: Failures before circuit breaker opens (default: 5).
        circuit_breaker_timeout: Seconds before circuit breaker recovery (default: 120).

    Note:
        This class is not frozen because it maintains an internal boto3 client
        instance that may need to be mutable for connection pooling in the future.
    """

    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region_name: str = attrs.field(default="us-east-1")
    max_retries: int = attrs.field(default=3)
    retry_min_wait: float = attrs.field(default=1.0)
    retry_max_wait: float = attrs.field(default=10.0)
    timeout_s: int = attrs.field(default=30)
    circuit_breaker_threshold: int = attrs.field(default=5)
    circuit_breaker_timeout: int = attrs.field(default=120)
    _client: Any = attrs.field(init=False, default=None)
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)

    @override
    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for S3.

        S3 is used by background jobs (not critical path), allow more tolerance.

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return ("s3", self.circuit_breaker_threshold, self.circuit_breaker_timeout)

    def __attrs_post_init__(self) -> None:
        """Initialize the boto3 S3 client and circuit breaker after attrs construction."""
        # Configure boto3 with timeouts and connection settings.
        # NOTE: We intentionally disable boto3's built-in retry mechanism (max_attempts=0)
        # and implement our own via tenacity. This provides:
        # 1. Consistent retry behavior across all clients (S3, Qdrant, Ollama)
        # 2. Integration with our circuit breaker pattern
        # 3. Custom logging on each retry attempt
        # 4. Fine-grained control over which errors trigger retries
        config = Config(
            connect_timeout=self.timeout_s,
            read_timeout=self.timeout_s,
            retries={"max_attempts": 0},
        )

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
            config=config,
        )

        # Initialize circuit breaker from base class
        self._init_circuit_breaker()

    @property
    def client(self) -> Any:
        """Get the underlying boto3 S3 client instance."""
        return self._client

    @classmethod
    def from_config(cls, config: Any) -> "S3ClientWrapper":
        """Create an S3ClientWrapper from a MinIOConfig.

        Args:
            config: MinIOConfig instance with S3 connection and resilience settings.

        Returns:
            Configured S3ClientWrapper instance.

        Example:
            ```python
            from config import MinIOConfig
            from clients.s3 import S3ClientWrapper

            config = MinIOConfig.from_env()
            client = S3ClientWrapper.from_config(config)
            ```
        """
        return cls(
            endpoint_url=config.endpoint_url,
            access_key_id=config.access_key_id,
            secret_access_key=config.secret_access_key,
            region_name=config.region,
            max_retries=config.max_retries,
            retry_min_wait=config.retry_min_wait,
            retry_max_wait=config.retry_max_wait,
            timeout_s=config.timeout,
            circuit_breaker_threshold=config.circuit_breaker_threshold,
            circuit_breaker_timeout=config.circuit_breaker_timeout,
        )

    def _with_retry(self, operation: str, func, *args, **kwargs):
        """Execute a function with retry logic.

        Args:
            operation: Name of the operation (for logging).
            func: Function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func(*args, **kwargs).

        Raises:
            UpstreamError: If all retries are exhausted or non-retriable error occurs.
        """
        retry = RetryWithBackoff(
            max_attempts=self.max_retries,
            wait_min=self.retry_min_wait,
            wait_max=self.retry_max_wait,
            is_retriable=_s3_classifier.is_retriable,
            logger=logger,
            client="s3",
            operation=operation,
        )
        try:
            return retry.call(func, *args, **kwargs)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            attempts = retry.last_attempt_number
            suffix = f" after {attempts} attempts" if attempts > 1 else ""
            msg = f"S3 {operation} failed{suffix}: {error_code}"
            logger.exception(msg, extra={"operation": operation, "error_code": error_code})
            raise UpstreamError(msg) from e
        except BotoCoreError as e:
            attempts = retry.last_attempt_number
            suffix = f" after {attempts} attempts" if attempts > 1 else ""
            msg = f"S3 {operation} failed{suffix}: {e}"
            logger.exception(msg, extra={"operation": operation})
            raise UpstreamError(msg) from e

    @with_client_metrics("s3", "put_object")
    @with_circuit_breaker("s3")
    def put_object(self, *, bucket: str, key: str, body: bytes) -> None:
        """Put an object into S3 with retry and circuit breaker protection.

        Args:
            bucket: Bucket name.
            key: Object key (path).
            body: Object content as bytes.

        Raises:
            UpstreamError: If S3 operation fails after retries or circuit is open.
        """
        self._with_retry(
            "put_object",
            self._client.put_object,
            Bucket=bucket,
            Key=key,
            Body=body,
        )

    @with_client_metrics("s3", "get_object")
    @with_circuit_breaker("s3")
    def get_object(self, *, bucket: str, key: str) -> bytes:
        """Get an object from S3 with retry and circuit breaker protection.

        Args:
            bucket: Bucket name.
            key: Object key (path).

        Returns:
            Object content as bytes.

        Raises:
            UpstreamError: If S3 operation fails after retries or circuit is open.
        """

        def _get_and_read():
            response = self._client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()

        result = self._with_retry("get_object", _get_and_read)
        return cast(bytes, result)

    @with_client_metrics("s3", "list_objects")
    @with_circuit_breaker("s3")
    def list_objects(self, *, bucket: str, prefix: str = "") -> list[str]:
        """List all object keys in a bucket with retry and circuit breaker protection.

        Args:
            bucket: Bucket name to list objects from.
            prefix: Optional prefix to filter objects (default: empty string).

        Returns:
            List of object keys (S3 keys) matching the prefix.

        Raises:
            UpstreamError: If S3 operation fails after retries or circuit is open.

        Example:
            ```python
            keys = client.list_objects(bucket="pipeline", prefix="inputs/")
            # Returns: ["inputs/file1.txt", "inputs/file2.txt", ...]
            ```
        """

        def _list_all():
            keys: list[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    keys.extend([obj["Key"] for obj in page["Contents"]])
            return keys

        result = self._with_retry("list_objects", _list_all)
        return cast(list[str], result)

    def iter_objects(
        self, *, bucket: str, prefix: str = "", page_size: int = 1000
    ) -> Generator[list[str], None, None]:
        """Yield pages of object keys from S3 as they arrive.

        Each yielded list corresponds to one S3 API response page.
        Use this instead of ``list_objects`` when the bucket is large and
        you want to start processing before the full listing completes.

        Note:
            No ``@with_circuit_breaker`` or ``_with_retry`` wrapping.
            The decorator pattern doesn't compose with generators (breaker.call
            returns the generator object immediately without executing the body).
            boto3's paginator handles transient HTTP retries internally.
            Callers should catch ``ClientError`` / ``BotoCoreError`` at the
            iteration site.

        Args:
            bucket: Bucket name to list objects from.
            prefix: Optional prefix to filter objects (default: empty string).
            page_size: Maximum keys per S3 API page (default: 1000, S3 max).
                Maps to the ``MaxKeys`` paginator parameter.

        Yields:
            Lists of object keys, one per S3 API response page.
        """
        paginator = self._client.get_paginator("list_objects_v2")
        page_num = 0
        total_keys = 0
        logger.debug(
            "iter_objects: starting pagination",
            extra={"bucket": bucket, "prefix": prefix, "page_size": page_size},
        )
        paginate_kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if page_size != 1000:
            paginate_kwargs["PaginationConfig"] = {"PageSize": page_size}
        for page in paginator.paginate(**paginate_kwargs):
            if "Contents" in page:
                keys = [obj["Key"] for obj in page["Contents"]]
                page_num += 1
                total_keys += len(keys)
                logger.info(
                    "iter_objects: page %d yielded %d keys (%d cumulative)",
                    page_num,
                    len(keys),
                    total_keys,
                    extra={
                        "page": page_num,
                        "page_keys": len(keys),
                        "cumulative_keys": total_keys,
                    },
                )
                yield keys
        logger.info(
            "iter_objects: pagination complete — %d pages, %d total keys",
            page_num,
            total_keys,
            extra={"total_pages": page_num, "total_keys": total_keys},
        )

    async def get_object_async(self, *, bucket: str, key: str) -> bytes:
        """Get an object from S3 asynchronously using ThreadPoolExecutor.

        Note:
            This uses ThreadPoolExecutor to run the blocking boto3 call in a thread pool.
            For true async I/O, consider using aioboto3 when dependency conflicts are resolved.

        Args:
            bucket: Bucket name.
            key: Object key (path).

        Returns:
            Object content as bytes.

        Raises:
            UpstreamError: If the object doesn't exist or access is denied.

        Example:
            ```python
            content = await client.get_object_async(bucket="pipeline", key="data.txt")
            ```
        """
        loop = asyncio.get_running_loop()
        # Run blocking boto3 call in thread pool for async-like behavior
        return await loop.run_in_executor(None, lambda: self.get_object(bucket=bucket, key=key))

    def close(self) -> None:
        """Close the S3 client and release resources.

        This should be called when the client is no longer needed.
        """
        if self._client is not None:
            # boto3 clients don't have explicit close, but we clear the reference
            self._client = None
            logger.debug("S3 client closed")
