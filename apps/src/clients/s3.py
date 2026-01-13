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
client.ensure_bucket("pipeline")
client.put_object(bucket="pipeline", key="data.txt", body=b"hello")
```

## Design

The client wrapper:
- Encapsulates S3 connection configuration
- Provides a clean interface for bucket and object operations
- Handles errors consistently
- Uses attrs for concise, correct class definition
"""


import asyncio
from typing import Any, cast

import attrs
import boto3  # type: ignore[import-untyped]
import pybreaker
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from core.exceptions import UpstreamError
from foundation.circuit_breaker import with_circuit_breaker
from .mixins import CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin


@attrs.define(frozen=False, slots=True)
class S3ClientWrapper(CircuitBreakerMixin, ConfigValidationMixin, LoggerMixin):
    """Wrapper for boto3 S3 client with convenience methods.

    Attributes:
        endpoint_url: S3 service endpoint URL. For MinIO: `http://minio.ml-system:9000`.
            For AWS S3: use regional endpoints like `https://s3.us-east-1.amazonaws.com`
            or omit to use default AWS endpoints.
        access_key_id: S3 access key for authentication.
        secret_access_key: S3 secret key for authentication.
        region_name: AWS region name (default: `us-east-1`). For MinIO, this is ignored
            but required by boto3.

    Note:
        This class is not frozen because it maintains an internal boto3 client
        instance that may need to be mutable for connection pooling in the future.
    """

    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region_name: str = attrs.field(default="us-east-1")
    _client: Any = attrs.field(init=False, default=None)
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)

    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for S3.

        S3 is used by background jobs (not critical path), allow more tolerance.

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return ("s3", 5, 120)

    def __attrs_post_init__(self) -> None:
        """Initialize the boto3 S3 client and circuit breaker after attrs construction."""
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
        )

        # Initialize circuit breaker from base class
        self._init_circuit_breaker()

    @property
    def client(self) -> Any:
        """Get the underlying boto3 S3 client instance."""
        return self._client

    def ensure_bucket(self, bucket: str) -> None:
        """Ensure an S3 bucket exists, creating it if missing.

        This is a dev convenience function. In production, buckets are typically
        created via infrastructure-as-code (Terraform, CloudFormation, etc.).

        Args:
            bucket: Bucket name to ensure exists.

        Note:
            The function first checks if the bucket exists via `head_bucket()`. If
            that raises a `ClientError` (bucket not found), it creates the bucket.
            Other errors (e.g., permission denied) will propagate as exceptions.
        """
        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError:
            self._client.create_bucket(Bucket=bucket)

    @with_circuit_breaker("s3")
    def put_object(self, *, bucket: str, key: str, body: bytes) -> None:
        """Put an object into S3.

        Args:
            bucket: Bucket name.
            key: Object key (path).
            body: Object content as bytes.

        Raises:
            UpstreamError: If S3 operation fails. Also raised when circuit breaker is open.
        """
        try:
            self._client.put_object(Bucket=bucket, Key=key, Body=body)
        except ClientError as e:
            msg = f"S3 put_object failed: {e}"
            raise UpstreamError(msg) from e

    @with_circuit_breaker("s3")
    def get_object(self, *, bucket: str, key: str) -> bytes:
        """Get an object from S3.

        Args:
            bucket: Bucket name.
            key: Object key (path).

        Returns:
            Object content as bytes.

        Raises:
            UpstreamError: If S3 operation fails. Also raised when circuit breaker is open.
            ClientError: If the object doesn't exist or access is denied.
        """
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()
            return cast(bytes, body)
        except ClientError as e:
            msg = f"S3 get_object failed: {e}"
            raise UpstreamError(msg) from e

    @with_circuit_breaker("s3")
    def list_objects(self, *, bucket: str, prefix: str = "") -> list[str]:
        """List all object keys in a bucket with the given prefix.

        Args:
            bucket: Bucket name to list objects from.
            prefix: Optional prefix to filter objects (default: empty string).

        Returns:
            List of object keys (S3 keys) matching the prefix.

        Raises:
            UpstreamError: If S3 operation fails. Also raised when circuit breaker is open.

        Example:
            ```python
            keys = client.list_objects(bucket="pipeline", prefix="inputs/")
            # Returns: ["inputs/file1.txt", "inputs/file2.txt", ...]
            ```
        """
        try:
            keys: list[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    keys.extend([obj["Key"] for obj in page["Contents"]])
            return keys
        except ClientError as e:
            msg = f"S3 list_objects failed: {e}"
            raise UpstreamError(msg) from e

    def exists(self, *, bucket: str, key: str) -> bool:
        """Check if an object exists in S3.

        Args:
            bucket: Bucket name.
            key: Object key (path).

        Returns:
            True if the object exists, False otherwise.
        """
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            # 404 means object doesn't exist
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            # Re-raise other errors (permission denied, etc.)
            raise

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
                ClientError: If the object doesn't exist or access is denied.

        Example:
                ```python
                content = await client.get_object_async(bucket="pipeline", key="data.txt")
                ```
        """
        loop = asyncio.get_event_loop()
        # Run blocking boto3 call in thread pool for async-like behavior
        return await loop.run_in_executor(None, lambda: self.get_object(bucket=bucket, key=key))
