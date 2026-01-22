"""Fixtures for client integration tests.

This module provides container-based fixtures for testing client wrappers
against real services. Containers are managed by testcontainers-python.

## MinIO Container

The MinIO container provides S3-compatible object storage:
- Ports: Random ephemeral host ports (testcontainers handles this)
- Container ports: 9000 (API), 9001 (Console)
- Default credentials: minioadmin/minioadmin
- Session-scoped for performance
- Cleanup: Automatic via fixture teardown + Ryuk orphan cleanup

## Usage

```python
def test_s3_operations(minio_client: S3ClientWrapper, test_bucket: str):
    minio_client.put_object(bucket=test_bucket, key="test.txt", body=b"hello")
    data = minio_client.get_object(bucket=test_bucket, key="test.txt")
    assert data == b"hello"
```
"""

import logging
import time
import uuid

import pytest
from testcontainers.minio import MinioContainer

from clients.s3 import S3ClientWrapper

logger = logging.getLogger(__name__)

# =============================================================================
# MinIO Container Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def minio_container():
    """Start a MinIO container for the test session.

    The container is started once and shared across all tests in the session.
    This significantly reduces test overhead compared to per-test containers.

    Yields:
        MinioContainer: Running MinIO container with connection info.
    """
    logger.info("Starting MinIO container...")

    container = MinioContainer(
        image="minio/minio:RELEASE.2024-01-01T16-36-33Z",
        access_key="minioadmin",
        secret_key="minioadmin",
    )

    container.start()

    # Wait for container to be healthy
    _wait_for_minio_health(container)

    logger.info(
        "MinIO container started",
        extra={
            "endpoint": container.get_config()["endpoint"],
            "container_id": container.get_wrapped_container().short_id,
        },
    )

    yield container

    logger.info("Stopping MinIO container...")
    container.stop()


def _wait_for_minio_health(container: MinioContainer, timeout: int = 30) -> None:
    """Wait for MinIO container to be ready.

    Args:
        container: MinIO container instance.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If MinIO doesn't become healthy within timeout.
    """
    import httpx

    config = container.get_config()
    health_url = f"http://{config['endpoint']}/minio/health/live"

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.5)

    raise TimeoutError(f"MinIO container not healthy after {timeout}s")


@pytest.fixture(scope="session")
def minio_config(minio_container: MinioContainer) -> dict[str, str]:
    """Get MinIO connection configuration.

    Returns:
        dict: Configuration with endpoint_url, access_key_id, secret_access_key.
    """
    config = minio_container.get_config()
    return {
        "endpoint_url": f"http://{config['endpoint']}",
        "access_key_id": config["access_key"],
        "secret_access_key": config["secret_key"],
    }


@pytest.fixture(scope="session")
def minio_client(minio_config: dict[str, str]) -> S3ClientWrapper:
    """Create an S3ClientWrapper connected to the test MinIO instance.

    Session-scoped to share the client across tests for efficiency.
    Tests should use unique bucket/key names to avoid collisions.

    Returns:
        S3ClientWrapper: Client connected to test MinIO.
    """
    client = S3ClientWrapper(
        endpoint_url=minio_config["endpoint_url"],
        access_key_id=minio_config["access_key_id"],
        secret_access_key=minio_config["secret_access_key"],
        max_retries=3,
        retry_min_wait=0.1,  # Fast retries for tests
        retry_max_wait=1.0,
        timeout_s=10,
    )

    logger.info(
        "Created S3 client for integration tests",
        extra={"endpoint": minio_config["endpoint_url"]},
    )

    yield client

    # Cleanup
    client.close()


@pytest.fixture
def test_bucket(minio_client: S3ClientWrapper) -> str:
    """Create a unique test bucket that's cleaned up after the test.

    Each test gets a fresh bucket to ensure isolation.

    Yields:
        str: Unique bucket name.
    """
    bucket_name = f"test-{uuid.uuid4().hex[:12]}"

    minio_client.ensure_bucket(bucket_name)
    logger.debug("Created test bucket", extra={"bucket": bucket_name})

    yield bucket_name

    # Cleanup: Delete all objects and bucket
    try:
        keys = minio_client.list_objects(bucket=bucket_name)
        for key in keys:
            minio_client.delete_object(bucket=bucket_name, key=key)
        # Note: boto3 doesn't have delete_bucket in our wrapper, but MinIO
        # will clean up on container stop anyway
    except Exception as e:
        logger.warning("Bucket cleanup failed", extra={"bucket": bucket_name, "error": str(e)})


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def unique_key() -> str:
    """Generate a unique object key for test isolation.

    Returns:
        str: Unique S3 key.
    """
    return f"test-{uuid.uuid4().hex}"


@pytest.fixture
def sample_data() -> bytes:
    """Provide sample data for upload tests.

    Returns:
        bytes: Sample binary data.
    """
    return b"Integration test sample data - " + uuid.uuid4().bytes
