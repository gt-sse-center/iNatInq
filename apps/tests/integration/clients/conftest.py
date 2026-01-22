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

## Qdrant Container

The Qdrant container provides vector database functionality:
- Ports: Random ephemeral host ports (testcontainers handles this)
- Container ports: 6333 (HTTP API), 6334 (gRPC)
- Session-scoped for performance
- Cleanup: Automatic via fixture teardown + Ryuk orphan cleanup

## Ollama Container

The Ollama container provides embedding generation functionality:
- Ports: Random ephemeral host ports (testcontainers handles this)
- Container port: 11434 (API)
- Uses the all-minilm model (smallest model for faster tests)
- Session-scoped for performance (model is pre-pulled on first use)
- Cleanup: Automatic via fixture teardown + Ryuk orphan cleanup

## Usage

```python
# S3/MinIO
def test_s3_operations(minio_client: S3ClientWrapper, test_bucket: str):
    minio_client.put_object(bucket=test_bucket, key="test.txt", body=b"hello")
    data = minio_client.get_object(bucket=test_bucket, key="test.txt")
    assert data == b"hello"

# Qdrant
async def test_qdrant_operations(qdrant_client: QdrantClientWrapper, test_collection: str):
    from qdrant_client.models import PointStruct
    await qdrant_client.batch_upsert_async(
        collection=test_collection,
        points=[PointStruct(id="1", vector=[0.1] * 768, payload={"text": "hello"})],
        vector_size=768,
    )

# Ollama
def test_ollama_operations(ollama_client: OllamaClient):
    embedding = ollama_client.embed("hello world")
    assert len(embedding) == 384  # all-minilm dimension
```
"""

import logging
import time
import uuid

import httpx
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.minio import MinioContainer
from testcontainers.qdrant import QdrantContainer

from clients.ollama import OllamaClient
from clients.qdrant import QdrantClientWrapper
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
        secret_key="minioadmin",  # noqa: S106 - Test credentials
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


# =============================================================================
# Qdrant Container Fixtures
# =============================================================================


def _get_qdrant_url(container: QdrantContainer) -> str:
    """Get the Qdrant HTTP URL from a container.

    Args:
        container: Running Qdrant container.

    Returns:
        str: HTTP URL for Qdrant REST API.
    """
    return f"http://{container.rest_host_address}"


@pytest.fixture(scope="session")
def qdrant_container():
    """Start a Qdrant container for the test session.

    The container is started once and shared across all tests in the session.
    This significantly reduces test overhead compared to per-test containers.

    Yields:
        QdrantContainer: Running Qdrant container with connection info.
    """
    logger.info("Starting Qdrant container...")

    # Use v1.16.x to match qdrant-client 1.16.x (minor version diff must be ≤1)
    container = QdrantContainer(image="qdrant/qdrant:v1.16.0")
    container.start()

    # Wait for container to be healthy
    _wait_for_qdrant_health(container)

    logger.info(
        "Qdrant container started",
        extra={
            "url": _get_qdrant_url(container),
            "container_id": container.get_wrapped_container().short_id,
        },
    )

    yield container

    logger.info("Stopping Qdrant container...")
    container.stop()


def _wait_for_qdrant_health(container: QdrantContainer, timeout: int = 60) -> None:
    """Wait for Qdrant container to be ready.

    Args:
        container: Qdrant container instance.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If Qdrant doesn't become healthy within timeout.
    """
    health_url = f"{_get_qdrant_url(container)}/healthz"

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.5)

    raise TimeoutError(f"Qdrant container not healthy after {timeout}s")


@pytest.fixture(scope="session")
def qdrant_url(qdrant_container: QdrantContainer) -> str:
    """Get Qdrant connection URL.

    Returns:
        str: Qdrant HTTP API URL.
    """
    return _get_qdrant_url(qdrant_container)


@pytest.fixture(scope="session")
def qdrant_client(qdrant_url: str) -> QdrantClientWrapper:
    """Create a QdrantClientWrapper connected to the test Qdrant instance.

    Session-scoped to share the client across tests for efficiency.
    Tests should use unique collection names to avoid collisions.

    Returns:
        QdrantClientWrapper: Client connected to test Qdrant.
    """
    client = QdrantClientWrapper(url=qdrant_url)

    logger.info(
        "Created Qdrant client for integration tests",
        extra={"url": qdrant_url},
    )

    yield client

    # Cleanup
    client.close()


@pytest.fixture
def test_collection(qdrant_client: QdrantClientWrapper) -> str:
    """Create a unique test collection that's cleaned up after the test.

    Each test gets a fresh collection to ensure isolation.

    Yields:
        str: Unique collection name.
    """
    import asyncio

    collection_name = f"test-{uuid.uuid4().hex[:12]}"

    logger.debug("Created test collection", extra={"collection": collection_name})

    yield collection_name

    # Cleanup: Delete collection using async client
    try:
        asyncio.run(qdrant_client._client.delete_collection(collection_name=collection_name))
        logger.debug("Deleted test collection", extra={"collection": collection_name})
    except Exception as e:
        logger.warning(
            "Collection cleanup failed",
            extra={"collection": collection_name, "error": str(e)},
        )


@pytest.fixture
def sample_vector() -> list[float]:
    """Provide a sample embedding vector for tests.

    Returns:
        list[float]: 768-dimensional sample vector.
    """
    import random

    random.seed(42)
    return [random.random() for _ in range(768)]  # noqa: S311 - Non-cryptographic use


@pytest.fixture
def vector_size() -> int:
    """Standard vector dimension for tests.

    Returns:
        int: Vector dimension (768 for nomic-embed-text compatibility).
    """
    return 768


# =============================================================================
# Ollama Container Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ollama_container():
    """Start an Ollama container for the test session.

    The container is started once and shared across all tests in the session.
    Uses the all-minilm model which is small (~25MB) for faster tests.

    Note:
        The first request will pull the model, which can take 30-60 seconds.
        Subsequent requests are fast.

    Yields:
        DockerContainer: Running Ollama container with connection info.
    """
    logger.info("Starting Ollama container...")

    # Use generic DockerContainer since testcontainers doesn't have Ollama module
    container = (
        DockerContainer("ollama/ollama:latest").with_exposed_ports(11434).with_env("OLLAMA_HOST", "0.0.0.0")
    )

    container.start()

    # Wait for Ollama API to be ready
    _wait_for_ollama_health(container)

    # Pull the embedding model (all-minilm is small and fast)
    _pull_ollama_model(container, "all-minilm")

    logger.info(
        "Ollama container started",
        extra={
            "url": _get_ollama_url(container),
            "container_id": container.get_wrapped_container().short_id,
        },
    )

    yield container

    logger.info("Stopping Ollama container...")
    container.stop()


def _get_ollama_url(container: DockerContainer) -> str:
    """Get the Ollama HTTP URL from a container.

    Args:
        container: Running Ollama container.

    Returns:
        str: HTTP URL for Ollama API.
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(11434)
    return f"http://{host}:{port}"


def _wait_for_ollama_health(container: DockerContainer, timeout: int = 60) -> None:
    """Wait for Ollama container to be ready.

    Args:
        container: Ollama container instance.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If Ollama doesn't become healthy within timeout.
    """
    url = _get_ollama_url(container)
    health_url = f"{url}/api/tags"  # List models endpoint

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.5)

    raise TimeoutError(f"Ollama container not healthy after {timeout}s")


def _pull_ollama_model(container: DockerContainer, model: str, timeout: int = 120) -> None:
    """Pull an embedding model in the Ollama container.

    Args:
        container: Running Ollama container.
        model: Model name to pull (e.g., "all-minilm").
        timeout: Maximum seconds to wait for pull.

    Raises:
        TimeoutError: If model doesn't pull within timeout.
    """
    url = _get_ollama_url(container)
    pull_url = f"{url}/api/pull"

    logger.info("Pulling Ollama model: %s", model)

    # Start the pull
    try:
        response = httpx.post(
            pull_url,
            json={"name": model},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to pull model {model}: {response.text}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to pull model {model}: {e}") from e

    logger.info("Ollama model pulled: %s", model)


@pytest.fixture(scope="session")
def ollama_url(ollama_container: DockerContainer) -> str:
    """Get Ollama connection URL.

    Returns:
        str: Ollama HTTP API URL.
    """
    return _get_ollama_url(ollama_container)


@pytest.fixture(scope="session")
def ollama_client(ollama_url: str) -> OllamaClient:
    """Create an OllamaClient connected to the test Ollama instance.

    Session-scoped to share the client across tests for efficiency.
    Uses the all-minilm model for fast embedding generation.

    Returns:
        OllamaClient: Client connected to test Ollama.
    """
    client = OllamaClient(
        base_url=ollama_url,
        model="all-minilm",
        timeout_s=30,
    )

    logger.info(
        "Created Ollama client for integration tests",
        extra={"url": ollama_url, "model": "all-minilm"},
    )

    yield client

    # Cleanup
    client.close()
