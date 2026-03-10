"""Fixtures for client integration tests.

This module provides container-based fixtures for testing client wrappers
against real services. Containers are managed by testcontainers-python.

All container fixtures are session-scoped for performance. Per-test fixtures
(test_bucket, test_collection, unique_key) provide isolation.
"""

import asyncio
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


def _wait_for_minio_health(container: MinioContainer, timeout: int = 30) -> None:
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
def minio_container():
    """Start a MinIO container for the test session."""
    logger.info("Starting MinIO container...")
    container = MinioContainer(
        image="minio/minio:RELEASE.2024-01-01T16-36-33Z",
        access_key="minioadmin",
        secret_key="minioadmin",  # noqa: S106 - Test credentials
    )
    container.start()
    _wait_for_minio_health(container)
    logger.info(
        "MinIO container started",
        extra={"endpoint": container.get_config()["endpoint"]},
    )
    yield container
    logger.info("Stopping MinIO container...")
    container.stop()


@pytest.fixture(scope="session")
def minio_config(minio_container: MinioContainer) -> dict[str, str]:
    """Get MinIO connection configuration."""
    config = minio_container.get_config()
    return {
        "endpoint_url": f"http://{config['endpoint']}",
        "access_key_id": config["access_key"],
        "secret_access_key": config["secret_key"],
    }


@pytest.fixture(scope="session")
def minio_client(minio_config: dict[str, str]) -> S3ClientWrapper:
    """Create an S3ClientWrapper connected to the test MinIO instance."""
    client = S3ClientWrapper(
        endpoint_url=minio_config["endpoint_url"],
        access_key_id=minio_config["access_key_id"],
        secret_access_key=minio_config["secret_access_key"],
        max_retries=3,
        retry_min_wait=0.1,
        retry_max_wait=1.0,
        timeout_s=10,
    )
    logger.info("Created S3 client for integration tests", extra={"endpoint": minio_config["endpoint_url"]})
    yield client
    client.close()


@pytest.fixture
def test_bucket(minio_client: S3ClientWrapper) -> str:
    """Create a unique test bucket that's cleaned up after the test."""
    bucket_name = f"test-{uuid.uuid4().hex[:12]}"
    minio_client.client.create_bucket(Bucket=bucket_name)
    yield bucket_name
    try:
        keys = minio_client.list_objects(bucket=bucket_name)
        for key in keys:
            minio_client.client.delete_object(Bucket=bucket_name, Key=key)
    except Exception as e:
        logger.warning("Bucket cleanup failed", extra={"bucket": bucket_name, "error": str(e)})


# =============================================================================
# Qdrant Container Fixtures
# =============================================================================


def _get_qdrant_url(container: QdrantContainer) -> str:
    return f"http://{container.rest_host_address}"


def _wait_for_qdrant_health(container: QdrantContainer, timeout: int = 60) -> None:
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
def qdrant_container():
    """Start a Qdrant container for the test session."""
    logger.info("Starting Qdrant container...")
    container = QdrantContainer(image="qdrant/qdrant:v1.16.0")
    container.start()
    _wait_for_qdrant_health(container)
    logger.info("Qdrant container started", extra={"url": _get_qdrant_url(container)})
    yield container
    logger.info("Stopping Qdrant container...")
    container.stop()


@pytest.fixture(scope="session")
def qdrant_url(qdrant_container: QdrantContainer) -> str:
    """Get Qdrant connection URL."""
    return _get_qdrant_url(qdrant_container)


@pytest.fixture(scope="session")
def qdrant_client(qdrant_url: str) -> QdrantClientWrapper:
    """Create a QdrantClientWrapper connected to the test Qdrant instance."""
    client = QdrantClientWrapper(url=qdrant_url)
    logger.info("Created Qdrant client for integration tests", extra={"url": qdrant_url})
    yield client
    client.close()


@pytest.fixture
def test_collection(qdrant_client: QdrantClientWrapper) -> str:
    """Create a unique test collection that's cleaned up after the test."""
    collection_name = f"test-{uuid.uuid4().hex[:12]}"
    yield collection_name
    try:
        asyncio.run(qdrant_client._client.delete_collection(collection_name=collection_name))
    except Exception as e:
        logger.warning("Collection cleanup failed", extra={"collection": collection_name, "error": str(e)})


# =============================================================================
# Ollama Container Fixtures
# =============================================================================


def _get_ollama_url(container: DockerContainer) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(11434)
    return f"http://{host}:{port}"


def _wait_for_ollama_health(container: DockerContainer, timeout: int = 60) -> None:
    url = _get_ollama_url(container)
    health_url = f"{url}/api/tags"
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
    url = _get_ollama_url(container)
    logger.info("Pulling Ollama model: %s", model)
    try:
        response = httpx.post(f"{url}/api/pull", json={"name": model}, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to pull model {model}: {response.text}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to pull model {model}: {e}") from e
    logger.info("Ollama model pulled: %s", model)


@pytest.fixture(scope="session")
def ollama_container():
    """Start an Ollama container for the test session."""
    logger.info("Starting Ollama container...")
    container = (
        DockerContainer("ollama/ollama:latest").with_exposed_ports(11434).with_env("OLLAMA_HOST", "0.0.0.0")
    )
    container.start()
    _wait_for_ollama_health(container)
    _pull_ollama_model(container, "all-minilm")
    logger.info("Ollama container started", extra={"url": _get_ollama_url(container)})
    yield container
    logger.info("Stopping Ollama container...")
    container.stop()


@pytest.fixture(scope="session")
def ollama_url(ollama_container: DockerContainer) -> str:
    """Get Ollama connection URL."""
    return _get_ollama_url(ollama_container)


@pytest.fixture(scope="session")
def ollama_client(ollama_url: str) -> OllamaClient:
    """Create an OllamaClient connected to the test Ollama instance."""
    client = OllamaClient(base_url=ollama_url, model="all-minilm", timeout_s=30)
    logger.info(
        "Created Ollama client for integration tests", extra={"url": ollama_url, "model": "all-minilm"}
    )
    yield client
    asyncio.run(client.close())


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def unique_key() -> str:
    """Generate a unique object key for test isolation."""
    return f"test-{uuid.uuid4().hex}"


@pytest.fixture
def sample_data() -> bytes:
    """Provide sample data for upload tests."""
    return b"Integration test sample data - " + uuid.uuid4().bytes


@pytest.fixture
def sample_vector() -> list[float]:
    """Provide a sample embedding vector for tests (768-dimensional)."""
    import random

    random.seed(42)
    return [random.random() for _ in range(768)]  # noqa: S311 - Non-cryptographic use


@pytest.fixture
def vector_size() -> int:
    """Standard vector dimension for tests (768 for nomic-embed-text)."""
    return 768


# =============================================================================
# CLIP Container Fixtures (ai4all/clip)
# =============================================================================


def _get_clip_url(container: DockerContainer) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(8000)
    return f"http://{host}:{port}"


def _wait_for_clip_health(container: DockerContainer, timeout: int = 120) -> None:
    url = _get_clip_url(container)
    health_endpoints = [f"{url}/health", f"{url}/", f"{url}/docs"]
    start = time.time()
    while time.time() - start < timeout:
        for health_url in health_endpoints:
            try:
                response = httpx.get(health_url, timeout=5.0)
                if response.status_code in (200, 404):
                    logger.info("CLIP container healthy at %s", health_url)
                    return
            except httpx.RequestError:
                pass
        time.sleep(1.0)
    raise TimeoutError(f"CLIP container not healthy after {timeout}s")


@pytest.fixture(scope="session")
def clip_container():
    """Start an ai4all/clip container for image embedding tests."""
    logger.info("Starting CLIP container (ai4all/clip)...")
    container = DockerContainer("ai4all/clip:latest").with_exposed_ports(8000)
    container.start()
    _wait_for_clip_health(container, timeout=240)
    logger.info("CLIP container started", extra={"url": _get_clip_url(container)})
    yield container
    logger.info("Stopping CLIP container...")
    container.stop()


@pytest.fixture(scope="session")
def clip_url(clip_container: DockerContainer) -> str:
    """Get CLIP connection URL."""
    return _get_clip_url(clip_container)


# =============================================================================
# Infinity Container Fixtures (michaelf34/infinity — SigLIP)
# =============================================================================


def _get_infinity_url(container: DockerContainer) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(7997)
    return f"http://{host}:{port}"


def _wait_for_infinity_health(container: DockerContainer, timeout: int = 300) -> None:
    url = _get_infinity_url(container)
    health_url = f"{url}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                logger.info("Infinity container healthy at %s", health_url)
                return
        except httpx.RequestError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Infinity container not healthy after {timeout}s")


@pytest.fixture(scope="session")
def infinity_container():
    """Start an Infinity container for SigLIP image/text embedding tests.

    Uses the michaelf34/infinity image with google/siglip-so400m-patch14-384 model.
    First start downloads the model (~1.7 GB). Subsequent runs use Docker layer
    caching but still need to load the model into memory (~30-60s).
    """
    logger.info("Starting Infinity container (michaelf34/infinity)...")
    container = (
        DockerContainer("michaelf34/infinity:latest")
        .with_exposed_ports(7997)
        .with_command("v2 --model-id google/siglip-so400m-patch14-384 --port 7997")
    )
    container.start()
    _wait_for_infinity_health(container, timeout=300)
    logger.info("Infinity container started", extra={"url": _get_infinity_url(container)})
    yield container
    logger.info("Stopping Infinity container...")
    container.stop()


@pytest.fixture(scope="session")
def infinity_url(infinity_container: DockerContainer) -> str:
    """Get Infinity connection URL."""
    return _get_infinity_url(infinity_container)
