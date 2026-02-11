"""Fixtures specific to client integration tests.

Container and client fixtures (minio_client, qdrant_client, ollama_client,
test_bucket, test_collection) are defined in tests/integration/conftest.py
and are available here. This module adds client-test-specific fixtures:
sample vectors, CLIP container, and utility fixtures.
"""

import logging
import time
import uuid

import httpx
import pytest
from testcontainers.core.container import DockerContainer

logger = logging.getLogger(__name__)

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
    _wait_for_clip_health(container)
    logger.info("CLIP container started", extra={"url": _get_clip_url(container)})
    yield container
    logger.info("Stopping CLIP container...")
    container.stop()


@pytest.fixture(scope="session")
def clip_url(clip_container: DockerContainer) -> str:
    """Get CLIP connection URL."""
    return _get_clip_url(clip_container)
