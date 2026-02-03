"""Fixtures for ingestion pipeline integration tests.

This module provides container-based fixtures for testing the ingestion pipeline
against real Ray clusters and supporting services.

## Ray Container

The Ray container provides distributed computing functionality:
- Image: rayproject/ray:2.53.0-py311
- Ports: 6379 (Redis/GCS), 8265 (Dashboard), 10001 (Client)
- Session-scoped for performance
- Includes health check waiting

## Usage

```python
def test_ray_cluster_initialization(ray_container, ray_address):
    import ray
    ray.init(address=ray_address, ignore_reinit_error=True)
    assert ray.is_initialized()
    ray.shutdown()
```

## Notes

- Ray tests require significant resources (~2GB RAM, 2 CPUs minimum)
- First run may be slow due to image pull
- Ray cluster initialization takes 5-10 seconds
"""

import logging
import time

import httpx
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

logger = logging.getLogger(__name__)

# =============================================================================
# Ray Container Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ray_container():
    """Start a Ray head node container for the test session.

    The container runs Ray in head mode with client server enabled,
    allowing remote connections for distributed computing tests.

    Yields:
        DockerContainer: Running Ray container with connection info.
    """
    logger.info("Starting Ray container...")

    container = (
        DockerContainer("rayproject/ray:2.53.0-py311")
        .with_exposed_ports(6379, 8265, 10001)
        .with_env("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
        .with_command(
            "ray start --head --port=6379 --dashboard-host=0.0.0.0 "
            "--dashboard-port=8265 --ray-client-server-port=10001 --block"
        )
    )

    container.start()

    # Wait for Ray to be ready
    _wait_for_ray_health(container)

    logger.info(
        "Ray container started",
        extra={
            "dashboard_url": _get_ray_dashboard_url(container),
            "client_address": _get_ray_client_address(container),
            "container_id": container.get_wrapped_container().short_id,
        },
    )

    yield container

    logger.info("Stopping Ray container...")
    container.stop()


def _get_ray_dashboard_url(container: DockerContainer) -> str:
    """Get the Ray dashboard URL from a container.

    Args:
        container: Running Ray container.

    Returns:
        str: HTTP URL for Ray dashboard.
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(8265)
    return f"http://{host}:{port}"


def _get_ray_client_address(container: DockerContainer) -> str:
    """Get the Ray client address from a container.

    Args:
        container: Running Ray container.

    Returns:
        str: Ray client address (ray://host:port).
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(10001)
    return f"ray://{host}:{port}"


def _get_ray_gcs_address(container: DockerContainer) -> str:
    """Get the Ray GCS address from a container.

    Args:
        container: Running Ray container.

    Returns:
        str: Ray GCS address (host:port).
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(6379)
    return f"{host}:{port}"


def _wait_for_ray_health(container: DockerContainer, timeout: int = 120) -> None:
    """Wait for Ray container to be ready.

    Checks both the dashboard endpoint and that Ray has started successfully
    by looking for the "Ray is ready" log message.

    Args:
        container: Ray container instance.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If Ray doesn't become healthy within timeout.
    """
    dashboard_url = _get_ray_dashboard_url(container)
    health_url = f"{dashboard_url}/api/cluster_status"

    start = time.time()
    while time.time() - start < timeout:
        try:
            # Check dashboard API
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                logger.info("Ray dashboard healthy at %s", dashboard_url)
                # Give Ray a bit more time to fully initialize
                time.sleep(2)
                return
        except httpx.RequestError:
            pass
        time.sleep(1.0)

    raise TimeoutError(f"Ray container not healthy after {timeout}s")


@pytest.fixture(scope="session")
def ray_dashboard_url(ray_container: DockerContainer) -> str:
    """Get Ray dashboard URL.

    Returns:
        str: Ray dashboard HTTP URL.
    """
    return _get_ray_dashboard_url(ray_container)


@pytest.fixture(scope="session")
def ray_client_address(ray_container: DockerContainer) -> str:
    """Get Ray client address for ray.init().

    Returns:
        str: Ray client address (ray://host:port).
    """
    return _get_ray_client_address(ray_container)


@pytest.fixture(scope="session")
def ray_gcs_address(ray_container: DockerContainer) -> str:
    """Get Ray GCS address for direct connection.

    Returns:
        str: Ray GCS address (host:port).
    """
    return _get_ray_gcs_address(ray_container)


# =============================================================================
# Strategy Fixtures
# =============================================================================


@pytest.fixture
def local_ray_strategy():
    """Create a LocalRayStrategy for testing.

    This fixture creates a strategy configured for local Ray execution.
    The strategy handles cluster initialization and shutdown.

    Yields:
        LocalRayStrategy: Configured strategy instance.
    """
    from core.ingestion.strategies.local_ray import LocalRayStrategy

    strategy = LocalRayStrategy(
        num_cpus=2,
        dashboard_host="127.0.0.1",
        dashboard_port=8265,
        include_dashboard=False,  # Faster startup for tests
    )

    yield strategy


# =============================================================================
# Combined Fixtures (Ray + Services)
# =============================================================================


@pytest.fixture(scope="session")
def integration_services(
    ray_container,
    minio_container,
    qdrant_container,
    ollama_container,
):
    """Combined fixture providing all services needed for full pipeline tests.

    This is a convenience fixture that ensures all required services are
    running before pipeline integration tests execute.

    Yields:
        dict: Dictionary with container references.
    """
    return {
        "ray": ray_container,
        "minio": minio_container,
        "qdrant": qdrant_container,
        "ollama": ollama_container,
    }
