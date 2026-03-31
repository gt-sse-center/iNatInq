"""Shared fixtures for DLQ integration tests.

Provides session-scoped Redis testcontainer for DLQ backend tests. This is
important because DLQ integration tests must verify real message persistence
and retry logic against an actual Redis instance.
"""

import pytest
import time
import logging
from collections.abc import Generator
from testcontainers.core.container import DockerContainer

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def redis_container() -> Generator[tuple[str, int]]:
    """Start Redis test container.

    Returns:
        (container_host_ip: str, container_port: int)
    """
    redis_port = 6379
    logger.info("Starting Redis container...")
    container = DockerContainer(image="redis:8.6").with_exposed_ports(redis_port)
    container.start()

    healthcheck_timeout = 60

    def is_healthy() -> bool:
        """Ping the container until healthy or timeout expires."""
        start_time = time.time()
        while time.time() - start_time < healthcheck_timeout:
            res = container.exec(["redis-cli", "ping"])
            if res.exit_code == 0:
                return True
            time.sleep(1.0)
        return False

    if not is_healthy():
        raise TimeoutError(f"Redis container not healthy after {healthcheck_timeout} seconds")

    container_host = container.get_container_host_ip()
    container_port = container.get_exposed_port(redis_port)
    logger.info("Redis container started", extra={"redis_host": container_host, "redis_port": container_port})
    yield container_host, container_port
    logger.info("Stopping Redis container...")
    container.stop()
