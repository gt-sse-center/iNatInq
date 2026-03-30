"""Fixtures for E2E tests.

Session-scoped fixtures manage Docker Compose lifecycle:
- Detect if stack is already running (skip up if so)
- Start stack with EMBEDDING_PROVIDER=clip
- Wait for all services to be healthy
- Clean up leftover test data from previous runs
- Tear down stack on session end (if we started it)

Test ordering: tests that use the ``ingested_collection`` session fixture run
before tests that submit their own Ray jobs (e.g. DLQ tests).  This prevents
memory pressure from accumulated Ray client processes on the head node.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from tests.e2e.helpers import api_url, minio_url, qdrant_url, wait_for_service

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = "zarf/compose/dev/docker-compose.yaml"

# MinIO dev credentials (default for Docker Compose)
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

# Service health endpoints
HEALTH_ENDPOINTS = {
    "pipeline": lambda: api_url("/healthz"),
    "qdrant": lambda: f"{qdrant_url()}/readyz",
    "minio": lambda: f"{minio_url()}/minio/health/live",
    "ray": "http://localhost:8265/",
    "clip": "http://localhost:8001/",
}


def _check_service_health(service: str, url: str, timeout: float = 2.0) -> bool:
    """Check if a service is healthy.

    Args:
        service: Service name (for logging).
        url: Health check URL.
        timeout: Request timeout in seconds.

    Returns:
        True if service responds 200, False otherwise.
    """
    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code == 200
    except httpx.RequestError:
        return False


def _check_redis_health() -> bool:
    """Check if Redis is healthy via docker exec.

    Returns:
        True if Redis responds to PING, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "exec", "redis", "redis-cli", "ping"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return result.returncode == 0 and "PONG" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _check_minio_bucket() -> bool:
    """Check if MinIO pipeline bucket exists.

    Returns:
        True if bucket exists, False otherwise.
    """
    try:
        response = httpx.head(f"{minio_url()}/pipeline", timeout=2.0)
        # 200 or 403 both indicate bucket exists (403 = exists but no anonymous read)
        return response.status_code in {200, 403}
    except httpx.RequestError:
        return False


def _all_services_healthy() -> bool:
    """Check if all required services are healthy.

    Returns:
        True if all services are healthy, False otherwise.
    """
    for service, url_or_fn in HEALTH_ENDPOINTS.items():
        url = url_or_fn() if callable(url_or_fn) else url_or_fn
        if not _check_service_health(service, url, timeout=2.0):
            logger.debug("Service %s not healthy at %s", service, url)
            return False

    if not _check_redis_health():
        logger.debug("Redis not healthy")
        return False

    if not _check_minio_bucket():
        logger.debug("MinIO pipeline bucket not ready")
        return False

    return True


def _wait_for_all_services(timeout: int = 120) -> None:
    """Wait for all services to become healthy.

    Args:
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If any service doesn't become healthy within timeout.
    """
    logger.info("Waiting for all services to be healthy (timeout=%ds)...", timeout)

    # Check Pipeline API
    wait_for_service(api_url("/healthz"), timeout=timeout)
    logger.info("✓ Pipeline API healthy")

    # Check Qdrant
    wait_for_service(f"{qdrant_url()}/readyz", timeout=timeout)
    logger.info("✓ Qdrant healthy")

    # Check MinIO
    wait_for_service(f"{minio_url()}/minio/health/live", timeout=timeout)
    logger.info("✓ MinIO healthy")

    # Check MinIO bucket (may take a few extra seconds after MinIO is healthy)
    start = time.time()
    while time.time() - start < 30:
        if _check_minio_bucket():
            logger.info("✓ MinIO pipeline bucket exists")
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("MinIO pipeline bucket not ready after 30s")

    # Check Ray
    wait_for_service("http://localhost:8265/", timeout=timeout)
    logger.info("✓ Ray head healthy")

    # Check CLIP
    wait_for_service("http://localhost:8001/", timeout=timeout)
    logger.info("✓ CLIP service healthy")

    # Check Redis
    start = time.time()
    while time.time() - start < 30:
        if _check_redis_health():
            logger.info("✓ Redis healthy")
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("Redis not healthy after 30s")


def _cleanup_test_data() -> None:
    """Clean up leftover test data from previous E2E runs.

    Deletes test collections in Qdrant and test prefixes in MinIO
    that match the ``e2e_test_*`` / ``e2e_dlq_test_*`` naming convention.
    """
    logger.info("Cleaning up leftover test data...")

    # Clean up Qdrant collections matching e2e_test_* or e2e_dlq_test_*
    try:
        from qdrant_client import QdrantClient

        qdrant = QdrantClient(url="http://localhost:6333")
        collections = qdrant.get_collections().collections
        for col in collections:
            if col.name.startswith("e2e_test_") or col.name.startswith("e2e_dlq_test_"):
                qdrant.delete_collection(collection_name=col.name)
                logger.info("Deleted leftover Qdrant collection: %s", col.name)
    except Exception as e:
        logger.warning("Failed to clean up Qdrant collections: %s", e)

    # Clean up MinIO objects matching e2e_test_* or e2e_dlq_test_* prefixes
    try:
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        for prefix in ("e2e_test_", "e2e_dlq_test_"):
            resp = s3.list_objects_v2(Bucket="pipeline", Prefix=prefix)
            if "Contents" in resp:
                for obj in resp["Contents"]:
                    s3.delete_object(Bucket="pipeline", Key=obj["Key"])
                logger.info(
                    "Deleted %d leftover MinIO objects with prefix: %s", len(resp["Contents"]), prefix
                )
    except Exception as e:
        logger.warning("Failed to clean up MinIO objects: %s", e)

    logger.debug("Test data cleanup complete")


@pytest.fixture(scope="session")
def docker_compose_stack() -> Generator[None, None, None]:
    """Session-scoped fixture that manages Docker Compose lifecycle.

    Detects if stack is already running. If not, starts it with
    EMBEDDING_PROVIDER=clip, waits for health, and tears down on exit.

    Yields:
        None (stack is ready when fixture yields).

    Raises:
        pytest.skip: If Docker is not available or stack cannot be started.
    """
    # Quick check: is Docker available?
    try:
        subprocess.run(
            ["docker", "version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Docker not available — skipping E2E tests")

    # Check if stack is already running
    stack_was_running = _all_services_healthy()
    if stack_was_running:
        logger.info("Docker Compose stack already running — skipping up")
        _cleanup_test_data()
        yield
        # Don't tear down if we didn't start it
        logger.info("Leaving pre-existing stack running")
        return

    # Stack not running — start it
    logger.info("Starting Docker Compose stack with EMBEDDING_PROVIDER=clip...")

    env = {**os.environ, "EMBEDDING_PROVIDER": "clip"}
    compose_cmd = [
        "docker",
        "compose",
        "-f",
        str(REPO_ROOT / COMPOSE_FILE),
        "up",
        "-d",
    ]

    try:
        result = subprocess.run(
            compose_cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        logger.debug("docker compose up output:\n%s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("docker compose up failed:\n%s", e.stderr)
        pytest.skip(f"Failed to start Docker Compose stack: {e.stderr}")
    except subprocess.TimeoutExpired:
        pytest.skip("docker compose up timed out after 120s")

    # Wait for all services to be healthy
    try:
        _wait_for_all_services(timeout=120)
    except TimeoutError as e:
        logger.error("Services did not become healthy: %s", e)
        # Try to clean up
        subprocess.run(
            ["docker", "compose", "-f", str(REPO_ROOT / COMPOSE_FILE), "down", "--timeout", "30"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        pytest.skip(f"Docker Compose stack not healthy: {e}")

    _cleanup_test_data()

    logger.info("Docker Compose stack ready for E2E tests")
    yield

    # Teardown: stop the stack if we started it
    logger.info("Tearing down Docker Compose stack...")
    try:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(REPO_ROOT / COMPOSE_FILE),
                "down",
                "--timeout",
                "30",
                "--remove-orphans",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        logger.info("Docker Compose stack stopped")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("docker compose down failed or timed out: %s", e)


@pytest.fixture(scope="session")
def ingested_collection(docker_compose_stack: None) -> Generator[str, None, None]:
    """Session-scoped fixture that ingests synthetic images once for all E2E tests.

    Generates 5 synthetic images, uploads to MinIO, submits Ray ingestion job,
    polls until completion, and verifies vectors exist in Qdrant.

    Yields:
        Collection name (e.g., "e2e_test_a1b2c3d4") for use in search/metrics tests.

    Cleanup:
        Deletes collection from Qdrant and test prefix from MinIO on session end.
    """
    import uuid
    from io import BytesIO

    import boto3
    from PIL import Image
    from qdrant_client import QdrantClient

    # Generate unique collection name
    collection_name = f"e2e_test_{uuid.uuid4().hex[:8]}"
    s3_prefix = f"{collection_name}/"

    logger.info("Setting up ingested_collection fixture: %s", collection_name)

    # 1. Generate 5 synthetic images programmatically
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128)]
    images = []
    for i, color in enumerate(colors):
        img = Image.new("RGB", (256, 256), color=color)
        images.append((f"image_{i}.png", img))

    # 2. Upload images to MinIO
    s3_client = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

    for filename, img in images:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        s3_client.put_object(
            Bucket="pipeline",
            Key=f"{s3_prefix}{filename}",
            Body=buffer,
        )
    logger.info("Uploaded %d synthetic images to MinIO prefix: %s", len(images), s3_prefix)

    # 3. Submit Ray ingestion job via API
    payload = {
        "s3_bucket": "pipeline",
        "s3_prefix": s3_prefix,
        "collection": collection_name,
    }

    response = httpx.post(
        api_url("/ray/jobs/images"),
        json=payload,
        timeout=10.0,
    )
    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]
    logger.info("Submitted Ray ingestion job: %s", job_id)

    # 4. Poll until job completes (180s timeout, exponential backoff)
    start = time.time()
    timeout = 180
    status = None
    poll_interval = 1.0

    while time.time() - start < timeout:
        status_response = httpx.get(
            api_url(f"/ray/jobs/{job_id}"),
            timeout=5.0,
        )
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data["status"]

        if status == "SUCCEEDED":
            logger.info("Ray job %s succeeded", job_id)
            break
        if status in {"FAILED", "STOPPED"}:
            raise RuntimeError(f"Ray job {job_id} failed with status: {status}")

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 2, 8.0)  # Cap at 8s

    if status != "SUCCEEDED":
        raise TimeoutError(f"Ray job {job_id} did not complete within {timeout}s (last status: {status})")

    # 5. Verify vectors exist in Qdrant
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection_info = qdrant_client.get_collection(collection_name=collection_name)

    if collection_info.points_count != 5:
        raise AssertionError(f"Expected 5 vectors in {collection_name}, got {collection_info.points_count}")

    logger.info("Verified %d vectors in Qdrant collection: %s", collection_info.points_count, collection_name)

    # 6. Warm up the search pipeline (embedding + Qdrant) so that
    #    connection pools and circuit breakers are stable before
    #    metric-sensitive tests run.
    warmup_response = httpx.get(
        api_url("/search/images"),
        params={"q": "warmup", "collection": collection_name, "limit": 1},
        timeout=10.0,
    )
    if warmup_response.status_code == 200:
        logger.info("Warmup search succeeded")
    else:
        logger.warning("Warmup search returned %d", warmup_response.status_code)

    # Yield collection name for tests to use
    yield collection_name

    # Cleanup: delete collection and S3 prefix
    logger.info("Cleaning up ingested_collection fixture: %s", collection_name)

    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        logger.info("Deleted Qdrant collection: %s", collection_name)
    except Exception as e:
        logger.warning("Failed to delete Qdrant collection %s: %s", collection_name, e)

    try:
        # List and delete all objects with the test prefix
        objects = s3_client.list_objects_v2(Bucket="pipeline", Prefix=s3_prefix)
        if "Contents" in objects:
            for obj in objects["Contents"]:
                s3_client.delete_object(Bucket="pipeline", Key=obj["Key"])
            logger.info("Deleted %d objects from MinIO prefix: %s", len(objects["Contents"]), s3_prefix)
    except Exception as e:
        logger.warning("Failed to delete MinIO prefix %s: %s", s3_prefix, e)


# ---------------------------------------------------------------------------
# Test ordering — ingested_collection tests first, DLQ tests last
# ---------------------------------------------------------------------------

# Priority: lower = runs earlier.  Tests using the ``ingested_collection``
# session fixture (search, metrics, resilience, ingestion) must run before
# DLQ tests, which submit their own Ray jobs and accumulate memory on the
# Ray head node.
_MODULE_PRIORITY = {
    "test_cli": 0,
    "test_ingestion": 1,
    "test_search": 2,
    "test_metrics": 3,
    "test_resilience": 4,
    "test_dlq": 5,
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Reorder E2E tests so session fixtures are initialised early."""
    items.sort(key=lambda item: _MODULE_PRIORITY.get(item.module.__name__.rsplit(".", 1)[-1], 99))
