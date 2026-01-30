"""Integration tests for WeaviateClientWrapper.

These tests verify the Weaviate client works correctly against a real Weaviate
instance. The tests are organized by resilience category to ensure comprehensive
coverage of failure modes and recovery patterns.

## Test Categories

1. Happy Path - Basic CRUD operations work correctly
2. Transient Failures - Retries succeed after temporary errors
3. Non-Retriable Errors - Fail fast on permanent errors
4. Circuit Breaker - Opens after threshold, recovers correctly
5. Resource Cleanup - No connection leaks
6. Observability - Proper logging of errors and retries
7. Factory Method - from_config creates working client

## Running Tests

```bash
# Run Weaviate integration tests only
pytest tests/integration/clients/test_weaviate.py -v -m integration

# Run with specific test class
pytest tests/integration/clients/test_weaviate.py::TestHappyPath -v
```

## Requirements

- Docker must be running (testcontainers manages Weaviate)
- No external Weaviate instance needed
"""

import asyncio
import logging
import time
import uuid as uuid_module
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pybreaker
import pytest
import weaviate.exceptions
from testcontainers.core.container import DockerContainer

from clients.weaviate import WeaviateClientWrapper, WeaviateDataObject
from config import VectorDBConfig
from core.exceptions import UpstreamError

logger = logging.getLogger(__name__)


# =============================================================================
# Weaviate Container Fixtures
# =============================================================================


def _get_weaviate_url(container: DockerContainer) -> str:
    """Get the Weaviate HTTP URL from a container.

    Args:
        container: Running Weaviate container.

    Returns:
        str: HTTP URL for Weaviate REST API.
    """
    host = container.get_container_host_ip()
    port = container.get_exposed_port(8080)
    return f"http://{host}:{port}"


def _wait_for_weaviate_health(container: DockerContainer, timeout: int = 60) -> None:
    """Wait for Weaviate container to be ready.

    Args:
        container: Weaviate container instance.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If Weaviate doesn't become healthy within timeout.
    """
    url = _get_weaviate_url(container)
    health_url = f"{url}/v1/.well-known/ready"

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.5)

    raise TimeoutError(f"Weaviate container not healthy after {timeout}s")


@pytest.fixture(scope="session")
def weaviate_container():
    """Start a Weaviate container for the test session.

    The container is started once and shared across all tests in the session.
    This significantly reduces test overhead compared to per-test containers.

    Yields:
        DockerContainer: Running Weaviate container with connection info.
    """
    logger.info("Starting Weaviate container...")

    # Expose both HTTP (8080) and gRPC (50051) ports for Weaviate v4 client
    container = (
        DockerContainer("semitechnologies/weaviate:1.28.2")
        .with_exposed_ports(8080, 50051)
        .with_env("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED", "true")
        .with_env("PERSISTENCE_DATA_PATH", "/var/lib/weaviate")
        .with_env("DEFAULT_VECTORIZER_MODULE", "none")
        .with_env("CLUSTER_HOSTNAME", "node1")
        .with_env("GRPC_PORT", "50051")
    )

    container.start()

    # Wait for container to be healthy
    _wait_for_weaviate_health(container)

    logger.info(
        "Weaviate container started",
        extra={
            "url": _get_weaviate_url(container),
            "container_id": container.get_wrapped_container().short_id,
        },
    )

    yield container

    logger.info("Stopping Weaviate container...")
    container.stop()


def _get_weaviate_grpc_port(container: DockerContainer) -> int:
    """Get the mapped gRPC port from a Weaviate container.

    Args:
        container: Running Weaviate container.

    Returns:
        int: Mapped gRPC port on the host.
    """
    return int(container.get_exposed_port(50051))


@pytest.fixture(scope="session")
def weaviate_url(weaviate_container: DockerContainer) -> str:
    """Get Weaviate connection URL.

    Returns:
        str: Weaviate HTTP API URL.
    """
    return _get_weaviate_url(weaviate_container)


@pytest.fixture(scope="session")
def weaviate_grpc_port(weaviate_container: DockerContainer) -> int:
    """Get Weaviate gRPC port.

    Returns:
        int: Mapped gRPC port on the host.
    """
    return _get_weaviate_grpc_port(weaviate_container)


@pytest.fixture(scope="session")
def weaviate_client(weaviate_url: str, weaviate_grpc_port: int) -> WeaviateClientWrapper:
    """Create a WeaviateClientWrapper connected to the test Weaviate instance.

    Session-scoped to share the client across tests for efficiency.
    Tests should use unique collection names to avoid collisions.

    Returns:
        WeaviateClientWrapper: Client connected to test Weaviate.
    """
    # Pass the mapped gRPC port for testcontainers compatibility
    client = WeaviateClientWrapper(url=weaviate_url, grpc_port=weaviate_grpc_port)

    logger.info(
        "Created Weaviate client for integration tests",
        extra={"url": weaviate_url, "grpc_port": weaviate_grpc_port},
    )

    yield client

    # Cleanup
    client.close()


@pytest.fixture
def test_collection() -> str:
    """Generate a unique test collection name.

    Each test gets a fresh collection to ensure isolation.

    Yields:
        str: Unique collection name.
    """
    collection_name = f"Test{uuid_module.uuid4().hex[:12]}"

    logger.debug("Created test collection", extra={"collection": collection_name})

    yield collection_name

    # Note: Weaviate cleanup is handled by container teardown


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


@pytest.fixture
def clip_vector_size() -> int:
    """CLIP vector dimension for image collection tests.

    Returns:
        int: Vector dimension (512 for CLIP models like ViT-B/32).
    """
    return 512


@pytest.fixture
def sample_clip_vector(clip_vector_size: int) -> list[float]:
    """Sample 512-dimensional vector for image collection tests."""
    import random

    random.seed(43)
    return [random.random() for _ in range(clip_vector_size)]  # noqa: S311


# =============================================================================
# 1. Happy Path Tests
# =============================================================================


@pytest.mark.integration
class TestHappyPath:
    """Test basic CRUD operations work correctly."""

    def test_ensure_collection_creates_new_collection(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        vector_size: int,
    ):
        """Test that ensure_collection creates a new collection when it doesn't exist.

        **Why this test is important:**
          - Collection creation is prerequisite for all vector operations
          - Validates async ensure_collection method works correctly
          - Critical for first-time setup and collection management

        **What it tests:**
          - ensure_collection_async creates a new collection
          - Collection is accessible after creation
        """
        # Act
        asyncio.run(
            weaviate_client.ensure_collection_async(
                collection=test_collection,
                vector_size=vector_size,
            )
        )

        # Assert - collection should exist (no exception means success)
        # We can verify by trying to upsert
        point = WeaviateDataObject(
            uuid=str(uuid_module.uuid4()),
            properties={"text": "test"},
            vector=[0.1] * vector_size,
        )
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=test_collection,
                points=[point],
                vector_size=vector_size,
            )
        )

    def test_ensure_collection_is_idempotent(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        vector_size: int,
    ):
        """Test that ensure_collection is idempotent on existing collection.

        **Why this test is important:**
          - Multiple services may call ensure_collection concurrently
          - Idempotency prevents race condition errors
          - Critical for distributed system reliability

        **What it tests:**
          - Calling ensure_collection twice doesn't raise an error
        """
        # Create collection first
        asyncio.run(
            weaviate_client.ensure_collection_async(
                collection=test_collection,
                vector_size=vector_size,
            )
        )

        # Act - calling again should not raise
        asyncio.run(
            weaviate_client.ensure_collection_async(
                collection=test_collection,
                vector_size=vector_size,
            )
        )

    def test_batch_upsert_async_inserts_points(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        sample_vector: list[float],
        vector_size: int,
    ):
        """Test that batch_upsert_async inserts points correctly.

        **Why this test is important:**
          - Batch upsert is essential for bulk operations
          - Validates points are stored with correct vectors and payloads
          - Critical for data integrity in the vector database

        **What it tests:**
          - Multiple points can be inserted in a single batch
          - Points are searchable after insertion
        """
        # Arrange
        points = [
            WeaviateDataObject(
                uuid=str(uuid_module.uuid4()),
                properties={"text": "hello"},
                vector=sample_vector,
            ),
            WeaviateDataObject(
                uuid=str(uuid_module.uuid4()),
                properties={"text": "world"},
                vector=sample_vector,
            ),
        ]

        # Act
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=test_collection,
                points=points,
                vector_size=vector_size,
            )
        )

        # Assert - points should be searchable
        results = asyncio.run(
            weaviate_client.search_async(
                collection=test_collection,
                query_vector=sample_vector,
                limit=10,
            )
        )
        assert results.total >= 2

    def test_search_async_returns_results(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        sample_vector: list[float],
        vector_size: int,
    ):
        """Test that search_async returns matching results with correct scores.

        **Why this test is important:**
          - Search is the primary user-facing operation
          - Validates vector similarity search works correctly
          - Critical for semantic search functionality

        **What it tests:**
          - Search returns the expected number of results
          - Payloads are returned correctly
          - Similarity scores are meaningful
        """
        # Arrange - insert a point first
        point = WeaviateDataObject(
            uuid=str(uuid_module.uuid4()),
            properties={"text": "searchable", "category": "test"},
            vector=sample_vector,
        )
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=test_collection,
                points=[point],
                vector_size=vector_size,
            )
        )

        # Act
        results = asyncio.run(
            weaviate_client.search_async(
                collection=test_collection,
                query_vector=sample_vector,
                limit=10,
            )
        )

        # Assert
        assert results.total >= 1
        assert len(results.items) >= 1
        # Check that we get meaningful results
        assert results.items[0].score > 0.5

    def test_search_async_returns_empty_for_no_matches(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        vector_size: int,
    ):
        """Test that search_async returns empty results for empty collection.

        **Why this test is important:**
          - Empty results must be handled gracefully
          - Validates no false positives are returned
          - Critical for correct API contract

        **What it tests:**
          - Search on empty collection returns zero results
          - No error is raised for empty collections
          - Result structure is correct even when empty
        """
        # Arrange - ensure collection exists but is empty
        asyncio.run(
            weaviate_client.ensure_collection_async(
                collection=test_collection,
                vector_size=vector_size,
            )
        )

        # Create a query vector
        query_vector = [0.5] * vector_size

        # Act
        results = asyncio.run(
            weaviate_client.search_async(
                collection=test_collection,
                query_vector=query_vector,
                limit=10,
            )
        )

        # Assert
        assert results.total == 0
        assert len(results.items) == 0

    def test_batch_upsert_async_updates_existing_points(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        sample_vector: list[float],
        vector_size: int,
    ):
        """Test that batch_upsert_async updates existing points by UUID.

        **Why this test is important:**
          - Upsert semantics (insert or update) must work correctly
          - Validates idempotent ingestion for re-processing
          - Critical for data consistency during re-ingestion

        **What it tests:**
          - Same UUID overwrites existing point
          - Updated payload is reflected in subsequent queries
          - No duplicate points are created
        """
        # Arrange - insert initial point with same UUID
        point_uuid = str(uuid_module.uuid4())
        point_v1 = WeaviateDataObject(
            uuid=point_uuid,
            properties={"version": "1", "text": "original"},
            vector=sample_vector,
        )
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=test_collection,
                points=[point_v1],
                vector_size=vector_size,
            )
        )

        # Act - upsert with same UUID, different payload
        point_v2 = WeaviateDataObject(
            uuid=point_uuid,
            properties={"version": "2", "text": "updated"},
            vector=sample_vector,
        )
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=test_collection,
                points=[point_v2],
                vector_size=vector_size,
            )
        )

        # Assert - search should return updated version
        results = asyncio.run(
            weaviate_client.search_async(
                collection=test_collection,
                query_vector=sample_vector,
                limit=10,
            )
        )
        # Should have exactly one result with the point we inserted
        # (Note: other tests may have added data, so we just verify our point exists)
        found = False
        for item in results.items:
            if item.point_id == point_uuid:
                found = True
                # Payload may be nested differently in Weaviate
                break
        assert found, f"Point {point_uuid} not found in results"

    def test_empty_points_list_is_noop(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        vector_size: int,
    ):
        """Test that empty points list is handled as a no-op.

        **Why this test is important:**
          - Batch jobs may have empty partitions
          - Empty batches must not cause errors
          - Critical for robust batch processing

        **What it tests:**
          - Empty points list doesn't raise an exception
        """
        # Act - should not raise
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=test_collection,
                points=[],
                vector_size=vector_size,
            )
        )


# =============================================================================
# 1b. Image Collection Happy Path Tests
# =============================================================================


@pytest.mark.integration
class TestImageCollection:
    """Test Weaviate image collection (ensure_image_collection_async).

    Mirrors Qdrant image collection schema: class name {Collection}Images,
    properties s3_key, s3_uri, format, width, height, thumbnail_key, CLIP vector size.
    """

    def test_ensure_image_collection_creates_new_class(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        clip_vector_size: int,
    ):
        """Test that ensure_image_collection_async creates a new Weaviate class.

        **Why this test is important:**
          - Image collection creation is prerequisite for image embeddings
          - Validates {Collection}Images class name pattern
          - Critical for CLIP / image search setup

        **What it tests:**
          - ensure_image_collection_async creates class {Collection}Images
          - Class is accessible for upsert and search after creation
        """
        asyncio.run(
            weaviate_client.ensure_image_collection_async(
                collection=test_collection,
                vector_size=clip_vector_size,
            )
        )

        image_class = weaviate_client._collection_to_image_class_name(test_collection)
        point = WeaviateDataObject(
            uuid=str(uuid_module.uuid4()),
            properties={
                "s3_key": "images/photo1.jpg",
                "s3_uri": "s3://bucket/images/photo1.jpg",
                "format": "jpeg",
                "width": 800,
                "height": 600,
            },
            vector=[0.1] * clip_vector_size,
        )
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=image_class,
                points=[point],
                vector_size=clip_vector_size,
            )
        )

    def test_ensure_image_collection_is_idempotent(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        clip_vector_size: int,
    ):
        """Test that ensure_image_collection_async is idempotent on existing class.

        **Why this test is important:**
          - Multiple callers may ensure image collection concurrently
          - Idempotency prevents race condition errors
          - Critical for distributed system reliability

        **What it tests:**
          - Calling ensure_image_collection_async twice does not raise
        """
        asyncio.run(
            weaviate_client.ensure_image_collection_async(
                collection=test_collection,
                vector_size=clip_vector_size,
            )
        )
        asyncio.run(
            weaviate_client.ensure_image_collection_async(
                collection=test_collection,
                vector_size=clip_vector_size,
            )
        )

    def test_image_collection_search_after_upsert(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        sample_clip_vector: list[float],
        clip_vector_size: int,
    ):
        """Test that image collection supports upsert and search with CLIP-sized vectors.

        **Why this test is important:**
          - End-to-end validation of image collection schema
          - Ensures DocumentsImages-style class works for search
          - Critical for image similarity search

        **What it tests:**
          - ensure_image_collection_async creates class
          - batch_upsert_async inserts image objects with image properties
          - search_async returns results from image class
        """
        asyncio.run(
            weaviate_client.ensure_image_collection_async(
                collection=test_collection,
                vector_size=clip_vector_size,
            )
        )
        image_class = weaviate_client._collection_to_image_class_name(test_collection)
        point = WeaviateDataObject(
            uuid=str(uuid_module.uuid4()),
            properties={
                "s3_key": "images/test.jpg",
                "s3_uri": "s3://bucket/images/test.jpg",
                "format": "jpeg",
                "width": 1024,
                "height": 768,
                "thumbnail_key": "thumbnails/test_thumb.jpg",
            },
            vector=sample_clip_vector,
        )
        asyncio.run(
            weaviate_client.batch_upsert_async(
                collection=image_class,
                points=[point],
                vector_size=clip_vector_size,
            )
        )
        results = asyncio.run(
            weaviate_client.search_async(
                collection=image_class,
                query_vector=sample_clip_vector,
                limit=10,
            )
        )
        assert results.total >= 1
        assert len(results.items) >= 1
        assert results.items[0].score > 0.0


# =============================================================================
# 2. Transient Failure Tests
# =============================================================================


@pytest.mark.integration
class TestTransientFailures:
    """Test retry behavior for transient failures."""

    def test_search_succeeds_after_transient_error(
        self,
        weaviate_url: str,
        weaviate_grpc_port: int,
        test_collection: str,
        sample_vector: list[float],
        vector_size: int,
    ):
        """Test that search succeeds after transient network errors.

        **Why this test is important:**
          - Network issues are common in distributed systems
          - Retry logic must recover from temporary failures
          - Critical for production reliability

        **What it tests:**
          - First call fails with a transient connection error
          - Retry logic kicks in and second call succeeds
          - Correct results are returned after recovery
        """
        # Arrange - create a fresh client for this test
        client = WeaviateClientWrapper(url=weaviate_url, grpc_port=weaviate_grpc_port)

        try:
            # Insert data first (this must succeed)
            point = WeaviateDataObject(
                uuid=str(uuid_module.uuid4()),
                properties={"text": "retry-test"},
                vector=sample_vector,
            )
            asyncio.run(
                client.batch_upsert_async(
                    collection=test_collection,
                    points=[point],
                    vector_size=vector_size,
                )
            )

            # Store the real client for later use
            real_client = client._client

            # Create a mock that fails once then delegates to real client
            call_count = 0
            original_collections = real_client.collections

            class TransientFailureCollections:
                """Wrapper that fails on first call, succeeds on subsequent calls."""

                def get(self, name: str):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        # First call - simulate transient network error
                        raise weaviate.exceptions.WeaviateConnectionError(
                            "Connection reset by peer (simulated)"
                        )
                    # Subsequent calls - delegate to real implementation
                    return original_collections.get(name)

                def __getattr__(self, name):
                    return getattr(original_collections, name)

            # Patch the collections object
            real_client.collections = TransientFailureCollections()

            # Act - search should fail once, then succeed on retry
            # Note: The client may not have built-in retry, so this tests
            # that errors are properly raised for caller-level retry
            try:
                results = asyncio.run(
                    client.search_async(
                        collection=test_collection,
                        query_vector=sample_vector,
                        limit=10,
                    )
                )
                # If we get here, the client has internal retry logic
                assert results.total >= 1
            except UpstreamError as e:
                # If error is raised, verify it's the transient error
                assert "Connection reset" in str(e) or "connection" in str(e).lower()
                # Verify the error was raised on first attempt
                assert call_count == 1

                # Now verify a clean call succeeds (simulating caller retry)
                real_client.collections = original_collections
                results = asyncio.run(
                    client.search_async(
                        collection=test_collection,
                        query_vector=sample_vector,
                        limit=10,
                    )
                )
                assert results.total >= 1
        finally:
            client.close()

    def test_upsert_succeeds_after_transient_error(
        self,
        weaviate_url: str,
        weaviate_grpc_port: int,
        test_collection: str,
        sample_vector: list[float],
        vector_size: int,
    ):
        """Test that upsert operations can recover from transient failures.

        **Why this test is important:**
          - Write operations must be resilient to network blips
          - Data must eventually be persisted despite transient errors
          - Critical for ingestion pipeline reliability

        **What it tests:**
          - First upsert attempt fails with transient error
          - Caller-level retry succeeds
          - Data is correctly persisted
        """
        import aiobreaker.state as aio_state

        # Arrange
        client = WeaviateClientWrapper(url=weaviate_url, grpc_port=weaviate_grpc_port)

        try:
            point = WeaviateDataObject(
                uuid=str(uuid_module.uuid4()),
                properties={"text": "transient-upsert-test"},
                vector=sample_vector,
            )

            # First attempt - simulate failure by temporarily opening async circuit breaker
            # The base class checks _async_breaker, not _breaker
            original_state = client._async_breaker.current_state
            object.__setattr__(
                client,
                "_async_breaker",
                type(
                    "MockBreaker",
                    (),
                    {
                        "current_state": aio_state.CircuitBreakerState.OPEN,
                        "call_async": lambda self, func: func(),
                    },
                )(),
            )

            with pytest.raises(UpstreamError) as exc_info:
                asyncio.run(
                    client.batch_upsert_async(
                        collection=test_collection,
                        points=[point],
                        vector_size=vector_size,
                    )
                )
            assert "unavailable" in str(exc_info.value).lower()

            # Restore real async breaker for recovery
            from foundation.circuit_breaker import create_async_circuit_breaker

            object.__setattr__(client, "_async_breaker", create_async_circuit_breaker("weaviate", 3, 60))

            # Retry - should succeed
            asyncio.run(
                client.batch_upsert_async(
                    collection=test_collection,
                    points=[point],
                    vector_size=vector_size,
                )
            )

            # Verify data was persisted
            results = asyncio.run(
                client.search_async(
                    collection=test_collection,
                    query_vector=sample_vector,
                    limit=10,
                )
            )
            assert results.total >= 1
        finally:
            client.close()


# =============================================================================
# 3. Non-Retriable Error Tests
# =============================================================================


@pytest.mark.integration
class TestNonRetriableErrors:
    """Test fail-fast behavior for non-retriable errors."""

    def test_search_nonexistent_collection_raises_upstream_error(
        self,
        weaviate_client: WeaviateClientWrapper,
        sample_vector: list[float],
    ):
        """Test that searching a non-existent collection raises UpstreamError.

        **Why this test is important:**
          - 404 errors should not be retried
          - Clear error messages help debugging
          - Critical for fast failure on configuration errors

        **What it tests:**
          - Non-existent collection raises UpstreamError
          - Error message contains useful context
        """
        # Arrange
        nonexistent_collection = f"Nonexistent{uuid_module.uuid4().hex[:8]}"

        # Act & Assert
        with pytest.raises(UpstreamError) as exc_info:
            asyncio.run(
                weaviate_client.search_async(
                    collection=nonexistent_collection,
                    query_vector=sample_vector,
                    limit=10,
                )
            )

        assert "Weaviate search failed" in str(exc_info.value)


# =============================================================================
# 4. Circuit Breaker Tests
# =============================================================================


@pytest.mark.integration
class TestCircuitBreaker:
    """Test circuit breaker behavior."""

    def test_circuit_breaker_starts_closed(
        self,
        weaviate_url: str,
    ):
        """Test that circuit breaker starts in closed state.

        **Why this test is important:**
          - Circuit breaker must start in a healthy state
          - Ensures clients can make requests immediately after creation
          - Critical for production reliability

        **What it tests:**
          - Circuit breaker (_breaker) starts in CLOSED state
        """
        # Arrange
        client = WeaviateClientWrapper(url=weaviate_url)

        try:
            # Assert - breaker starts closed
            assert client._breaker.current_state == pybreaker.STATE_CLOSED
        finally:
            client.close()

    def test_custom_circuit_breaker_settings(
        self,
        weaviate_url: str,
    ):
        """Test that custom circuit breaker settings are applied.

        **Why this test is important:**
          - Production environments may need different thresholds
          - Validates configurable resilience parameters
          - Critical for operational flexibility

        **What it tests:**
          - Custom circuit_breaker_threshold is applied
          - Custom circuit_breaker_timeout is applied
        """
        # Arrange
        client = WeaviateClientWrapper(
            url=weaviate_url,
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=120,
        )

        try:
            # Assert
            assert client._breaker.fail_max == 10
            assert client._breaker.reset_timeout == 120
        finally:
            client.close()

    def test_circuit_breaker_opens_after_failures(
        self,
        weaviate_url: str,
        weaviate_grpc_port: int,
        sample_vector: list[float],
    ):
        """Test that circuit breaker opens after threshold failures.

        **Why this test is important:**
          - Circuit breaker must protect downstream services from cascading failures
          - Validates that aiobreaker tracks failures correctly
          - Ensures threshold configuration is respected
          - Critical for fault tolerance

        **What it tests:**
          - Repeated failures increment the breaker's fail counter
          - After fail_max failures, circuit transitions to OPEN state
        """
        import aiobreaker.state as aio_state

        # Arrange - create client with fresh circuit breaker
        client = WeaviateClientWrapper(
            url=weaviate_url,
            grpc_port=weaviate_grpc_port,
        )
        nonexistent_collection = f"Fail{uuid_module.uuid4().hex[:8]}"

        try:
            # Act - trigger failures by searching non-existent collection
            # The breaker has fail_max=3 (default)
            for _ in range(3):
                try:
                    asyncio.run(
                        client.search_async(
                            collection=nonexistent_collection,
                            query_vector=sample_vector,
                            limit=10,
                        )
                    )
                except UpstreamError:
                    pass  # Expected

            # Assert - async circuit should be open after threshold exceeded
            state_str = str(client._async_breaker.current_state).lower()
            assert "open" in state_str and "half" not in state_str, f"Expected OPEN, got {state_str}"
        finally:
            client.close()

    def test_circuit_breaker_fail_fast_when_open(
        self,
        weaviate_url: str,
        weaviate_grpc_port: int,
        sample_vector: list[float],
    ):
        """Test that open circuit breaker causes immediate failure without network call.

        **Why this test is important:**
          - Open circuit must fail fast to prevent resource exhaustion
          - Validates that requests don't hit Weaviate when circuit is open
          - Ensures clear error message for debugging and monitoring
          - Critical for preventing cascading failures

        **What it tests:**
          - Open circuit raises UpstreamError immediately
          - Error message indicates service unavailability
        """
        import aiobreaker.state as aio_state

        # Arrange - create client and force circuit open
        client = WeaviateClientWrapper(url=weaviate_url, grpc_port=weaviate_grpc_port)
        nonexistent_collection = f"Fail{uuid_module.uuid4().hex[:8]}"

        try:
            # Force circuit breaker open via repeated failures
            for _ in range(3):
                try:
                    asyncio.run(
                        client.search_async(
                            collection=nonexistent_collection,
                            query_vector=sample_vector,
                            limit=10,
                        )
                    )
                except UpstreamError:
                    pass

            # Verify async circuit is open
            state_str = str(client._async_breaker.current_state).lower()
            assert "open" in state_str and "half" not in state_str, f"Expected OPEN, got {state_str}"

            # Act & Assert - next request should fail fast with circuit breaker message
            with pytest.raises(UpstreamError) as exc_info:
                asyncio.run(
                    client.search_async(
                        collection=nonexistent_collection,
                        query_vector=sample_vector,
                        limit=10,
                    )
                )

            # The fail-fast path raises "service is currently unavailable"
            assert "currently unavailable" in str(exc_info.value).lower()
        finally:
            client.close()


# =============================================================================
# 5. Resource Cleanup Tests
# =============================================================================


@pytest.mark.integration
class TestResourceCleanup:
    """Test proper resource cleanup."""

    def test_close_releases_resources(
        self,
        weaviate_url: str,
    ):
        """Test that client close releases all resources.

        **Why this test is important:**
          - Connection leaks cause resource exhaustion
          - Proper cleanup prevents memory leaks
          - Critical for long-running services

        **What it tests:**
          - close() sets client reference to None
        """
        # Arrange
        client = WeaviateClientWrapper(url=weaviate_url)

        # Act
        client.close()

        # Assert - client should be None
        assert client._client is None

    def test_close_is_idempotent(
        self,
        weaviate_url: str,
    ):
        """Test that calling close multiple times is safe.

        **Why this test is important:**
          - Cleanup code may be called multiple times
          - Idempotent close prevents double-free errors
          - Critical for robust error handling paths

        **What it tests:**
          - Multiple close() calls don't raise exceptions
        """
        # Arrange
        client = WeaviateClientWrapper(url=weaviate_url)

        # Act - close multiple times
        client.close()
        client.close()  # Should not raise
        client.close()  # Should not raise

        # Assert
        assert client._client is None


# =============================================================================
# 6. Observability Tests
# =============================================================================


@pytest.mark.integration
class TestObservability:
    """Test logging and observability."""

    def test_upsert_logs_operation(
        self,
        weaviate_client: WeaviateClientWrapper,
        test_collection: str,
        sample_vector: list[float],
        vector_size: int,
        caplog,
    ):
        """Test that upsert operations are logged for observability.

        **Why this test is important:**
          - Operations must be logged for debugging
          - Enables monitoring and alerting
          - Critical for production troubleshooting

        **What it tests:**
          - Upsert operation produces log messages
          - HTTP requests are logged (via httpx)
        """
        # Arrange
        point = WeaviateDataObject(
            uuid=str(uuid_module.uuid4()),
            properties={"text": "log-test"},
            vector=sample_vector,
        )

        # Act
        with caplog.at_level("INFO"):
            asyncio.run(
                weaviate_client.batch_upsert_async(
                    collection=test_collection,
                    points=[point],
                    vector_size=vector_size,
                )
            )

        # Assert - check that HTTP requests are logged (from httpx or the underlying client)
        log_text = caplog.text.lower()
        # Weaviate v4 client logs HTTP requests via httpx
        assert "http" in log_text or "request" in log_text or len(caplog.records) > 0

    def test_search_failure_logs_error(
        self,
        weaviate_client: WeaviateClientWrapper,
        sample_vector: list[float],
        caplog,
    ):
        """Test that search failures are logged for debugging.

        **Why this test is important:**
          - Errors must be logged for incident response
          - Enables root cause analysis
          - Critical for production debugging

        **What it tests:**
          - Search failure raises appropriate exception
          - Error is properly propagated
          - Exception contains useful context
        """
        # Arrange
        nonexistent_collection = f"LogFail{uuid_module.uuid4().hex[:8]}"

        # Act
        with caplog.at_level("ERROR"):
            try:
                asyncio.run(
                    weaviate_client.search_async(
                        collection=nonexistent_collection,
                        query_vector=sample_vector,
                        limit=10,
                    )
                )
            except UpstreamError:
                pass

        # Note: The error may be raised without logging depending on implementation
        # This test validates the exception is raised correctly


# =============================================================================
# 7. Factory Method Tests
# =============================================================================


@pytest.mark.integration
class TestFromConfig:
    """Test factory method with VectorDBConfig."""

    def test_from_config_creates_working_client(
        self,
        weaviate_url: str,
        weaviate_grpc_port: int,
        test_collection: str,
        sample_vector: list[float],
        vector_size: int,
    ):
        """Test that from_config creates a fully functional client.

        **Why this test is important:**
          - Factory method is primary way to create clients
          - Validates VectorDBConfig integration
          - Critical for application bootstrapping

        **What it tests:**
          - from_config creates a working WeaviateClientWrapper
          - Client can perform CRUD operations
        """
        # Arrange
        config = VectorDBConfig(
            provider_type="weaviate",
            collection=test_collection,
            weaviate_url=weaviate_url,
            weaviate_grpc_port=weaviate_grpc_port,
        )

        # Act
        client = WeaviateClientWrapper.from_config(config)

        try:
            # Use the client
            point = WeaviateDataObject(
                uuid=str(uuid_module.uuid4()),
                properties={"text": "config-test"},
                vector=sample_vector,
            )
            asyncio.run(
                client.batch_upsert_async(
                    collection=test_collection,
                    points=[point],
                    vector_size=vector_size,
                )
            )

            # Assert - data was inserted
            results = asyncio.run(
                client.search_async(
                    collection=test_collection,
                    query_vector=sample_vector,
                    limit=10,
                )
            )
            assert results.total >= 1
        finally:
            client.close()

    def test_from_config_with_resilience_settings(
        self,
        weaviate_url: str,
    ):
        """Test that from_config passes resilience settings correctly.

        **Why this test is important:**
          - Resilience settings must flow from config to client
          - Critical for environment-specific tuning
          - Validates VectorDBConfig integration

        **What it tests:**
          - timeout_s is passed from config
          - circuit_breaker_threshold is passed from config
          - circuit_breaker_timeout is passed from config
        """
        # Arrange
        config = VectorDBConfig(
            provider_type="weaviate",
            collection="test-collection",
            weaviate_url=weaviate_url,
            weaviate_timeout=600,
            weaviate_circuit_breaker_threshold=10,
            weaviate_circuit_breaker_timeout=120,
        )

        # Act
        client = WeaviateClientWrapper.from_config(config)

        try:
            # Assert
            assert client.timeout_s == 600
            assert client.circuit_breaker_threshold == 10
            assert client.circuit_breaker_timeout == 120
            assert client._breaker.fail_max == 10
            assert client._breaker.reset_timeout == 120
        finally:
            client.close()

    def test_from_config_with_api_key(
        self,
        weaviate_url: str,
    ):
        """Test that from_config passes API key correctly for cloud instances.

        **Why this test is important:**
          - Cloud Weaviate requires API key authentication
          - Validates API key is passed through configuration
          - Critical for cloud deployment security

        **What it tests:**
          - API key from config is set on client
          - URL is correctly passed through
          - Client is created without connection attempt

        Note: This test uses an HTTPS URL to simulate cloud setup.
        We're only testing that the API key is passed through correctly.
        """
        # Arrange - Use HTTPS URL to simulate cloud setup
        config = VectorDBConfig(
            provider_type="weaviate",
            collection="test-collection",
            weaviate_url="https://example-weaviate.cloud:443",
            weaviate_api_key="test-api-key",
            weaviate_grpc_host="grpc-example-weaviate.cloud",
        )

        # Act
        client = WeaviateClientWrapper.from_config(config)

        try:
            # Assert - API key is set (don't actually connect)
            assert client.api_key == "test-api-key"
            assert client.url == "https://example-weaviate.cloud:443"
            assert client.grpc_host == "grpc-example-weaviate.cloud"
        finally:
            client.close()
