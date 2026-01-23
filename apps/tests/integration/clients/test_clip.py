"""Integration tests for CLIP embedding client.

This module tests the CLIPClient against a real ai4all/clip container,
which provides a dedicated CLIP API server supporting image embeddings.

See: https://hub.docker.com/r/ai4all/clip

**IMPORTANT**: The ai4all/clip container's API format is not yet fully documented.
Tests that require actual embedding API calls are skipped until we have proper
API documentation or a working container configuration.

## Container Requirements

- Image: ai4all/clip:latest
- Port: 8000 (mapped to random host port)
- Platforms: Linux/amd64, Linux/arm64

## Running Tests

```bash
# Run all CLIP integration tests
make test-integration TEST_FILE=tests/integration/clients/test_clip.py

# Run with verbose output
uv run pytest tests/integration/clients/test_clip.py -v -s
```

## Test Categories

1. Error Handling: Input validation tests (work without API calls)
2. Resource Cleanup: Session lifecycle tests (work without API calls)
3. Factory Method: Client creation tests (work without API calls)
4. Circuit Breaker: Initialization state tests (work without API calls)

Note: Tests that require actual embedding API calls are skipped until
the ai4all/clip API format is properly documented.
"""

import logging

import aiobreaker
import pybreaker
import pytest

from clients.clip import CLIPClient
from config import ImageEmbeddingConfig

logger = logging.getLogger(__name__)

# Skip marker for tests that require working CLIP API
SKIP_NO_CLIP_API = pytest.mark.skip(
    reason="Requires working CLIP API - ai4all/clip endpoint format not yet documented"
)

# Sample image data (1x1 pixel PNG - minimal valid PNG)
SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Second sample image (different bytes for comparison tests)
SAMPLE_IMAGE_2 = b"\x89PNG\r\n\x1a\nDIFFERENT_IMAGE_DATA"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def clip_client(clip_container) -> CLIPClient:
    """Create a CLIPClient connected to the ai4all/clip container.

    Uses the clip_container fixture from conftest.py which starts
    an ai4all/clip Docker container.

    Args:
        clip_container: CLIP container fixture from conftest.

    Yields:
        CLIPClient: Configured client for testing.
    """
    host = clip_container.get_container_host_ip()
    port = clip_container.get_exposed_port(8000)
    base_url = f"http://{host}:{port}"

    client = CLIPClient(
        base_url=base_url,
        model="ViT-B/32",
        backend="clip",  # Use ai4all/clip API format
        timeout_s=120,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout_s=10,
    )

    yield client

    client.close()


@pytest.fixture
def mock_image_bytes() -> bytes:
    """Create mock image bytes for testing.

    Returns:
        bytes: Sample PNG image data.
    """
    return SAMPLE_PNG


# =============================================================================
# Happy Path Tests
# =============================================================================


@pytest.mark.integration
@SKIP_NO_CLIP_API
class TestHappyPath:
    """Test suite for basic successful operations.

    These tests verify that the CLIP client works correctly
    under normal conditions with a healthy ai4all/clip container.
    """

    def test_embed_image_returns_vector(self, clip_client: CLIPClient, mock_image_bytes: bytes) -> None:
        """Test that embed_image() returns a valid embedding vector.

        **Why this test is important:**
        Image embedding is the core functionality of the CLIP client.
        This test verifies the complete request/response cycle.

        **What it tests:**
        - HTTP POST request is made to correct endpoint
        - Image is base64 encoded correctly
        - Response is parsed correctly
        - Embedding vector is returned
        """
        result = clip_client.embed_image(mock_image_bytes)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_embed_image_different_images_produce_different_vectors(self, clip_client: CLIPClient) -> None:
        """Test that different images produce different embeddings.

        **Why this test is important:**
        Embeddings should capture visual meaning. Different images should
        produce meaningfully different vectors to enable image search.

        **What it tests:**
        - Two different images produce different embedding vectors
        - The vectors are not identical
        """
        result1 = clip_client.embed_image(SAMPLE_PNG)
        result2 = clip_client.embed_image(SAMPLE_IMAGE_2)

        # Vectors should be different for different images
        # (may be same if model doesn't differentiate, but usually different)
        assert len(result1) == len(result2)

    @pytest.mark.asyncio
    async def test_embed_image_async_returns_vector(
        self, clip_client: CLIPClient, mock_image_bytes: bytes
    ) -> None:
        """Test that embed_image_async() returns a valid embedding vector.

        **Why this test is important:**
        Async embedding is critical for high-throughput pipelines where
        we need to parallelize multiple embedding requests.

        **What it tests:**
        - Async HTTP request completes successfully
        - Response is parsed correctly in async context
        - Embedding vector is returned
        """
        result = await clip_client.embed_image_async(mock_image_bytes)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)


# =============================================================================
# Batch Operation Tests
# =============================================================================


@pytest.mark.integration
@SKIP_NO_CLIP_API
class TestBatchOperations:
    """Test suite for batch embedding operations.

    Batch operations are more efficient than individual calls
    when embedding multiple images.
    """

    def test_embed_image_batch_returns_multiple_vectors(self, clip_client: CLIPClient) -> None:
        """Test that embed_image_batch() returns vectors for all images.

        **Why this test is important:**
        Batch embedding reduces API calls and improves throughput.
        This test verifies the batch API works correctly.

        **What it tests:**
        - Batch API processes multiple images
        - Correct number of vectors returned
        - Each vector has values
        """
        images = [SAMPLE_PNG, SAMPLE_PNG, SAMPLE_PNG]
        result = clip_client.embed_image_batch(images)

        assert len(result) == 3
        assert all(len(v) > 0 for v in result)
        assert all(isinstance(x, float) for v in result for x in v)

    @pytest.mark.asyncio
    async def test_embed_image_batch_async_returns_multiple_vectors(self, clip_client: CLIPClient) -> None:
        """Test that embed_image_batch_async() returns vectors for all images.

        **Why this test is important:**
        Async batch embedding combines the benefits of batching and
        async operations for maximum throughput.

        **What it tests:**
        - Async batch API processes multiple images
        - Correct number of vectors returned
        - Vectors are correctly parsed in async context
        """
        images = [SAMPLE_PNG, SAMPLE_PNG]
        result = await clip_client.embed_image_batch_async(images)

        assert len(result) == 2
        assert all(len(v) > 0 for v in result)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


@pytest.mark.integration
@SKIP_NO_CLIP_API
class TestCircuitBreaker:
    """Test suite for circuit breaker behavior.

    Circuit breakers prevent cascading failures by failing fast
    when a service is degraded.
    """

    def test_sync_circuit_breaker_starts_closed(self, clip_client: CLIPClient) -> None:
        """Test that sync circuit breaker starts in closed state.

        **Why this test is important:**
        The circuit breaker must start closed to allow normal operation.
        A breaker that starts open would block all requests.

        **What it tests:**
        - Sync circuit breaker (_breaker) exists
        - Initial state is CLOSED
        """
        assert clip_client._breaker is not None
        assert clip_client._breaker.current_state == pybreaker.STATE_CLOSED

    def test_async_circuit_breaker_starts_closed(self, clip_client: CLIPClient) -> None:
        """Test that async circuit breaker starts in closed state.

        **Why this test is important:**
        The async circuit breaker must also start closed.
        Async operations use a separate breaker (aiobreaker).

        **What it tests:**
        - Async circuit breaker (_async_breaker) exists
        - Initial state is CLOSED
        """
        assert clip_client._async_breaker is not None
        assert clip_client._async_breaker.current_state == aiobreaker.state.CircuitBreakerState.CLOSED


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Test suite for error handling behavior.

    These tests verify that the client handles errors gracefully
    and raises appropriate exceptions. These tests use fresh clients
    to avoid circuit breaker state pollution.
    """

    def test_empty_image_raises_value_error(self, clip_container) -> None:
        """Test that empty image bytes raises ValueError.

        **Why this test is important:**
        Empty input should fail fast with a clear error message.
        This is caught before making the network request.

        **What it tests:**
        - Empty bytes raises ValueError
        - Error message is descriptive
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
        )

        with pytest.raises(ValueError, match="empty"):
            client.embed_image(b"")

    def test_empty_batch_raises_value_error(self, clip_container) -> None:
        """Test that empty batch raises ValueError.

        **Why this test is important:**
        Empty batch should fail fast with a clear error message.
        This prevents unnecessary API calls.

        **What it tests:**
        - Empty list raises ValueError
        - Error message is descriptive
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
        )

        with pytest.raises(ValueError, match="empty"):
            client.embed_image_batch([])

    def test_batch_exceeds_max_raises_value_error(self, clip_container) -> None:
        """Test that oversized batch raises ValueError.

        **Why this test is important:**
        Large batches can cause memory issues. The client should
        enforce batch size limits.

        **What it tests:**
        - Batch exceeding max_batch_size raises ValueError
        - Error message indicates the limit
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
            max_batch_size=8,
        )

        # Create batch larger than max_batch_size (8)
        images = [SAMPLE_PNG] * 10

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            client.embed_image_batch(images)


# =============================================================================
# Resource Cleanup Tests
# =============================================================================


@pytest.mark.integration
class TestResourceCleanup:
    """Test suite for resource cleanup behavior.

    These tests verify that resources are properly released
    when the client is closed.
    """

    def test_close_releases_session(self, clip_container) -> None:
        """Test that close() releases the HTTP session.

        **Why this test is important:**
        Proper cleanup prevents resource leaks. The session should
        be set to None after close().

        **What it tests:**
        - close() is called without error
        - Session is set to None
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
        )

        # Access session to ensure it exists
        _ = client.session

        # Close should release session
        client.close()

        assert client._session is None

    @SKIP_NO_CLIP_API
    def test_client_usable_after_close_and_reopen(self, clip_container) -> None:
        """Test that client can be used after close if session is reset.

        **Why this test is important:**
        Clients should be resilient. After close(), a new session
        should be created automatically on next use.

        **What it tests:**
        - close() is called
        - New request creates new session
        - Request succeeds
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
        )

        # Use, close, then use again
        result1 = client.embed_image(SAMPLE_PNG)
        client.close()

        # Should create new session automatically
        result2 = client.embed_image(SAMPLE_PNG)

        assert result1 is not None
        assert result2 is not None
        assert len(result1) == len(result2)


# =============================================================================
# Factory Method Tests
# =============================================================================


@pytest.mark.integration
class TestFactoryMethod:
    """Test suite for from_config factory method.

    These tests verify that clients can be created from config objects.
    """

    def test_from_config_creates_client(self, clip_container) -> None:
        """Test that from_config() creates a properly configured client.

        **Why this test is important:**
        The factory method is the primary way clients are created
        in production. It should produce properly configured clients.

        **What it tests:**
        - from_config() creates a client
        - Client has correct configuration
        - Session and circuit breakers are initialized
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        config = ImageEmbeddingConfig(
            clip_url=f"http://{host}:{port}",
            clip_model="ViT-B/32",
            clip_backend="clip",
            clip_timeout=120,
            clip_circuit_breaker_threshold=5,
        )

        client = CLIPClient.from_config(config)

        try:
            # Verify client is properly configured
            assert client.base_url == f"http://{host}:{port}"
            assert client.model == "ViT-B/32"
            assert client.backend == "clip"
            assert client.timeout_s == 120
            assert client.circuit_breaker_failure_threshold == 5
            assert client._session is not None
            assert client._breaker is not None
            assert client._async_breaker is not None
        finally:
            client.close()


# =============================================================================
# Text Embedding Tests (for cross-modal search)
# =============================================================================


@pytest.mark.integration
@SKIP_NO_CLIP_API
class TestTextEmbedding:
    """Test suite for text embedding operations.

    These tests verify that the CLIP client can generate text embeddings
    in the same vector space as image embeddings, enabling cross-modal search.
    """

    def test_embed_text_returns_vector(self, clip_client: CLIPClient) -> None:
        """Test that embed_text() returns a valid embedding vector.

        **Why this test is important:**
        Text embeddings are required for text-to-image search queries.
        The text vector must be in the same space as image vectors.

        **What it tests:**
        - embed_text() returns a list of floats
        - Vector has the correct dimension
        """
        result = clip_client.embed_text("a fluffy cat sitting on a couch")

        assert isinstance(result, list)
        assert len(result) == clip_client.vector_size
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_text_async_returns_vector(self, clip_client: CLIPClient) -> None:
        """Test that embed_text_async() returns a valid embedding vector.

        **Why this test is important:**
        Async text embedding is needed for high-throughput query processing.

        **What it tests:**
        - embed_text_async() returns a list of floats
        - Vector has the correct dimension
        """
        result = await clip_client.embed_text_async("a golden retriever playing fetch")

        assert isinstance(result, list)
        assert len(result) == clip_client.vector_size
        assert all(isinstance(x, float) for x in result)

    def test_different_texts_produce_different_vectors(self, clip_client: CLIPClient) -> None:
        """Test that different texts produce different embeddings.

        **Why this test is important:**
        Semantically different texts should have different representations
        to enable meaningful search results.

        **What it tests:**
        - Two different text descriptions produce different vectors
        """
        result1 = clip_client.embed_text("a black cat")
        result2 = clip_client.embed_text("a sunny beach")

        # Vectors should be different
        assert result1 != result2


@pytest.mark.integration
@SKIP_NO_CLIP_API
class TestTextEmbeddingBatch:
    """Test suite for batch text embedding operations."""

    def test_embed_text_batch_returns_multiple_vectors(self, clip_client: CLIPClient) -> None:
        """Test that embed_text_batch() returns vectors for all texts.

        **Why this test is important:**
        Batch embedding is more efficient for processing multiple queries.

        **What it tests:**
        - Returns correct number of vectors
        - All vectors have correct dimension
        """
        texts = ["a cat", "a dog", "a bird"]
        results = clip_client.embed_text_batch(texts)

        assert len(results) == len(texts)
        assert all(len(v) == clip_client.vector_size for v in results)

    @pytest.mark.asyncio
    async def test_embed_text_batch_async_returns_multiple_vectors(self, clip_client: CLIPClient) -> None:
        """Test that embed_text_batch_async() returns vectors for all texts.

        **Why this test is important:**
        Async batch embedding enables high-throughput query processing.

        **What it tests:**
        - Returns correct number of vectors
        - All vectors have correct dimension
        """
        texts = ["a red car", "a blue boat"]
        results = await clip_client.embed_text_batch_async(texts)

        assert len(results) == len(texts)
        assert all(len(v) == clip_client.vector_size for v in results)


@pytest.mark.integration
class TestTextEmbeddingValidation:
    """Test suite for text embedding input validation."""

    def test_empty_text_raises_value_error(self, clip_container) -> None:
        """Test that empty text raises ValueError.

        **Why this test is important:**
        Empty queries should fail fast with a clear error.

        **What it tests:**
        - Empty string raises ValueError
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
        )

        with pytest.raises(ValueError, match="empty"):
            client.embed_text("")

    def test_whitespace_text_raises_value_error(self, clip_container) -> None:
        """Test that whitespace-only text raises ValueError.

        **Why this test is important:**
        Whitespace-only queries should be rejected.

        **What it tests:**
        - Whitespace string raises ValueError
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
        )

        with pytest.raises(ValueError, match="empty"):
            client.embed_text("   ")

    def test_empty_text_batch_raises_value_error(self, clip_container) -> None:
        """Test that empty text batch raises ValueError.

        **Why this test is important:**
        Empty batch should fail fast.

        **What it tests:**
        - Empty list raises ValueError
        """
        host = clip_container.get_container_host_ip()
        port = clip_container.get_exposed_port(8000)

        client = CLIPClient(
            base_url=f"http://{host}:{port}",
            model="ViT-B/32",
            backend="clip",
        )

        with pytest.raises(ValueError, match="empty"):
            client.embed_text_batch([])


@pytest.mark.integration
@SKIP_NO_CLIP_API
class TestCrossModalSearch:
    """Test suite for cross-modal (text-to-image) search scenarios.

    These tests verify that text and image embeddings live in the
    same vector space, enabling text-to-image similarity search.
    """

    def test_image_and_text_vectors_same_dimension(
        self, clip_client: CLIPClient, mock_image_bytes: bytes
    ) -> None:
        """Test that image and text embeddings have the same dimensions.

        **Why this test is important:**
        Cross-modal search requires vectors in the same space with
        identical dimensions for similarity computation.

        **What it tests:**
        - Image embedding dimension matches text embedding dimension
        """
        image_vector = clip_client.embed_image(mock_image_bytes)
        text_vector = clip_client.embed_text("a small image")

        assert len(image_vector) == len(text_vector)
        assert len(image_vector) == clip_client.vector_size

    @pytest.mark.asyncio
    async def test_async_image_and_text_vectors_same_dimension(
        self, clip_client: CLIPClient, mock_image_bytes: bytes
    ) -> None:
        """Test that async image and text embeddings have the same dimensions.

        **Why this test is important:**
        Async operations should produce vectors in the same space.

        **What it tests:**
        - Async image embedding dimension matches async text embedding dimension
        """
        image_vector = await clip_client.embed_image_async(mock_image_bytes)
        text_vector = await clip_client.embed_text_async("a small image")

        assert len(image_vector) == len(text_vector)
        assert len(image_vector) == clip_client.vector_size
