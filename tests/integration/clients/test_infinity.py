"""Integration tests for Infinity embedding client (SigLIP).

This module tests the InfinityClient against a real michaelf34/infinity container,
which provides an OpenAI-compatible embedding API supporting both text and image
modalities via the ``modality`` parameter.

See: https://github.com/michaelfeil/infinity

## Container Requirements

- Image: michaelf34/infinity:latest
- Port: 7997 (mapped to random host port)
- Command: v2 --model-id google/siglip-so400m-patch14-384 --port 7997
- Model: google/siglip-so400m-patch14-384 (SigLIP SO400M, 1152-dim vectors)

## Running Tests

```bash
# Run all Infinity integration tests
uv run pytest tests/integration/clients/test_infinity.py -v -s

# Run a specific test class
uv run pytest tests/integration/clients/test_infinity.py::TestHappyPath -v -s
```

## Test Categories

1. Happy Path: Core embedding functionality (image + text)
2. Batch Operations: Multi-item embedding requests
3. Circuit Breaker: Initialization state
4. Error Handling: Input validation (empty image, empty batch, oversized batch)
5. Resource Cleanup: Session lifecycle
6. Factory Method: Client creation from config
7. Text Embedding: Text-specific embedding tests
8. Text Embedding Batch: Batch text embedding
9. Text Embedding Validation: Text input validation
10. Cross-Modal Search: Image/text vector space alignment
"""

import asyncio
import logging

import aiobreaker
import pytest
from testcontainers.core.container import DockerContainer

from clients.infinity import INFINITY_VECTOR_SIZES, InfinityClient
from config import EmbeddingConfig, ProviderType

logger = logging.getLogger(__name__)


def _create_solid_color_png(color: tuple[int, int, int], size: int = 10) -> bytes:
    """Create a solid color PNG image.

    Args:
        color: RGB tuple (r, g, b).
        size: Image dimensions (size x size).

    Returns:
        PNG image bytes.
    """
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (size, size), color=color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# Sample images for testing - created with PIL for validity
SAMPLE_PNG = _create_solid_color_png((255, 0, 0))  # Red image
SAMPLE_IMAGE_2 = _create_solid_color_png((0, 0, 255))  # Blue image (different from red)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def infinity_client(infinity_container: DockerContainer) -> InfinityClient:
    """Create an InfinityClient connected to the Infinity container.

    Uses the infinity_container fixture from conftest.py which starts
    a michaelf34/infinity Docker container with SigLIP SO400M.

    Args:
        infinity_container: Infinity container fixture from conftest.

    Yields:
        InfinityClient: Configured client for testing.
    """
    host = infinity_container.get_container_host_ip()
    port = infinity_container.get_exposed_port(7997)
    base_url = f"http://{host}:{port}"

    client = InfinityClient(
        base_url=base_url,
        model="google/siglip-so400m-patch14-384",
        timeout_s=120,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout_s=10,
    )

    yield client

    asyncio.run(client.close())


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
class TestHappyPath:
    """Test suite for basic successful operations.

    These tests verify that the Infinity client works correctly
    under normal conditions with a healthy Infinity container
    serving the SigLIP SO400M model.
    """

    @pytest.mark.asyncio
    async def test_embed_image_returns_vector(
        self, infinity_client: InfinityClient, mock_image_bytes: bytes
    ) -> None:
        """Test that embed_image() returns a valid embedding vector.

        **Why this test is important:**
        Image embedding is the core functionality of the Infinity client.
        This test verifies the complete request/response cycle including
        base64 encoding and modality parameter handling.

        **What it tests:**
        - HTTP POST request to /embeddings with modality=image
        - Image is base64 encoded as data URI
        - Response parsed via Pydantic model
        - Embedding vector has expected dimension (1152 for SigLIP SO400M)
        """
        result = await infinity_client.embed_image(mock_image_bytes)

        assert isinstance(result, list)
        assert len(result) == INFINITY_VECTOR_SIZES["google/siglip-so400m-patch14-384"]
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_image_different_images_produce_different_vectors(
        self, infinity_client: InfinityClient
    ) -> None:
        """Test that different images produce different embeddings.

        **Why this test is important:**
        Embeddings should capture visual meaning. Different images should
        produce meaningfully different vectors to enable image search.

        **What it tests:**
        - Two different colored images produce different embedding vectors
        - The vectors have the same dimension
        """
        result1 = await infinity_client.embed_image(SAMPLE_PNG)
        result2 = await infinity_client.embed_image(SAMPLE_IMAGE_2)

        assert len(result1) == len(result2)
        assert result1 != result2


# =============================================================================
# Batch Operation Tests
# =============================================================================


@pytest.mark.integration
class TestBatchOperations:
    """Test suite for batch embedding operations.

    Batch operations send multiple inputs in a single API call,
    reducing overhead compared to individual calls.
    """

    @pytest.mark.asyncio
    async def test_embed_image_batch_returns_multiple_vectors(self, infinity_client: InfinityClient) -> None:
        """Test that embed_image_batch() returns vectors for all images.

        **Why this test is important:**
        Batch embedding reduces API calls and improves throughput during
        ingestion. This test verifies the batch API works correctly.

        **What it tests:**
        - Batch API processes multiple images
        - Correct number of vectors returned
        - Each vector has correct dimension
        """
        images = [SAMPLE_PNG, SAMPLE_PNG, SAMPLE_IMAGE_2]
        result = await infinity_client.embed_image_batch(images)

        assert len(result) == 3
        assert all(len(v) == INFINITY_VECTOR_SIZES["google/siglip-so400m-patch14-384"] for v in result)
        assert all(isinstance(x, float) for v in result for x in v)

    @pytest.mark.asyncio
    async def test_embed_text_batch_returns_multiple_vectors(self, infinity_client: InfinityClient) -> None:
        """Test that embed_text_batch() returns vectors for all texts.

        **Why this test is important:**
        Batch text embedding is used for processing multiple search queries
        or generating embeddings for document chunks.

        **What it tests:**
        - Returns correct number of vectors
        - All vectors have correct dimension
        """
        texts = ["a cat", "a dog", "a bird"]
        results = await infinity_client.embed_text_batch(texts)

        assert len(results) == len(texts)
        assert all(len(v) == INFINITY_VECTOR_SIZES["google/siglip-so400m-patch14-384"] for v in results)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


@pytest.mark.integration
class TestCircuitBreaker:
    """Test suite for circuit breaker behavior.

    Circuit breakers prevent cascading failures by failing fast
    when the Infinity service is degraded.
    """

    def test_async_circuit_breaker_starts_closed(self, infinity_client: InfinityClient) -> None:
        """Test that async circuit breaker starts in closed state.

        **Why this test is important:**
        The async circuit breaker must start closed to allow normal operation.
        Async operations use aiobreaker.

        **What it tests:**
        - Async circuit breaker (_async_breaker) exists
        - Initial state is CLOSED
        """
        assert infinity_client._async_breaker is not None
        assert infinity_client._async_breaker.current_state == aiobreaker.state.CircuitBreakerState.CLOSED


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Test suite for error handling behavior.

    These tests verify that the client handles errors gracefully
    and raises appropriate exceptions. Fresh clients are created
    per test to avoid circuit breaker state pollution.
    """

    @pytest.mark.asyncio
    async def test_empty_image_raises_value_error(self, infinity_container: DockerContainer) -> None:
        """Test that empty image bytes raises ValueError.

        **Why this test is important:**
        Empty input should fail fast with a clear error message
        before making the network request.

        **What it tests:**
        - Empty bytes raises ValueError
        - Error message indicates the issue
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
        )

        with pytest.raises(ValueError, match="empty"):
            await client.embed_image(b"")

    @pytest.mark.asyncio
    async def test_empty_batch_raises_value_error(self, infinity_container: DockerContainer) -> None:
        """Test that empty image batch raises ValueError.

        **Why this test is important:**
        Empty batch should fail fast with a clear error message.
        This prevents unnecessary API calls.

        **What it tests:**
        - Empty list raises ValueError
        - Error message is descriptive
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
        )

        with pytest.raises(ValueError, match="empty"):
            await client.embed_image_batch([])

    @pytest.mark.asyncio
    async def test_batch_exceeds_max_raises_value_error(self, infinity_container: DockerContainer) -> None:
        """Test that oversized batch raises ValueError.

        **Why this test is important:**
        Large batches can cause memory issues on the Infinity server.
        The client should enforce batch size limits.

        **What it tests:**
        - Batch exceeding max_batch_size raises ValueError
        - Error message indicates the limit
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
            max_batch_size=8,
        )

        images = [SAMPLE_PNG for _ in range(10)]

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await client.embed_image_batch(images)


# =============================================================================
# Resource Cleanup Tests
# =============================================================================


@pytest.mark.integration
class TestResourceCleanup:
    """Test suite for resource cleanup behavior.

    These tests verify that resources are properly released
    when the client is closed and recreated on next use.
    """

    @pytest.mark.asyncio
    async def test_client_usable_after_close_and_reopen(self, infinity_container: DockerContainer) -> None:
        """Test that client can be used after close if session is reset.

        **Why this test is important:**
        Clients should be resilient. After close(), a new async client
        should be created automatically on next use.

        **What it tests:**
        - close() releases resources
        - New request creates new async client
        - Subsequent request succeeds
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
        )

        result1 = await client.embed_image(SAMPLE_PNG)
        await client.close()

        result2 = await client.embed_image(SAMPLE_PNG)

        assert result1 is not None
        assert result2 is not None
        assert len(result1) == len(result2)


# =============================================================================
# Factory Method Tests
# =============================================================================


@pytest.mark.integration
class TestFactoryMethod:
    """Test suite for from_config factory method.

    These tests verify that clients can be created from EmbeddingConfig
    objects and produce properly configured instances.
    """

    @pytest.mark.asyncio
    async def test_from_config_creates_client(self, infinity_container: DockerContainer) -> None:
        """Test that from_config() creates a properly configured client.

        **Why this test is important:**
        The factory method is the primary way clients are created
        in production. It should produce properly configured clients.

        **What it tests:**
        - from_config() creates a client with correct settings
        - Circuit breakers are initialized
        - Client can make real API calls
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        config = EmbeddingConfig(
            provider_type=ProviderType.INFINITY,
            infinity_url=f"http://{host}:{port}",
            infinity_model="google/siglip-so400m-patch14-384",
            infinity_timeout=120,
        )

        client = InfinityClient.from_config(config)

        try:
            assert client.base_url == f"http://{host}:{port}"
            assert client.model == "google/siglip-so400m-patch14-384"
            assert client.timeout_s == 120
            assert client._async_breaker is not None

            # Verify the factory-created client can make real API calls
            result = await client.embed_text("test query")
            assert isinstance(result, list)
            assert len(result) == INFINITY_VECTOR_SIZES["google/siglip-so400m-patch14-384"]
        finally:
            await client.close()


# =============================================================================
# Text Embedding Tests
# =============================================================================


@pytest.mark.integration
class TestTextEmbedding:
    """Test suite for text embedding operations.

    These tests verify that the Infinity client can generate text embeddings
    in the same vector space as image embeddings, enabling cross-modal search.
    SigLIP models support both text and image modalities natively.
    """

    @pytest.mark.asyncio
    async def test_embed_text_returns_vector(self, infinity_client: InfinityClient) -> None:
        """Test that embed_text() returns a valid embedding vector.

        **Why this test is important:**
        Text embeddings are required for text-to-image search queries.
        The text vector must be in the same space as image vectors.

        **What it tests:**
        - embed_text() sends modality=text to the Infinity API
        - Returns a list of floats
        - Vector has the correct dimension (1152 for SigLIP SO400M)
        """
        result = await infinity_client.embed_text("a fluffy cat sitting on a couch")

        assert isinstance(result, list)
        assert len(result) == infinity_client.vector_size
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_different_texts_produce_different_vectors(self, infinity_client: InfinityClient) -> None:
        """Test that different texts produce different embeddings.

        **Why this test is important:**
        Semantically different texts should have different representations
        to enable meaningful search results.

        **What it tests:**
        - Two different text descriptions produce different vectors
        """
        result1 = await infinity_client.embed_text("a black cat")
        result2 = await infinity_client.embed_text("a sunny beach")

        assert result1 != result2


# =============================================================================
# Text Embedding Validation Tests
# =============================================================================


@pytest.mark.integration
class TestTextEmbeddingValidation:
    """Test suite for text embedding input validation."""

    @pytest.mark.asyncio
    async def test_empty_text_raises_value_error(self, infinity_container: DockerContainer) -> None:
        """Test that empty text raises ValueError.

        **Why this test is important:**
        Empty queries should fail fast with a clear error.

        **What it tests:**
        - Empty string raises ValueError
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
        )

        with pytest.raises(ValueError, match="empty"):
            await client.embed_text("")

    @pytest.mark.asyncio
    async def test_whitespace_text_raises_value_error(self, infinity_container: DockerContainer) -> None:
        """Test that whitespace-only text raises ValueError.

        **Why this test is important:**
        Whitespace-only queries should be rejected before hitting the API.

        **What it tests:**
        - Whitespace string raises ValueError
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
        )

        with pytest.raises(ValueError, match="empty"):
            await client.embed_text("   ")

    @pytest.mark.asyncio
    async def test_empty_text_batch_raises_value_error(self, infinity_container: DockerContainer) -> None:
        """Test that empty text batch raises ValueError.

        **Why this test is important:**
        Empty batch should fail fast.

        **What it tests:**
        - Empty list raises ValueError
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
        )

        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_batch([])

    @pytest.mark.asyncio
    async def test_text_batch_exceeds_max_raises_value_error(
        self, infinity_container: DockerContainer
    ) -> None:
        """Test that oversized text batch raises ValueError.

        **Why this test is important:**
        Large text batches should be caught before hitting the API.

        **What it tests:**
        - Batch exceeding max_batch_size raises ValueError
        """
        host = infinity_container.get_container_host_ip()
        port = infinity_container.get_exposed_port(7997)

        client = InfinityClient(
            base_url=f"http://{host}:{port}",
            model="google/siglip-so400m-patch14-384",
            max_batch_size=3,
        )

        texts = [f"text {i}" for i in range(5)]

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await client.embed_text_batch(texts)


# =============================================================================
# Cross-Modal Search Tests
# =============================================================================


@pytest.mark.integration
class TestCrossModalSearch:
    """Test suite for cross-modal (text-to-image) search scenarios.

    These tests verify that text and image embeddings live in the
    same vector space, enabling text-to-image similarity search.
    SigLIP models are specifically designed for this capability.
    """

    @pytest.mark.asyncio
    async def test_image_and_text_vectors_same_dimension(
        self, infinity_client: InfinityClient, mock_image_bytes: bytes
    ) -> None:
        """Test that image and text embeddings have the same dimensions.

        **Why this test is important:**
        Cross-modal search requires vectors in the same space with
        identical dimensions for similarity computation (cosine/dot product).

        **What it tests:**
        - Image embedding dimension matches text embedding dimension
        - Both match the expected model vector size
        """
        image_vector = await infinity_client.embed_image(mock_image_bytes)
        text_vector = await infinity_client.embed_text("a small image")

        assert len(image_vector) == len(text_vector)
        assert len(image_vector) == infinity_client.vector_size

    @pytest.mark.asyncio
    async def test_related_text_closer_to_image_than_unrelated(self, infinity_client: InfinityClient) -> None:
        """Test that semantically related text is closer to an image.

        **Why this test is important:**
        The core value of SigLIP is that text describing an image should
        produce vectors closer to that image than unrelated text. This
        is what enables text-to-image search.

        **What it tests:**
        - A red image is more similar to 'a red square' than 'a blue ocean'
        - Cosine similarity ordering is correct
        """
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        red_image = _create_solid_color_png((255, 0, 0), size=64)
        image_vector = await infinity_client.embed_image(red_image)

        related_vector = await infinity_client.embed_text("a solid red square")
        unrelated_vector = await infinity_client.embed_text("a blue ocean with waves")

        sim_related = cosine_similarity(image_vector, related_vector)
        sim_unrelated = cosine_similarity(image_vector, unrelated_vector)

        # Related text should be more similar to the image
        assert sim_related > sim_unrelated, (
            f"Expected related text similarity ({sim_related:.4f}) > "
            f"unrelated text similarity ({sim_unrelated:.4f})"
        )
