"""Unit tests for InfinityClient image and text embedding client.

Tests for the InfinityClient class that generates embeddings via Infinity's
OpenAI-compatible API with SigLIP models.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clients.infinity import INFINITY_VECTOR_SIZES, InfinityClient
from clients.interfaces import EmbeddingProvider
from config import EmbeddingConfig, ProviderType


class TestInfinityClientInit:
    """Tests for InfinityClient initialization."""

    def test_creates_with_required_params(self) -> None:
        """Test that InfinityClient initializes with required parameters.

        **Why this test is important:**
          - Client initialization is the foundation for all operations
          - Ensures required parameters are accepted
          - Validates default values are set correctly

        **What it tests:**
          - base_url and model are stored correctly
          - Default timeout and batch size are set
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )

        assert client.base_url == "http://localhost:7997"
        assert client.model == "google/siglip-so400m-patch14-384"
        assert client.timeout_s == 120  # Default
        assert client.max_batch_size == 8  # Default

    def test_creates_with_custom_params(self) -> None:
        """Test that InfinityClient accepts custom configuration parameters.

        **Why this test is important:**
          - Custom configuration is needed for different deployment environments
          - Validates all configurable options work correctly

        **What it tests:**
          - Custom timeout, circuit breaker, and batch settings are stored
          - Vector size override is applied correctly
        """
        client = InfinityClient(
            base_url="http://custom:7997",
            model="google/siglip-so400m-patch14-384",
            timeout_s=60,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_s=15,
            max_batch_size=4,
            vector_size_override=1024,
        )

        assert client.base_url == "http://custom:7997"
        assert client.model == "google/siglip-so400m-patch14-384"
        assert client.timeout_s == 60
        assert client.circuit_breaker_failure_threshold == 3
        assert client.circuit_breaker_recovery_timeout_s == 15
        assert client.max_batch_size == 4
        assert client.vector_size_override == 1024

    def test_initializes_circuit_breakers(self) -> None:
        """Test that InfinityClient initializes circuit breakers.

        **Why this test is important:**
          - Circuit breakers prevent cascading failures
          - Async operations need protection

        **What it tests:**
          - Async circuit breaker is created
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )

        assert client._async_breaker is not None


class TestInfinityClientVectorSize:
    """Tests for InfinityClient.vector_size property."""

    def test_returns_known_model_size(self) -> None:
        """Test that vector_size returns correct size for known models.

        **Why this test is important:**
          - Vector size must match model output for correct indexing
          - Mismatched sizes cause vector DB errors

        **What it tests:**
          - Known model (SigLIP SO400M) returns its documented vector size
        """
        model = "google/siglip-so400m-patch14-384"
        client = InfinityClient(base_url="http://localhost:7997", model=model)
        assert client.vector_size == INFINITY_VECTOR_SIZES[model]

    def test_returns_override_when_set(self) -> None:
        """Test that vector_size_override takes precedence.

        **Why this test is important:**
          - Allows using custom or fine-tuned models with different sizes
          - Override must take priority over model lookup

        **What it tests:**
          - Override value is returned instead of model default
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
            vector_size_override=768,
        )
        assert client.vector_size == 768

    def test_returns_default_for_unknown_model(self) -> None:
        """Test that unknown models fall back to default size.

        **Why this test is important:**
          - New models shouldn't break the client
          - Default provides reasonable fallback

        **What it tests:**
          - Unknown model returns 1152 (SigLIP SO400M default)
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="unknown-model",
        )
        assert client.vector_size == 1152

    @pytest.mark.parametrize(("model", "expected_size"), INFINITY_VECTOR_SIZES.items())
    def test_known_model_sizes(self, model: str, expected_size: int) -> None:
        """Test that all known models return their correct vector sizes.

        **Why this test is important:**
          - Documents expected sizes for all supported models
          - Catches regressions in model size mapping

        **What it tests:**
          - Each model in the size map returns correct dimension
        """
        client = InfinityClient(base_url="http://localhost:7997", model=model)
        assert client.vector_size == expected_size


class TestInfinityClientEmbedImage:
    """Tests for InfinityClient.embed_image method."""

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_success(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_image returns embedding vector.

        **Why this test is important:**
          - Core functionality for image embedding
          - Must return correct format

        **What it tests:**
          - Returns embedding from API response
          - Async client is used correctly
        """
        expected_embedding = [0.1] * 1152
        infinity_response = {"data": [{"embedding": expected_embedding, "index": 0}]}
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = infinity_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        result = await client.embed_image(b"fake image data")

        assert result == expected_embedding
        mock_client.post.assert_called_once()

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_sends_base64_data_uri(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_image sends base64 data URI in request.

        **Why this test is important:**
          - Infinity requires specific data URI format
          - Format must include mime type prefix

        **What it tests:**
          - Request payload contains data:image/jpeg;base64,... format
        """
        infinity_response = {"data": [{"embedding": [0.1] * 1152, "index": 0}]}
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = infinity_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        # Use valid JPEG header
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        await client.embed_image(jpeg_bytes)

        # Verify post was called with correct payload structure
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert "input" in payload
        assert len(payload["input"]) == 1
        assert payload["input"][0].startswith("data:image/")
        assert "base64," in payload["input"][0]

    @pytest.mark.asyncio
    async def test_embed_image_empty_raises(self) -> None:
        """Test that embed_image rejects empty image bytes.

        **Why this test is important:**
          - Empty images are invalid and waste API calls
          - Fail-fast improves error diagnosis

        **What it tests:**
          - ValueError raised with descriptive message
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        with pytest.raises(ValueError, match="empty"):
            await client.embed_image(b"")


class TestInfinityClientEmbedImageBatch:
    """Tests for InfinityClient.embed_image_batch method."""

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_batch_success(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_image_batch returns vectors for all images.

        **Why this test is important:**
          - Batch embedding is core use case for Ray workers
          - Must process all images correctly

        **What it tests:**
          - Returns correct number of embeddings
          - Single request with multiple images
        """
        mock_response = {
            "data": [
                {"embedding": [0.1] * 1152, "index": 0},
                {"embedding": [0.2] * 1152, "index": 1},
            ]
        }
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = mock_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        # Valid JPEG headers
        images = [b"\xff\xd8\xff\xe0\x00\x10JFIF", b"\xff\xd8\xff\xe0\x00\x10JFIF"]
        result = await client.embed_image_batch(images)

        assert len(result) == 2
        assert all(len(v) == 1152 for v in result)

    @pytest.mark.asyncio
    async def test_embed_image_batch_empty_raises(self) -> None:
        """Test that embed_image_batch rejects empty list.

        **Why this test is important:**
          - Empty batch is a programming error
          - Early validation prevents downstream confusion

        **What it tests:**
          - ValueError raised for empty list
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        with pytest.raises(ValueError, match="empty"):
            await client.embed_image_batch([])

    @pytest.mark.asyncio
    async def test_embed_image_batch_exceeds_max_raises(self) -> None:
        """Test that embed_image_batch rejects oversized batches.

        **Why this test is important:**
          - Large batches can overwhelm the API
          - Batch size limits prevent memory issues

        **What it tests:**
          - ValueError raised when batch exceeds max_batch_size
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        images = [b"foo"] * 10  # Exceeds default max of 8

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await client.embed_image_batch(images)


class TestInfinityClientEmbedText:
    """Tests for InfinityClient.embed_text method."""

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_success(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_text returns embedding vector.

        **Why this test is important:**
          - Text embedding is core functionality
          - Enables text-to-image search

        **What it tests:**
          - Returns embedding from API response
        """
        expected_embedding = [0.1] * 1152
        infinity_response = {"data": [{"embedding": expected_embedding, "index": 0}]}
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = infinity_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        result = await client.embed_text("a fluffy cat")

        assert result == expected_embedding
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_text_empty_raises(self) -> None:
        """Test that embed_text rejects empty text.

        **Why this test is important:**
          - Empty text is invalid input
          - Fail fast on invalid input

        **What it tests:**
          - ValueError raised for empty text
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text("")


class TestInfinityClientEmbedTextBatch:
    """Tests for InfinityClient.embed_text_batch method."""

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_batch_success(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_text_batch returns embeddings for all texts.

        **Why this test is important:**
          - Batch text embedding is used for bulk queries
          - Must process all texts correctly

        **What it tests:**
          - Returns correct number of embeddings
        """
        mock_response = {
            "data": [
                {"embedding": [0.1] * 1152, "index": 0},
                {"embedding": [0.2] * 1152, "index": 1},
            ]
        }
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = mock_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        texts = ["cat", "dog"]
        result = await client.embed_text_batch(texts)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_embed_text_batch_empty_list_raises(self) -> None:
        """Test that embed_text_batch rejects empty list.

        **Why this test is important:**
          - Empty batch is a programming error
          - Early validation prevents downstream confusion

        **What it tests:**
          - ValueError raised for empty list
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_batch([])

    @pytest.mark.asyncio
    async def test_embed_text_batch_empty_string_raises(self) -> None:
        """Test that embed_text_batch rejects batches with empty text.

        **Why this test is important:**
          - Any empty text invalidates the batch
          - Consistent with sync batch behavior

        **What it tests:**
          - ValueError raised when batch contains empty string
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_batch(["cat", "", "dog"])

    @pytest.mark.asyncio
    async def test_embed_text_batch_exceeds_max_raises(self) -> None:
        """Test that embed_text_batch rejects oversized batches.

        **Why this test is important:**
          - Large batches can overwhelm the API
          - Batch size limits prevent memory issues

        **What it tests:**
          - ValueError raised when batch exceeds max_batch_size
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
            max_batch_size=4,
        )
        texts = ["text"] * 10  # Exceeds max of 4

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await client.embed_text_batch(texts)


class TestInfinityClientModality:
    """Tests for modality parameter in requests."""

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_modality_parameter_text(self, mock_async_client_cls: MagicMock) -> None:
        """Test that text embedding sends modality='text' parameter.

        **Why this test is important:**
          - Infinity uses modality to distinguish text vs image
          - Incorrect modality returns wrong vector space

        **What it tests:**
          - Request payload includes modality: "text"
        """
        infinity_response = {"data": [{"embedding": [0.1] * 1152, "index": 0}]}
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = infinity_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        await client.embed_text("test text")

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["modality"] == "text"

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_modality_parameter_image(self, mock_async_client_cls: MagicMock) -> None:
        """Test that image embedding sends modality='image' parameter.

        **Why this test is important:**
          - Infinity uses modality to distinguish text vs image
          - Incorrect modality returns wrong vector space

        **What it tests:**
          - Request payload includes modality: "image"
        """
        infinity_response = {"data": [{"embedding": [0.1] * 1152, "index": 0}]}
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = infinity_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        await client.embed_image(jpeg_bytes)

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["modality"] == "image"


class TestInfinityClientErrorHandling:
    """Tests for error handling behavior."""

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_upstream_error_on_http_failure(self, mock_async_client_cls: MagicMock) -> None:
        """Test that HTTP errors raise UpstreamError.

        **Why this test is important:**
          - Network errors must be surfaced with proper error type
          - UpstreamError enables circuit breaker integration

        **What it tests:**
          - HTTPError is caught and wrapped
          - Error message is descriptive
        """
        import httpx

        from core.exceptions import UpstreamError

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        with pytest.raises(UpstreamError):
            await client.embed_text("test")

    @patch("clients.infinity.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_upstream_error_on_missing_embedding(self, mock_async_client_cls: MagicMock) -> None:
        """Test that malformed responses raise UpstreamError.

        **Why this test is important:**
          - API may return malformed responses
          - Must fail clearly rather than silently corrupt data

        **What it tests:**
          - Empty response triggers UpstreamError
          - Error message indicates unexpected format
        """
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {}  # Missing "data" field
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        from core.exceptions import UpstreamError

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        with pytest.raises(UpstreamError, match="Unexpected response"):
            await client.embed_text("test")


class TestInfinityClientCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_async_circuit_breaker_starts_closed(self) -> None:
        """Test that async circuit breaker starts in closed state.

        **Why this test is important:**
          - Async breaker is used in Ray workers
          - Must allow initial requests

        **What it tests:**
          - Async breaker exists and is closed
        """
        import aiobreaker

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )

        assert client._async_breaker is not None
        assert client._async_breaker.current_state == aiobreaker.state.CircuitBreakerState.CLOSED


class TestInfinityClientEmbeddingProviderSatisfaction:
    """Tests for EmbeddingProvider abstract class implementation."""

    def test_implements_abc(self) -> None:
        """Test that InfinityClient implements EmbeddingProvider ABC.

        **Why this test is important:**
          - Protocol compliance enables dependency injection
          - Allows swapping implementations in tests

        **What it tests:**
          - isinstance check passes for protocol
        """
        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )
        assert isinstance(client, EmbeddingProvider)

    def test_has_required_methods(self) -> None:
        """Test that InfinityClient has all required abc methods."""

        client = InfinityClient(
            base_url="http://localhost:7997",
            model="google/siglip-so400m-patch14-384",
        )

        for method in EmbeddingProvider.__abstractmethods__:
            assert hasattr(client, method)


class TestInfinityClientFromConfig:
    """Tests for InfinityClient.from_config factory method."""

    def test_vector_size_from_config(self) -> None:
        """Test that from_config creates client from EmbeddingConfig.

        **Why this test is important:**
          - Factory method is the standard way to create clients
          - Must correctly map all config fields to client attributes

        **What it tests:**
          - All config values are transferred to client
          - Client is usable after creation
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.INFINITY,
            infinity_url="http://test:7997",
            infinity_model="google/siglip-so400m-patch14-384",
            infinity_timeout=90,
            infinity_vector_size=1024,
        )

        client = InfinityClient.from_config(config)

        assert client.base_url == "http://test:7997"
        assert client.model == "google/siglip-so400m-patch14-384"
        assert client.timeout_s == 90
        assert client.vector_size_override == 1024

    def test_raises_without_url(self) -> None:
        """Test that from_config requires infinity_url.

        **Why this test is important:**
          - URL is required for API connectivity
          - Must fail early with clear message

        **What it tests:**
          - ValueError raised for missing URL
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.INFINITY,
            infinity_model="google/siglip-so400m-patch14-384",
        )

        with pytest.raises(ValueError, match="infinity_url is required"):
            InfinityClient.from_config(config)

    def test_raises_without_model(self) -> None:
        """Test that from_config requires infinity_model.

        **Why this test is important:**
          - Model is required for embedding generation
          - Must fail early with clear message

        **What it tests:**
          - ValueError raised for missing model
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.INFINITY,
            infinity_url="http://test:7997",
        )

        with pytest.raises(ValueError, match="infinity_model is required"):
            InfinityClient.from_config(config)


class TestInfinityClientRegistration:
    """Tests for provider registry integration."""

    def test_provider_registry_clip(self) -> None:
        """Test that 'clip' is registered in the unified provider registry.

        **Why this test is important:**
          - Registry enables factory pattern for provider creation
          - Ensures CLIPClient is available via unified registry

        **What it tests:**
          - 'clip' key exists in _PROVIDER_REGISTRY
        """
        from clients.interfaces.embedding import _PROVIDER_REGISTRY
        from config import ProviderType

        assert ProviderType.LOCAL_CLIP in _PROVIDER_REGISTRY

    def test_provider_registry_infinity(self) -> None:
        """Test that 'infinity' is registered in the unified provider registry.

        **Why this test is important:**
          - Registry enables factory pattern for provider creation
          - Ensures InfinityClient is available via unified registry

        **What it tests:**
          - 'infinity' key exists in _PROVIDER_REGISTRY
        """
        from clients.interfaces.embedding import _PROVIDER_REGISTRY
        from config import ProviderType

        assert ProviderType.INFINITY in _PROVIDER_REGISTRY


class TestInfinityClientConfigFields:
    """Tests for EmbeddingConfig Infinity fields."""

    def test_config_infinity_fields(self) -> None:
        """Test that EmbeddingConfig loads infinity_* env vars correctly.

        **Why this test is important:**
          - Configuration is the foundation for provider creation
          - Must correctly parse environment variables

        **What it tests:**
          - Config has infinity_url, infinity_model, infinity_timeout fields
          - Fields can be set directly
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.INFINITY,
            infinity_url="http://test:7997",
            infinity_model="google/siglip-so400m-patch14-384",
            infinity_timeout=120,
            infinity_vector_size=1152,
        )

        assert config.infinity_url == "http://test:7997"
        assert config.infinity_model == "google/siglip-so400m-patch14-384"
        assert config.infinity_timeout == 120
        assert config.infinity_vector_size == 1152


class TestInfinityVectorSizes:
    """Tests for INFINITY_VECTOR_SIZES constant."""

    def test_vector_size_from_model_lookup(self) -> None:
        """Test that INFINITY_VECTOR_SIZES has expected SigLIP models.

        **Why this test is important:**
          - Vector size must match model output
          - Documents known SigLIP model dimensions

        **What it tests:**
          - Dict has SigLIP SO400M → 1152
          - Dict has SigLIP2 Base → 768
        """
        assert "google/siglip-so400m-patch14-384" in INFINITY_VECTOR_SIZES
        assert INFINITY_VECTOR_SIZES["google/siglip-so400m-patch14-384"] == 1152
        assert "google/siglip2-base-patch16-224" in INFINITY_VECTOR_SIZES
        assert INFINITY_VECTOR_SIZES["google/siglip2-base-patch16-224"] == 768
