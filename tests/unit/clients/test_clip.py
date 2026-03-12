# pyright: reportPrivateUsage=false

"""Unit tests for CLIPClient image embedding client.

Tests for the CLIPClient class that generates image embeddings via CLIP-compatible
APIs (Ollama LLaVA, etc.).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiobreaker
import httpx
import pytest

from clients.clip import CLIP_VECTOR_SIZES, CLIPClient
from clients.interfaces import EmbeddingProvider
from config import EmbeddingConfig, ProviderType
from foundation.exceptions import UpstreamError


class TestCLIPClientInit:
    """Tests for CLIPClient initialization."""

    def test_creates_with_required_params(self) -> None:
        """Test that CLIPClient initializes with required parameters.

        **Why this test is important:**
          - Client initialization is the foundation for all operations
          - Ensures required parameters are accepted
          - Validates default values are set correctly

        **What it tests:**
          - base_url and model are stored correctly
          - Default timeout and batch size are set
        """
        client = CLIPClient(base_url="http://localhost:11434", model="test-model", is_hosted=False)

        assert client.base_url == "http://localhost:11434"
        assert client.model == "test-model"
        assert client.timeout_s == 120  # Default
        assert client.max_batch_size == 8  # Default

    def test_creates_with_custom_params(self) -> None:
        """Test that CLIPClient accepts custom configuration parameters.

        **Why this test is important:**
          - Custom configuration is needed for different deployment environments
          - Validates all configurable options work correctly

        **What it tests:**
          - Custom timeout, circuit breaker, and batch settings are stored
          - Vector size override is applied correctly
        """
        client = CLIPClient(
            is_hosted=False,
            base_url="http://custom:11434",
            model="bakllava",
            timeout_s=60,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_s=15,
            max_batch_size=4,
            vector_size_override=1024,
        )

        assert client.base_url == "http://custom:11434"
        assert client.model == "bakllava"
        assert client.timeout_s == 60
        assert client.circuit_breaker_failure_threshold == 3
        assert client.circuit_breaker_recovery_timeout_s == 15
        assert client.max_batch_size == 4
        assert client.vector_size_override == 1024

    def test_initializes_circuit_breakers(self) -> None:
        """Test that CLIPClient initializes circuit breakers.

        **Why this test is important:**
          - Circuit breakers prevent cascading failures
          - Async operations need protection

        **What it tests:**
          - Async circuit breaker is created
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)

        assert client._async_breaker is not None


class TestCLIPClientVectorSize:
    """Tests for CLIPClient.vector_size property."""

    def test_returns_known_model_size(self) -> None:
        """Test that vector_size returns correct size for known models.

        **Why this test is important:**
          - Vector size must match model output for correct indexing
          - Mismatched sizes cause vector DB errors

        **What it tests:**
          - Known model (llava) returns its documented vector size
        """
        model = "clip-vit-base-patch32"
        client = CLIPClient(base_url="http://localhost:11434", model=model, is_hosted=False)
        assert client.vector_size == CLIP_VECTOR_SIZES[model]

    def test_returns_override_when_set(self) -> None:
        """Test that vector_size_override takes precedence.

        **Why this test is important:**
          - Allows using custom or fine-tuned models with different sizes
          - Override must take priority over model lookup

        **What it tests:**
          - Override value is returned instead of model default
        """
        client = CLIPClient(
            base_url="http://localhost:11434",
            model="llava",
            vector_size_override=768,
            is_hosted=False,
        )
        assert client.vector_size == 768

    def test_returns_default_for_unknown_model(self) -> None:
        """Test that unknown models fall back to default size.

        **Why this test is important:**
          - New models shouldn't break the client
          - Default provides reasonable fallback

        **What it tests:**
          - Unknown model returns 512 (CLIP default)
        """
        client = CLIPClient(
            base_url="http://localhost:11434",
            model="unknown-model",
            is_hosted=False,
        )
        assert client.vector_size == 512

    @pytest.mark.parametrize(("model", "expected_size"), CLIP_VECTOR_SIZES.items())
    def test_known_model_sizes(self, model: str, expected_size: int) -> None:
        """Test that all known models return their correct vector sizes.

        **Why this test is important:**
          - Documents expected sizes for all supported models
          - Catches regressions in model size mapping

        **What it tests:**
          - Each model in the size map returns correct dimension
        """
        client = CLIPClient(base_url="http://localhost:11434", model=model, is_hosted=False)
        assert client.vector_size == expected_size


class TestCLIPClientEmbedImage:
    """Tests for CLIPClient.embed_image method."""

    @pytest.mark.asyncio
    async def test_embed_image_empty_raises(self) -> None:
        """Test that embed_image rejects empty image bytes.

        **Why this test is important:**
          - Empty images are invalid and waste API calls
          - Fail-fast improves error diagnosis

        **What it tests:**
          - ValueError raised with descriptive message
        """
        client = CLIPClient(base_url="http://localhost:11434", model="clip", is_hosted=False)
        with pytest.raises(ValueError, match="empty"):
            await client.embed_image(b"")


class TestCLIPClientEmbedImageBatch:
    """Tests for CLIPClient.embed_image_batch method."""

    @pytest.mark.asyncio
    async def test_embed_image_batch_empty_raises(self) -> None:
        """Test that embed_image_batch rejects empty list.

        **Why this test is important:**
          - Empty batch is a programming error
          - Early validation prevents downstream confusion

        **What it tests:**
          - ValueError raised for empty list
        """
        client = CLIPClient(base_url="http://localhost:11434", model="clip", is_hosted=False)
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
        client = CLIPClient(base_url="http://localhost:11434", model="clip", is_hosted=False)
        images = [b"foo"] * 10  # Exceeds default max of 8

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await client.embed_image_batch(images)


class TestCLIPClientAsync:
    """Tests for async methods."""

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_async_returns_embedding(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_image returns embedding vector.

        **Why this test is important:**
          - Async operations are used in Ray workers for parallelism
          - Must return same format as sync version

        **What it tests:**
          - Returns embedding from API response
          - Async client is used correctly
        """
        # Local CLIP API returns list of {vector: [...]}
        expected_embedding = [0.1] * 512
        local_clip_response = [{"vector": expected_embedding}]
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = local_clip_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        result = await client.embed_image(b"fake image")

        assert result == expected_embedding
        mock_client.post.assert_called_once()

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_batch_async_returns_vectors(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_image_batch returns vectors for all images.

        **Why this test is important:**
          - Async batch is main code path for Ray workers
          - Must process all images correctly

        **What it tests:**
          - Returns correct number of embeddings
          - Single request with multiple images (local CLIP format)
        """
        # Local CLIP returns list of {vector: [...]}
        mock_response = [{"vector": [0.1] * 512}, {"vector": [0.1] * 512}]
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = mock_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        images = [b"img1", b"img2"]
        result = await client.embed_image_batch(images)

        assert len(result) == 2
        assert all(len(v) == 512 for v in result)


class TestCLIPClientErrorHandling:
    """Tests for error handling behavior."""

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_raises_upstream_error_on_request_error(
        self, mock_async_client_cls: MagicMock
    ) -> None:
        """Test that embed_image raises UpstreamError on network failure.

        **Why this test is important:**
          - Network errors must be surfaced with proper error type
          - UpstreamError enables circuit breaker integration

        **What it tests:**
          - RequestException is caught and wrapped
          - Error message is descriptive
        """
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(UpstreamError, match="Clip embed_image failed"):
            await client.embed_image(b"fake image")

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_raises_upstream_error_on_missing_embedding(
        self, mock_async_client_cls: MagicMock
    ) -> None:
        """Test that embed_image handles missing embedding in response.

        **Why this test is important:**
          - API may return malformed responses
          - Must fail clearly rather than silently corrupt data

        **What it tests:**
          - Empty response triggers UpstreamError
          - Error message indicates unexpected format
        """
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {}
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(UpstreamError, match="Unexpected response"):
            await client.embed_image(b"fake image")

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_raises_upstream_error_on_http_error(
        self, mock_async_client_cls: MagicMock
    ) -> None:
        """Test that embed_image handles HTTP error status codes.

        **Why this test is important:**
          - Server errors (5xx) must trigger circuit breaker
          - Consistent error handling across failure modes

        **What it tests:**
          - HTTP errors are wrapped in UpstreamError
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(UpstreamError, match="Clip embed_image failed"):
            await client.embed_image(b"fake image")

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_async_raises_upstream_error_on_http_error(
        self, mock_async_client_cls: MagicMock
    ) -> None:
        """Test that embed_image_async handles HTTP errors.

        **Why this test is important:**
          - Async path must have same error behavior as sync
          - Enables consistent circuit breaker integration

        **What it tests:**
          - httpx.HTTPError is wrapped in UpstreamError
        """
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(UpstreamError, match="Clip embed_image failed"):
            await client.embed_image(b"fake image")

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_async_raises_upstream_error_on_missing_embedding(
        self, mock_async_client_cls: MagicMock
    ) -> None:
        """Test that embed_image handles missing embedding in response.

        **Why this test is important:**
          - Async path must validate response format
          - Malformed responses must be caught

        **What it tests:**
          - Empty response triggers UpstreamError
        """
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {}  # Not a list of {vector: [...]}
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(UpstreamError, match="Unexpected response"):
            await client.embed_image(b"fake image")


class TestCLIPClientCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_async_circuit_breaker_starts_closed(self) -> None:
        """Test that async circuit breaker starts in closed state.

        **Why this test is important:**
          - Async breaker is used in Ray workers
          - Must allow initial requests

        **What it tests:**
          - Async breaker exists and is closed
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)

        assert client._async_breaker is not None
        assert client._async_breaker.current_state == aiobreaker.state.CircuitBreakerState.CLOSED


class TestCLIPClientEmbeddingProviderSatisfaction:
    """Tests for EmbeddingProvider abstract class implementation."""

    def test_implements_abc(self) -> None:
        """Test that CLIPClient implements EmbeddingProvider ABC.

        **Why this test is important:**
          - Protocol compliance enables dependency injection
          - Allows swapping implementations in tests

        **What it tests:**
          - isinstance check passes for protocol
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        assert isinstance(client, EmbeddingProvider)

    def test_has_required_methods(self) -> None:
        """Test that CLIPClient has all required abc methods."""

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)

        for method in EmbeddingProvider.__abstractmethods__:
            assert hasattr(client, method)


class TestCLIPClientFromConfig:
    """Tests for CLIPClient.from_config factory method."""

    def test_creates_from_config(self) -> None:
        """Test that from_config creates client from EmbeddingConfig.

        **Why this test is important:**
          - Factory method is the standard way to create clients
          - Must correctly map all config fields to client attributes

        **What it tests:**
          - All config values are transferred to client
          - Client is usable after creation
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.LOCAL_CLIP,
            clip_url="http://test:11434",
            clip_model="llava",
            clip_timeout=90,
            clip_circuit_breaker_threshold=3,
            clip_circuit_breaker_timeout=20,
            clip_max_batch_size=4,
            clip_vector_size=1024,
        )

        client = CLIPClient.from_config(config)

        assert client.base_url == "http://test:11434"
        assert client.model == "llava"
        assert client.timeout_s == 90
        assert client.circuit_breaker_failure_threshold == 3
        assert client.circuit_breaker_recovery_timeout_s == 20
        assert client.max_batch_size == 4
        assert client.vector_size_override == 1024

    def test_raises_without_url(self) -> None:
        """Test that from_config requires clip_url.

        **Why this test is important:**
          - URL is required for API connectivity
          - Must fail early with clear message

        **What it tests:**
          - ValueError raised for missing URL
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.LOCAL_CLIP,
            clip_model="llava",
        )

        with pytest.raises(ValueError, match="clip_url is required"):
            CLIPClient.from_config(config)

    def test_raises_without_model(self) -> None:
        """Test that from_config requires clip_model.

        **Why this test is important:**
          - Model is required for embedding generation
          - Must fail early with clear message

        **What it tests:**
          - ValueError raised for missing model
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.LOCAL_CLIP,
            clip_url="http://test:11434",
        )

        with pytest.raises(ValueError, match="clip_model is required"):
            CLIPClient.from_config(config)

    def test_hosted_clip_requires_api_key(self) -> None:
        """Test that hosted_clip backend requires CLIP_API_KEY.

        **Why this test is important:**
          - Hosted endpoints typically require auth
          - Missing API key should fail fast

        **What it tests:**
          - ValueError raised when CLIP_API_KEY is missing
        """
        config = EmbeddingConfig(
            provider_type=ProviderType.HOSTED_CLIP,
            clip_url="http://hosted-clip/score",
            clip_model="clip-vit-base-patch32",
        )

        with pytest.raises(ValueError, match="CLIP_API_KEY is required"):
            CLIPClient.from_config(config)


class TestCLIPClientEmbedTextAsync:
    """Tests for CLIPClient.embed_text async method."""

    @pytest.mark.asyncio
    async def test_embed_text_async_returns_embedding(self) -> None:
        """Test that embed_text (async) returns embedding vector.

        **Why this test is important:**
          - Async text embedding used in Ray workers
          - Must match sync behavior

        **What it tests:**
          - Returns vector of expected dimension
          - Async client is used correctly
        """
        # Local CLIP API returns list of {vector: [...]}
        local_clip_response = [{"vector": [0.1] * 512}]
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = local_clip_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with patch.object(client, "_get_async_client", return_value=mock_client):
            result = await client.embed_text("a fluffy cat")

        assert len(result) == 512
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_text_async_empty_raises_value_error(self) -> None:
        """Test that embed_text (async) rejects empty text.

        **Why this test is important:**
          - Consistent validation between sync and async
          - Fail fast on invalid input

        **What it tests:**
          - ValueError raised for empty text
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text("")


class TestCLIPClientEmbedTextBatchAsync:
    """Tests for CLIPClient.embed_text_batch_async method."""

    @pytest.mark.asyncio
    async def test_embed_text_batch_async_returns_multiple_embeddings(self) -> None:
        """Test that embed_text_batch_async returns embeddings for all texts.

        **Why this test is important:**
          - Async batch is main code path for text queries in Ray
          - Must process all texts correctly

        **What it tests:**
          - Returns correct number of embeddings
          - Concurrent execution via asyncio.gather
        """
        # Local CLIP API returns list of {vector: [...]}
        local_clip_response = [{"vector": [0.1] * 512}, {"vector": [0.1] * 512}]
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = local_clip_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)

        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        texts = ["cat", "dog"]
        with patch.object(client, "_get_async_client", new_callable=AsyncMock, return_value=mock_client):
            results = await client.embed_text_batch(texts)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_embed_text_batch_async_empty_list_raises_value_error(self) -> None:
        """Test that embed_text_batch_async rejects empty list.

        **Why this test is important:**
          - Consistent validation between sync and async
          - Empty batch is a programming error

        **What it tests:**
          - ValueError raised for empty list
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_batch([])

    @pytest.mark.asyncio
    async def test_embed_text_batch_async_empty_string_raises_value_error(self) -> None:
        """Test that embed_text_batch_async rejects batches with empty text.

        **Why this test is important:**
          - Any empty text invalidates the batch
          - Consistent with sync batch behavior

        **What it tests:**
          - ValueError raised when batch contains empty string
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava", is_hosted=False)
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_batch(["cat", "", "dog"])

    @pytest.mark.asyncio
    async def test_embed_text_batch_async_exceeds_max_raises_value_error(self) -> None:
        """Test that embed_text_batch rejects oversized batches.

        **Why this test is important:**
          - Large batches can overwhelm the API
          - Batch size limits prevent memory issues

        **What it tests:**
          - ValueError raised when batch exceeds max_batch_size
        """
        client = CLIPClient(
            base_url="http://localhost:11434",
            model="llava",
            is_hosted=False,
            max_batch_size=4,
        )
        texts = ["text"] * 10  # Exceeds max of 4

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await client.embed_text_batch(texts)
