"""Unit tests for CLIPClient image embedding client.

Tests for the CLIPClient class that generates image embeddings via CLIP-compatible
APIs (Ollama LLaVA, etc.).
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from clients.clip import CLIPClient
from clients.interfaces.embedding import ImageEmbeddingProvider
from config import ImageEmbeddingConfig


@pytest.fixture
def mock_clip_session() -> MagicMock:
    """Create a mock requests.Session for testing CLIP client."""
    session = MagicMock(spec=requests.Session)
    session.post = MagicMock()
    return session


@pytest.fixture
def clip_client_with_mock(mock_clip_session: MagicMock) -> CLIPClient:
    """Create a CLIPClient instance with mocked session."""
    client = CLIPClient(
        base_url="http://localhost:11434",
        model="llava",
        timeout_s=60,
    )
    client.set_session(mock_clip_session)
    return client


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
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client.base_url == "http://localhost:11434"
        assert client.model == "llava"
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

    def test_initializes_session(self) -> None:
        """Test that CLIPClient initializes an HTTP session.

        **Why this test is important:**
          - HTTP session is required for making API requests
          - Session reuse improves performance via connection pooling

        **What it tests:**
          - Session is created during initialization
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client.session is not None

    def test_initializes_circuit_breakers(self) -> None:
        """Test that CLIPClient initializes circuit breakers.

        **Why this test is important:**
          - Circuit breakers prevent cascading failures
          - Both sync and async operations need protection

        **What it tests:**
          - Sync circuit breaker is created
          - Async circuit breaker is created
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client._breaker is not None
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
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        assert client.vector_size == 4096

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
        )
        assert client.vector_size == 512

    @pytest.mark.parametrize(
        ("model", "expected_size"),
        [
            ("llava", 4096),
            ("llava:7b", 4096),
            ("llava:13b", 5120),
            ("bakllava", 4096),
            ("clip-vit-base-patch32", 512),
            ("clip-vit-large-patch14", 768),
        ],
    )
    def test_known_model_sizes(self, model: str, expected_size: int) -> None:
        """Test that all known models return their correct vector sizes.

        **Why this test is important:**
          - Documents expected sizes for all supported models
          - Catches regressions in model size mapping

        **What it tests:**
          - Each model in the size map returns correct dimension
        """
        client = CLIPClient(base_url="http://localhost:11434", model=model)
        assert client.vector_size == expected_size


class TestCLIPClientEncoding:
    """Tests for image encoding."""

    def test_encode_image_returns_base64(self) -> None:
        """Test that _encode_image returns valid base64 string.

        **Why this test is important:**
          - Image data must be base64 encoded for API transport
          - Encoding errors would cause API failures

        **What it tests:**
          - Returns string type
          - Base64 decodes back to original bytes
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        image_bytes = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

        result = client._encode_image(image_bytes)

        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == image_bytes

    def test_encode_image_empty_raises(self) -> None:
        """Test that _encode_image rejects empty bytes.

        **Why this test is important:**
          - Empty images are invalid input
          - Fail fast prevents wasted API calls

        **What it tests:**
          - ValueError raised for empty bytes
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        with pytest.raises(ValueError, match="empty"):
            client._encode_image(b"")


class TestCLIPClientEmbedImage:
    """Tests for CLIPClient.embed_image method."""

    @pytest.fixture
    def mock_response(self) -> dict:
        """Create mock embedding response."""
        return {"embedding": [0.1] * 512}

    def test_embed_image_makes_correct_request(
        self,
        clip_client_with_mock: CLIPClient,
        mock_clip_session: MagicMock,
        mock_response: dict,
    ) -> None:
        """Test that embed_image makes correct API request.

        **Why this test is important:**
          - Request format must match API specification
          - Incorrect payloads cause embedding failures

        **What it tests:**
          - POST to correct endpoint
          - Payload includes model, prompt, and images
        """
        mock_clip_session.post.return_value.json.return_value = mock_response
        mock_clip_session.post.return_value.raise_for_status = MagicMock()

        image_bytes = b"fake image data"
        clip_client_with_mock.embed_image(image_bytes)

        # Verify request was made
        mock_clip_session.post.assert_called_once()
        call_args = mock_clip_session.post.call_args
        assert "api/embeddings" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "llava"
        assert call_args[1]["json"]["prompt"] == ""
        assert len(call_args[1]["json"]["images"]) == 1

    def test_embed_image_returns_embedding(
        self,
        clip_client_with_mock: CLIPClient,
        mock_clip_session: MagicMock,
        mock_response: dict,
    ) -> None:
        """Test that embed_image returns embedding vector from response.

        **Why this test is important:**
          - Correct response parsing is critical for downstream use
          - Wrong vector format would break similarity search

        **What it tests:**
          - Returns embedding array from response
          - Vector has expected length
        """
        mock_clip_session.post.return_value.json.return_value = mock_response
        mock_clip_session.post.return_value.raise_for_status = MagicMock()

        result = clip_client_with_mock.embed_image(b"fake image")

        assert result == mock_response["embedding"]
        assert len(result) == 512

    def test_embed_image_empty_raises(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_image rejects empty image bytes.

        **Why this test is important:**
          - Empty images are invalid and waste API calls
          - Fail-fast improves error diagnosis

        **What it tests:**
          - ValueError raised with descriptive message
        """
        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_image(b"")


class TestCLIPClientEmbedImageBatch:
    """Tests for CLIPClient.embed_image_batch method."""

    @pytest.fixture
    def mock_response(self) -> dict:
        """Create mock embedding response."""
        return {"embedding": [0.1] * 512}

    def test_embed_image_batch_empty_raises(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_image_batch rejects empty list.

        **Why this test is important:**
          - Empty batch is a programming error
          - Early validation prevents downstream confusion

        **What it tests:**
          - ValueError raised for empty list
        """
        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_image_batch([])

    def test_embed_image_batch_exceeds_max_raises(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_image_batch rejects oversized batches.

        **Why this test is important:**
          - Large batches can overwhelm the API
          - Batch size limits prevent memory issues

        **What it tests:**
          - ValueError raised when batch exceeds max_batch_size
        """
        images = [b"image"] * 10  # Exceeds default max of 8

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            clip_client_with_mock.embed_image_batch(images)

    def test_embed_image_batch_returns_correct_count(
        self,
        clip_client_with_mock: CLIPClient,
        mock_clip_session: MagicMock,
        mock_response: dict,
    ) -> None:
        """Test that embed_image_batch returns one embedding per image.

        **Why this test is important:**
          - Batch operations must maintain 1:1 mapping
          - Wrong count would corrupt vector index

        **What it tests:**
          - Returns same number of embeddings as input images
          - Each image gets its own API call
        """
        mock_clip_session.post.return_value.json.return_value = mock_response
        mock_clip_session.post.return_value.raise_for_status = MagicMock()

        images = [b"image1", b"image2", b"image3"]
        results = clip_client_with_mock.embed_image_batch(images)

        assert len(results) == 3
        assert mock_clip_session.post.call_count == 3


class TestCLIPClientAsync:
    """Tests for async methods."""

    @pytest.fixture
    def mock_response(self) -> dict:
        """Create mock embedding response."""
        return {"embedding": [0.1] * 512}

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_async_returns_embedding(
        self, mock_async_client_cls: MagicMock, mock_response: dict
    ) -> None:
        """Test that embed_image_async returns embedding vector.

        **Why this test is important:**
          - Async operations are used in Ray workers for parallelism
          - Must return same format as sync version

        **What it tests:**
          - Returns embedding from API response
          - Async client is used correctly
        """
        # Setup mock response
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = mock_response
        mock_post_response.raise_for_status = MagicMock()

        # Setup mock async client
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_post_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        result = await client.embed_image_async(b"fake image")

        assert result == mock_response["embedding"]
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_image_batch_async_empty_raises(self) -> None:
        """Test that embed_image_batch_async rejects empty list.

        **Why this test is important:**
          - Consistent validation between sync and async
          - Fail fast for programming errors

        **What it tests:**
          - ValueError raised for empty list
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        with pytest.raises(ValueError, match="empty"):
            await client.embed_image_batch_async([])

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_batch_async_returns_vectors(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_image_batch_async returns vectors for all images.

        **Why this test is important:**
          - Async batch is main code path for Ray workers
          - Must process all images correctly

        **What it tests:**
          - Returns correct number of embeddings
          - Concurrent execution (asyncio.gather)
        """
        mock_response = {"embedding": [0.1] * 512}
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = mock_response
        mock_post_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_post_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        images = [b"img1", b"img2"]
        result = await client.embed_image_batch_async(images)

        assert len(result) == 2
        assert all(len(v) == 512 for v in result)


class TestCLIPClientErrorHandling:
    """Tests for error handling behavior."""

    def test_embed_image_raises_upstream_error_on_request_error(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that embed_image raises UpstreamError on network failure.

        **Why this test is important:**
          - Network errors must be surfaced with proper error type
          - UpstreamError enables circuit breaker integration

        **What it tests:**
          - RequestException is caught and wrapped
          - Error message is descriptive
        """
        import requests

        mock_clip_session.post.side_effect = requests.RequestException("Connection failed")

        from core.exceptions import UpstreamError

        with pytest.raises(UpstreamError, match="CLIP embedding request failed"):
            clip_client_with_mock.embed_image(b"fake image")

    def test_embed_image_raises_upstream_error_on_missing_embedding(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that embed_image handles missing embedding in response.

        **Why this test is important:**
          - API may return malformed responses
          - Must fail clearly rather than silently corrupt data

        **What it tests:**
          - Empty response triggers UpstreamError
          - Error message indicates unexpected format
        """
        mock_clip_session.post.return_value.json.return_value = {}
        mock_clip_session.post.return_value.raise_for_status = MagicMock()

        from core.exceptions import UpstreamError

        with pytest.raises(UpstreamError, match="Unexpected response format"):
            clip_client_with_mock.embed_image(b"fake image")

    def test_embed_image_raises_upstream_error_on_http_error(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that embed_image handles HTTP error status codes.

        **Why this test is important:**
          - Server errors (5xx) must trigger circuit breaker
          - Consistent error handling across failure modes

        **What it tests:**
          - HTTP errors are wrapped in UpstreamError
        """
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_clip_session.post.return_value = mock_response

        from core.exceptions import UpstreamError

        with pytest.raises(UpstreamError, match="CLIP embedding request failed"):
            clip_client_with_mock.embed_image(b"fake image")

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
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPError("Connection failed")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        from core.exceptions import UpstreamError

        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        with pytest.raises(UpstreamError, match="CLIP embedding request failed"):
            await client.embed_image_async(b"fake image")

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_async_raises_upstream_error_on_missing_embedding(
        self, mock_async_client_cls: MagicMock
    ) -> None:
        """Test that embed_image_async handles missing embedding in response.

        **Why this test is important:**
          - Async path must validate response format
          - Malformed responses must be caught

        **What it tests:**
          - Empty response triggers UpstreamError
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # Missing 'embedding' key
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        from core.exceptions import UpstreamError

        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        with pytest.raises(UpstreamError, match="Unexpected response format"):
            await client.embed_image_async(b"fake image")


class TestCLIPClientCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_circuit_breaker_starts_closed(self) -> None:
        """Test that sync circuit breaker starts in closed state.

        **Why this test is important:**
          - Closed state allows requests to flow
          - Incorrect initial state would block all requests

        **What it tests:**
          - Sync breaker exists and is closed
        """
        import pybreaker

        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client._breaker is not None
        assert client._breaker.current_state == pybreaker.STATE_CLOSED

    def test_async_circuit_breaker_starts_closed(self) -> None:
        """Test that async circuit breaker starts in closed state.

        **Why this test is important:**
          - Async breaker is used in Ray workers
          - Must allow initial requests

        **What it tests:**
          - Async breaker exists and is closed
        """
        import aiobreaker

        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client._async_breaker is not None
        assert client._async_breaker.current_state == aiobreaker.state.CircuitBreakerState.CLOSED

    def test_circuit_breaker_uses_configured_thresholds(self) -> None:
        """Test that circuit breaker respects configuration.

        **Why this test is important:**
          - Thresholds control failure tolerance
          - Configuration must take effect

        **What it tests:**
          - Custom failure threshold is applied
          - Custom recovery timeout is applied
        """
        client = CLIPClient(
            base_url="http://localhost:11434",
            model="llava",
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_s=15,
        )

        assert client._breaker.fail_max == 3
        assert client._breaker.reset_timeout == 15

    def test_embed_image_handles_circuit_breaker_open(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_image fails fast when circuit is open.

        **Why this test is important:**
          - Open circuit should prevent requests
          - Fail-fast protects downstream services

        **What it tests:**
          - UpstreamError raised when circuit is open
          - No actual request made
        """
        import pybreaker

        from core.exceptions import UpstreamError

        # Set circuit breaker to OPEN state
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        mock_breaker.current_state = pybreaker.STATE_OPEN
        object.__setattr__(clip_client_with_mock, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="service is currently unavailable"):
            clip_client_with_mock.embed_image(b"fake image")


class TestCLIPClientProtocol:
    """Tests for ImageEmbeddingProvider protocol compliance."""

    def test_implements_protocol(self) -> None:
        """Test that CLIPClient implements ImageEmbeddingProvider protocol.

        **Why this test is important:**
          - Protocol compliance enables dependency injection
          - Allows swapping implementations in tests

        **What it tests:**
          - isinstance check passes for protocol
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        assert isinstance(client, ImageEmbeddingProvider)

    def test_has_required_methods(self) -> None:
        """Test that CLIPClient has all required protocol methods.

        **Why this test is important:**
          - Protocol methods must exist for interface contract
          - Missing methods would break callers

        **What it tests:**
          - vector_size property exists
          - embed_image method exists
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert hasattr(client, "vector_size")
        assert hasattr(client, "embed_image")
        assert hasattr(client, "embed_image_async")
        assert hasattr(client, "embed_image_batch")
        assert hasattr(client, "embed_image_batch_async")


class TestCLIPClientFromConfig:
    """Tests for CLIPClient.from_config factory method."""

    def test_creates_from_config(self) -> None:
        """Test that from_config creates client from ImageEmbeddingConfig.

        **Why this test is important:**
          - Factory method is the standard way to create clients
          - Must correctly map all config fields to client attributes

        **What it tests:**
          - All config values are transferred to client
          - Client is usable after creation
        """
        config = ImageEmbeddingConfig(
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
        config = ImageEmbeddingConfig(clip_model="llava")

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
        config = ImageEmbeddingConfig(clip_url="http://test:11434")

        with pytest.raises(ValueError, match="clip_model is required"):
            CLIPClient.from_config(config)

    def test_accepts_custom_session(self) -> None:
        """Test that from_config accepts custom HTTP session.

        **Why this test is important:**
          - Custom session enables connection pooling
          - Allows shared session across clients

        **What it tests:**
          - Custom session is stored on client
        """
        config = ImageEmbeddingConfig(
            clip_url="http://test:11434",
            clip_model="llava",
        )
        custom_session = MagicMock()

        client = CLIPClient.from_config(config, session=custom_session)

        assert client._session == custom_session


class TestCLIPClientCleanup:
    """Tests for resource cleanup."""

    def test_close_closes_session(self) -> None:
        """Test that close() releases HTTP session.

        **Why this test is important:**
          - Prevents resource leaks in long-running processes
          - Required for proper shutdown

        **What it tests:**
          - Session is set to None after close
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        # Access session to ensure it's created
        _ = client.session

        client.close()

        assert client._session is None

    def test_set_session_replaces_session(self) -> None:
        """Test that set_session replaces existing session.

        **Why this test is important:**
          - Enables connection pooling with shared session
          - Required for retry session integration

        **What it tests:**
          - New session replaces old one
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        new_session = MagicMock()

        client.set_session(new_session)

        assert client._session == new_session


class TestImageEmbeddingConfigFromEnv:
    """Tests for ImageEmbeddingConfig.from_env factory method."""

    def test_creates_with_defaults(self) -> None:
        """Test that from_env creates config with sensible defaults.

        **Why this test is important:**
          - Defaults enable zero-config local development
          - Must work without environment variables

        **What it tests:**
          - Default URL, model, timeout, and batch size
        """
        with patch.dict("os.environ", {}, clear=True):
            config = ImageEmbeddingConfig.from_env()

            assert config.provider_type == "clip"
            assert config.clip_url == "http://localhost:11434"
            assert config.clip_model == "llava"
            assert config.clip_timeout == 120
            assert config.clip_max_batch_size == 8

    def test_respects_env_vars(self) -> None:
        """Test that from_env reads from environment variables.

        **Why this test is important:**
          - Environment variables configure production deployments
          - All settings must be overridable

        **What it tests:**
          - All CLIP_* environment variables are respected
        """
        env = {
            "CLIP_URL": "http://custom:11434",
            "CLIP_MODEL": "bakllava",
            "CLIP_TIMEOUT": "60",
            "CLIP_CIRCUIT_BREAKER_THRESHOLD": "3",
            "CLIP_CIRCUIT_BREAKER_TIMEOUT": "15",
            "CLIP_MAX_BATCH_SIZE": "4",
            "CLIP_VECTOR_SIZE": "768",
        }

        with patch.dict("os.environ", env, clear=True):
            config = ImageEmbeddingConfig.from_env()

            assert config.clip_url == "http://custom:11434"
            assert config.clip_model == "bakllava"
            assert config.clip_timeout == 60
            assert config.clip_circuit_breaker_threshold == 3
            assert config.clip_circuit_breaker_timeout == 15
            assert config.clip_max_batch_size == 4
            assert config.clip_vector_size == 768

    def test_falls_back_to_ollama_url(self) -> None:
        """Test that from_env falls back to OLLAMA_BASE_URL.

        **Why this test is important:**
          - Backwards compatibility with Ollama-based setups
          - Reduces configuration duplication

        **What it tests:**
          - OLLAMA_BASE_URL used when CLIP_URL not set
        """
        env = {"OLLAMA_BASE_URL": "http://ollama:11434"}

        with patch.dict("os.environ", env, clear=True):
            config = ImageEmbeddingConfig.from_env()

            assert config.clip_url == "http://ollama:11434"


# =============================================================================
# Text Embedding Tests
# =============================================================================


class TestCLIPClientEmbedText:
    """Tests for CLIPClient.embed_text method."""

    def test_embed_text_returns_embedding(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that embed_text returns embedding vector.

        **Why this test is important:**
          - Text embeddings enable cross-modal search
          - Must return vectors in same space as image embeddings

        **What it tests:**
          - Returns vector of expected dimension
          - All elements are floats
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_clip_session.post.return_value = mock_response

        result = clip_client_with_mock.embed_text("a fluffy cat")

        assert len(result) == 512
        assert all(isinstance(x, float) for x in result)

    def test_embed_text_empty_raises_value_error(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_text rejects empty text.

        **Why this test is important:**
          - Empty text produces meaningless embeddings
          - Fail fast prevents wasted API calls

        **What it tests:**
          - ValueError raised with descriptive message
        """
        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_text("")

    def test_embed_text_whitespace_raises_value_error(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_text rejects whitespace-only text.

        **Why this test is important:**
          - Whitespace is effectively empty
          - Must be caught like empty string

        **What it tests:**
          - ValueError raised for whitespace
        """
        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_text("   ")

    def test_embed_text_makes_correct_request_ollama(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that embed_text makes correct request to Ollama backend.

        **Why this test is important:**
          - Ollama API format differs from ai4all/clip
          - Must use prompt field (not texts array)

        **What it tests:**
          - POST to /api/embeddings endpoint
          - Payload uses prompt field
          - No images array in payload
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_clip_session.post.return_value = mock_response

        clip_client_with_mock.embed_text("test query")

        # Check the URL
        call_args = mock_clip_session.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/embeddings"

        # Check the payload has prompt (not images)
        payload = call_args[1]["json"]
        assert payload["model"] == "llava"
        assert payload["prompt"] == "test query"
        assert "images" not in payload

    def test_embed_text_makes_correct_request_clip_backend(self, mock_clip_session: MagicMock) -> None:
        """Test that embed_text makes correct request to ai4all/clip backend.

        **Why this test is important:**
          - ai4all/clip uses different API format than Ollama
          - Must use /embedding/text with texts array

        **What it tests:**
          - POST to /embedding/text endpoint
          - Payload uses texts array format
        """
        client = CLIPClient(
            base_url="http://clip:8000",
            model="ViT-B/32",
            backend="clip",
        )
        client.set_session(mock_clip_session)

        mock_response = MagicMock()
        # ai4all/clip API returns list of {text, vector} objects
        mock_response.json.return_value = [{"text": "test query", "vector": [0.1] * 512}]
        mock_response.raise_for_status = MagicMock()
        mock_clip_session.post.return_value = mock_response

        client.embed_text("test query")

        call_args = mock_clip_session.post.call_args
        # ai4all/clip uses /embedding/text endpoint
        assert call_args[0][0] == "http://clip:8000/embedding/text"

        payload = call_args[1]["json"]
        # ai4all/clip expects {"texts": [...]} format
        assert payload["texts"] == ["test query"]


class TestCLIPClientEmbedTextAsync:
    """Tests for CLIPClient.embed_text_async method."""

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_async_returns_embedding(self, mock_async_client_cls: MagicMock) -> None:
        """Test that embed_text_async returns embedding vector.

        **Why this test is important:**
          - Async text embedding used in Ray workers
          - Must match sync behavior

        **What it tests:**
          - Returns vector of expected dimension
          - Async client is used correctly
        """
        mock_response = {"embedding": [0.1] * 512}

        mock_client = MagicMock()
        mock_async_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_async_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_post_response = MagicMock()
        mock_post_response.json.return_value = mock_response
        mock_post_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)

        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        result = await client.embed_text_async("a fluffy cat")

        assert len(result) == 512
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_text_async_empty_raises_value_error(self) -> None:
        """Test that embed_text_async rejects empty text.

        **Why this test is important:**
          - Consistent validation between sync and async
          - Fail fast on invalid input

        **What it tests:**
          - ValueError raised for empty text
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_async("")


class TestCLIPClientEmbedTextBatch:
    """Tests for CLIPClient.embed_text_batch method."""

    def test_embed_text_batch_returns_multiple_embeddings(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that embed_text_batch returns embeddings for all texts.

        **Why this test is important:**
          - Batch operations improve throughput
          - Must maintain 1:1 mapping with inputs

        **What it tests:**
          - Returns correct number of embeddings
          - Each text gets its own API call
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_clip_session.post.return_value = mock_response

        texts = ["cat", "dog", "bird"]
        results = clip_client_with_mock.embed_text_batch(texts)

        assert len(results) == 3
        assert mock_clip_session.post.call_count == 3

    def test_embed_text_batch_empty_list_raises_value_error(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_text_batch rejects empty list.

        **Why this test is important:**
          - Empty batch is a programming error
          - Consistent with embed_image_batch behavior

        **What it tests:**
          - ValueError raised for empty list
        """
        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_text_batch([])

    def test_embed_text_batch_empty_string_raises_value_error(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that embed_text_batch rejects batches containing empty text.

        **Why this test is important:**
          - Any empty text invalidates the batch
          - Fail fast before partial processing

        **What it tests:**
          - ValueError raised when batch contains empty string
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_clip_session.post.return_value = mock_response

        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_text_batch(["cat", "", "dog"])

    def test_embed_text_batch_exceeds_max_raises_value_error(self, clip_client_with_mock: CLIPClient) -> None:
        """Test that embed_text_batch rejects oversized batches.

        **Why this test is important:**
          - Batch size limits prevent memory issues
          - Consistent with embed_image_batch behavior

        **What it tests:**
          - ValueError raised when batch exceeds max_batch_size
        """
        texts = ["text"] * 10  # Exceeds default max of 8

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            clip_client_with_mock.embed_text_batch(texts)


class TestCLIPClientEmbedTextBatchAsync:
    """Tests for CLIPClient.embed_text_batch_async method."""

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_batch_async_returns_multiple_embeddings(
        self, mock_async_client_cls: MagicMock
    ) -> None:
        """Test that embed_text_batch_async returns embeddings for all texts.

        **Why this test is important:**
          - Async batch is main code path for text queries in Ray
          - Must process all texts correctly

        **What it tests:**
          - Returns correct number of embeddings
          - Concurrent execution via asyncio.gather
        """
        mock_response = {"embedding": [0.1] * 512}

        mock_client = MagicMock()
        mock_async_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_async_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_post_response = MagicMock()
        mock_post_response.json.return_value = mock_response
        mock_post_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)

        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        texts = ["cat", "dog"]
        results = await client.embed_text_batch_async(texts)

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
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_batch_async([])

    @pytest.mark.asyncio
    async def test_embed_text_batch_async_empty_string_raises_value_error(self) -> None:
        """Test that embed_text_batch_async rejects batches with empty text.

        **Why this test is important:**
          - Any empty text invalidates the batch
          - Consistent with sync batch behavior

        **What it tests:**
          - ValueError raised when batch contains empty string
        """
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        with pytest.raises(ValueError, match="empty"):
            await client.embed_text_batch_async(["cat", "", "dog"])


class TestCLIPClientCrossModalSearch:
    """Tests for cross-modal (text-to-image) search scenarios."""

    def test_image_and_text_use_same_vector_size(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that image and text embeddings have the same dimensions.

        **Why this test is important:**
          - Cross-modal search requires vectors in same space
          - Mismatched dimensions would break similarity calculations

        **What it tests:**
          - Image and text embeddings have equal length
        """
        # Mock both image and text embedding responses
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_clip_session.post.return_value = mock_response

        image_embedding = clip_client_with_mock.embed_image(b"fake_image")
        text_embedding = clip_client_with_mock.embed_text("a cat")

        assert len(image_embedding) == len(text_embedding)

    def test_text_embedding_request_differs_from_image(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """Test that text embedding uses different payload than image embedding.

        **Why this test is important:**
          - Text uses prompt field, image uses images array
          - API distinguishes modalities by payload structure

        **What it tests:**
          - Text payload has prompt, no images
          - Image payload has images, empty prompt
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 512}
        mock_response.raise_for_status = MagicMock()
        mock_clip_session.post.return_value = mock_response

        # Make text request
        clip_client_with_mock.embed_text("a cat")
        text_payload = mock_clip_session.post.call_args[1]["json"]

        mock_clip_session.post.reset_mock()

        # Make image request
        clip_client_with_mock.embed_image(b"fake_image")
        image_payload = mock_clip_session.post.call_args[1]["json"]

        # Text request has prompt, no images
        assert "prompt" in text_payload
        assert text_payload["prompt"] == "a cat"
        assert "images" not in text_payload

        # Image request has images, empty prompt
        assert "images" in image_payload
        assert image_payload["prompt"] == ""
