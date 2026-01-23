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
        """CLIPClient should initialize with base_url and model."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client.base_url == "http://localhost:11434"
        assert client.model == "llava"
        assert client.timeout_s == 120  # Default
        assert client.max_batch_size == 8  # Default

    def test_creates_with_custom_params(self) -> None:
        """CLIPClient should accept custom configuration."""
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
        """CLIPClient should initialize a requests session."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client.session is not None

    def test_initializes_circuit_breakers(self) -> None:
        """CLIPClient should initialize sync and async circuit breakers."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client._breaker is not None
        assert client._async_breaker is not None


class TestCLIPClientVectorSize:
    """Tests for CLIPClient.vector_size property."""

    def test_returns_known_model_size(self) -> None:
        """vector_size should return correct size for known models."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        assert client.vector_size == 4096

    def test_returns_override_when_set(self) -> None:
        """vector_size should return override when specified."""
        client = CLIPClient(
            base_url="http://localhost:11434",
            model="llava",
            vector_size_override=768,
        )
        assert client.vector_size == 768

    def test_returns_default_for_unknown_model(self) -> None:
        """vector_size should return 512 for unknown models."""
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
        """vector_size should return correct size for each known model."""
        client = CLIPClient(base_url="http://localhost:11434", model=model)
        assert client.vector_size == expected_size


class TestCLIPClientEncoding:
    """Tests for image encoding."""

    def test_encode_image_returns_base64(self) -> None:
        """_encode_image should return base64 string."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        image_bytes = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

        result = client._encode_image(image_bytes)

        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert decoded == image_bytes

    def test_encode_image_empty_raises(self) -> None:
        """_encode_image should raise ValueError for empty bytes."""
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
        """embed_image should make POST request with correct payload."""
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
        """embed_image should return embedding vector."""
        mock_clip_session.post.return_value.json.return_value = mock_response
        mock_clip_session.post.return_value.raise_for_status = MagicMock()

        result = clip_client_with_mock.embed_image(b"fake image")

        assert result == mock_response["embedding"]
        assert len(result) == 512

    def test_embed_image_empty_raises(self, clip_client_with_mock: CLIPClient) -> None:
        """embed_image should raise ValueError for empty bytes."""
        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_image(b"")


class TestCLIPClientEmbedImageBatch:
    """Tests for CLIPClient.embed_image_batch method."""

    @pytest.fixture
    def mock_response(self) -> dict:
        """Create mock embedding response."""
        return {"embedding": [0.1] * 512}

    def test_embed_image_batch_empty_raises(self, clip_client_with_mock: CLIPClient) -> None:
        """embed_image_batch should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="empty"):
            clip_client_with_mock.embed_image_batch([])

    def test_embed_image_batch_exceeds_max_raises(self, clip_client_with_mock: CLIPClient) -> None:
        """embed_image_batch should raise ValueError when exceeding max_batch_size."""
        images = [b"image"] * 10  # Exceeds default max of 8

        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            clip_client_with_mock.embed_image_batch(images)

    def test_embed_image_batch_returns_correct_count(
        self,
        clip_client_with_mock: CLIPClient,
        mock_clip_session: MagicMock,
        mock_response: dict,
    ) -> None:
        """embed_image_batch should return one embedding per image."""
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
        """embed_image_async should return embedding vector."""
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
        """embed_image_batch_async should raise ValueError for empty list."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        with pytest.raises(ValueError, match="empty"):
            await client.embed_image_batch_async([])

    @patch("clients.clip.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_image_batch_async_returns_vectors(self, mock_async_client_cls: MagicMock) -> None:
        """embed_image_batch_async should return vectors for all images."""
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
        """embed_image should raise UpstreamError on request failure."""
        import requests

        mock_clip_session.post.side_effect = requests.RequestException("Connection failed")

        from core.exceptions import UpstreamError

        with pytest.raises(UpstreamError, match="CLIP embedding request failed"):
            clip_client_with_mock.embed_image(b"fake image")

    def test_embed_image_raises_upstream_error_on_missing_embedding(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """embed_image should raise UpstreamError when embedding is missing."""
        mock_clip_session.post.return_value.json.return_value = {}
        mock_clip_session.post.return_value.raise_for_status = MagicMock()

        from core.exceptions import UpstreamError

        with pytest.raises(UpstreamError, match="Unexpected response format"):
            clip_client_with_mock.embed_image(b"fake image")

    def test_embed_image_raises_upstream_error_on_http_error(
        self, clip_client_with_mock: CLIPClient, mock_clip_session: MagicMock
    ) -> None:
        """embed_image should raise UpstreamError on HTTP error status."""
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
        """embed_image_async should raise UpstreamError on HTTP error."""
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
        """embed_image_async should raise UpstreamError when embedding is missing."""
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
        """Circuit breaker should start in closed state."""
        import pybreaker

        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client._breaker is not None
        assert client._breaker.current_state == pybreaker.STATE_CLOSED

    def test_async_circuit_breaker_starts_closed(self) -> None:
        """Async circuit breaker should start in closed state."""
        import aiobreaker

        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert client._async_breaker is not None
        assert client._async_breaker.current_state == aiobreaker.state.CircuitBreakerState.CLOSED

    def test_circuit_breaker_uses_configured_thresholds(self) -> None:
        """Circuit breaker should use configured failure threshold."""
        client = CLIPClient(
            base_url="http://localhost:11434",
            model="llava",
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_s=15,
        )

        assert client._breaker.fail_max == 3
        assert client._breaker.reset_timeout == 15

    def test_embed_image_handles_circuit_breaker_open(self, clip_client_with_mock: CLIPClient) -> None:
        """embed_image should raise UpstreamError when circuit is open."""
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
        """CLIPClient should implement ImageEmbeddingProvider protocol."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        assert isinstance(client, ImageEmbeddingProvider)

    def test_has_required_methods(self) -> None:
        """CLIPClient should have all required protocol methods."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")

        assert hasattr(client, "vector_size")
        assert hasattr(client, "embed_image")
        assert hasattr(client, "embed_image_async")
        assert hasattr(client, "embed_image_batch")
        assert hasattr(client, "embed_image_batch_async")


class TestCLIPClientFromConfig:
    """Tests for CLIPClient.from_config factory method."""

    def test_creates_from_config(self) -> None:
        """from_config should create CLIPClient from ImageEmbeddingConfig."""
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
        """from_config should raise ValueError if clip_url is missing."""
        config = ImageEmbeddingConfig(clip_model="llava")

        with pytest.raises(ValueError, match="clip_url is required"):
            CLIPClient.from_config(config)

    def test_raises_without_model(self) -> None:
        """from_config should raise ValueError if clip_model is missing."""
        config = ImageEmbeddingConfig(clip_url="http://test:11434")

        with pytest.raises(ValueError, match="clip_model is required"):
            CLIPClient.from_config(config)

    def test_accepts_custom_session(self) -> None:
        """from_config should accept custom session."""
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
        """close() should close the HTTP session."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        # Access session to ensure it's created
        _ = client.session

        client.close()

        assert client._session is None

    def test_set_session_replaces_session(self) -> None:
        """set_session should replace existing session."""
        client = CLIPClient(base_url="http://localhost:11434", model="llava")
        new_session = MagicMock()

        client.set_session(new_session)

        assert client._session == new_session


class TestImageEmbeddingConfigFromEnv:
    """Tests for ImageEmbeddingConfig.from_env factory method."""

    def test_creates_with_defaults(self) -> None:
        """from_env should create config with defaults."""
        with patch.dict("os.environ", {}, clear=True):
            config = ImageEmbeddingConfig.from_env()

            assert config.provider_type == "clip"
            assert config.clip_url == "http://localhost:11434"
            assert config.clip_model == "llava"
            assert config.clip_timeout == 120
            assert config.clip_max_batch_size == 8

    def test_respects_env_vars(self) -> None:
        """from_env should read from environment variables."""
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
        """from_env should fall back to OLLAMA_BASE_URL if CLIP_URL not set."""
        env = {"OLLAMA_BASE_URL": "http://ollama:11434"}

        with patch.dict("os.environ", env, clear=True):
            config = ImageEmbeddingConfig.from_env()

            assert config.clip_url == "http://ollama:11434"
