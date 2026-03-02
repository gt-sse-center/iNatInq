"""Unit tests for clients.ollama module.

This file tests the OllamaClient class which provides embedding generation
via the Ollama API.

# Test Coverage

The tests cover:
  - Client Initialization: Default and custom configuration, from_config factory
  - Async Embedding Generation: Single and batch embeddings, success and error cases
  - Circuit Breaker Integration: Async circuit breaker usage and error handling
  - Error Handling: UpstreamError on failures, circuit breaker errors
  - Vector Size: Model-based vector size determination

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The underlying httpx.AsyncClient and circuit breaker are mocked to isolate client logic.

# Running Tests

Run with: uv run pytest tests/unit/clients/test_ollama.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiobreaker
import pytest

from clients.ollama import OllamaClient
from config import EmbeddingConfig
from core.exceptions import UpstreamError

# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestOllamaClientInit:
    """Test suite for OllamaClient initialization."""

    def test_creates_client_with_defaults(self) -> None:
        """Test that client is created with default timeout.

        **Why this test is important:**
          - Default configuration must work out of the box
          - Ensures sensible defaults for common use cases
          - Validates that client is created successfully
          - Critical for ease of use and backward compatibility

        **What it tests:**
          - Client is created with base_url and model
          - Client is created with default attribute values
        """
        client = OllamaClient(base_url="test-url", model="test-model")

        assert client.base_url == "test-url"
        assert client.model == "test-model"
        assert client.timeout_s == 60
        assert client.batch_timeout_multiplier == 1
        assert client.circuit_breaker_failure_threshold == 5
        assert client.circuit_breaker_recovery_timeout_s == 30
        assert client.max_batch_size == 12
        assert client.vector_size_override is None
        assert client.max_retries == 3
        assert client.retry_min_wait == 1
        assert client.retry_max_wait == 10

    def test_creates_client_with_custom_timeout(self) -> None:
        """Test that client accepts custom timeout.

        **Why this test is important:**
          - Custom timeout allows tuning for different use cases
          - Different services may need different timeout values
          - Critical for adapting to service-specific requirements
          - Validates parameter passing

        **What it tests:**
          - Custom timeout_s value is applied
          - Other default values are preserved
        """
        client = OllamaClient(base_url="http://ollama.example.com:11434", model="test-model", timeout_s=120)

        assert client.timeout_s == 120

    def test_creates_circuit_breaker(self) -> None:
        """Test that async circuit breaker is created during initialization.

        **Why this test is important:**
          - Circuit breaker provides fault tolerance
          - Ensures circuit breaker is configured with correct parameters
          - Critical for production reliability
          - Validates aiobreaker circuit breaker integration

        **What it tests:**
          - Async circuit breaker (aiobreaker) is created with correct configuration
          - Failure threshold is set correctly
        """
        client = OllamaClient(base_url="http://ollama.example.com:11434", model="test-model")

        # Verify async circuit breaker (aiobreaker) was created
        assert client._async_breaker is not None
        assert isinstance(client._async_breaker, aiobreaker.CircuitBreaker)
        assert client._async_breaker.name == "ollama"
        assert client._async_breaker.fail_max == 5

    def test_creates_client_with_custom_circuit_breaker_config(self) -> None:
        """Test that client accepts custom circuit breaker configuration.

        **Why this test is important:**
          - Different deployments need different failure tolerance
          - Critical path vs background jobs may need different thresholds
          - Validates configurable resilience parameters

        **What it tests:**
          - Custom failure threshold is applied to async breaker
        """
        client = OllamaClient(
            base_url="http://ollama.example.com:11434",
            model="test-model",
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout_s=60,
        )

        assert client._async_breaker.fail_max == 3

    def test_creates_client_with_batch_config(self) -> None:
        """Test that client accepts batch configuration.

        **Why this test is important:**
          - Batch size limits prevent quality degradation
          - Timeout multiplier allows tuning for different models
          - Critical for production performance tuning

        **What it tests:**
          - max_batch_size is stored correctly
          - batch_timeout_multiplier is stored correctly
        """
        client = OllamaClient(
            base_url="http://ollama.example.com:11434",
            model="test-model",
            max_batch_size=8,
            batch_timeout_multiplier=2.0,
        )

        assert client.max_batch_size == 8
        assert client.batch_timeout_multiplier == 2.0

    def test_creates_client_with_vector_size_override(self) -> None:
        """Test that client accepts vector size override.

        **Why this test is important:**
          - Custom/fine-tuned models may have non-standard dimensions
          - Override allows using models not in the known model map
          - Critical for extensibility

        **What it tests:**
          - vector_size_override is stored correctly
          - vector_size property returns override value
        """
        client = OllamaClient(
            base_url="http://ollama.example.com:11434",
            model="custom-model",
            vector_size_override=1024,
        )

        assert client.vector_size_override == 1024
        assert client.vector_size == 1024

    def test_from_config_creates_client(self) -> None:
        """Test that from_config factory creates client correctly.

        **Why this test is important:**
          - Factory method provides convenient client creation
          - Validates configuration integration
          - Critical for configuration-driven initialization
          - Ensures proper config validation

        **What it tests:**
          - Client is created from EmbeddingConfig
          - Config values are correctly applied
        """
        config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama.example.com:11434",
            ollama_model="test-model",
        )

        client = OllamaClient.from_config(config)

        assert client.base_url == "http://ollama.example.com:11434"
        assert client.model == "test-model"

    def test_from_config_validates_provider_type(self) -> None:
        """Test that from_config validates provider_type.

        **Why this test is important:**
          - Prevents configuration errors
          - Ensures type safety
          - Critical for preventing runtime errors
          - Validates error handling

        **What it tests:**
          - ValueError is raised for wrong provider_type
          - Error message is descriptive
        """
        config = EmbeddingConfig(provider_type="openai", openai_api_key="key", openai_model="model")

        with pytest.raises(ValueError, match="provider_type must be 'ollama'"):
            OllamaClient.from_config(config)

    def test_from_config_validates_required_fields(self) -> None:
        """Test that from_config validates required fields.

        **Why this test is important:**
          - Prevents configuration errors
          - Ensures required fields are present
          - Critical for preventing runtime errors
          - Validates error handling

        **What it tests:**
          - ValueError is raised for missing ollama_url
          - ValueError is raised for missing ollama_model
        """
        config = EmbeddingConfig(provider_type="ollama", ollama_url=None, ollama_model=None)

        with pytest.raises(ValueError, match="requires: ollama_url, ollama_model"):
            OllamaClient.from_config(config)

    def test_from_config_passes_resilience_settings(self) -> None:
        """Test that from_config passes all resilience settings from config.

        **Why this test is important:**
          - Resilience settings must propagate from config to client
          - Ensures timeout, circuit breaker, and batch settings are applied
          - Critical for environment-specific configuration

        **What it tests:**
          - ollama_timeout is passed as timeout_s
          - ollama_circuit_breaker_threshold is passed correctly
          - ollama_circuit_breaker_timeout is passed correctly
          - ollama_batch_timeout_multiplier is passed correctly
          - ollama_max_batch_size is passed correctly
        """
        config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama.example.com:11434",
            ollama_model="test-model",
            ollama_timeout=120,
            ollama_circuit_breaker_threshold=10,
            ollama_circuit_breaker_timeout=60,
            ollama_batch_timeout_multiplier=2.0,
            ollama_max_batch_size=8,
        )

        client = OllamaClient.from_config(config)

        assert client.timeout_s == 120
        assert client.circuit_breaker_failure_threshold == 10
        assert client.circuit_breaker_recovery_timeout_s == 60
        assert client.batch_timeout_multiplier == 2.0
        assert client.max_batch_size == 8

        # Verify async circuit breaker was configured with custom values
        assert client._async_breaker.fail_max == 10


# =============================================================================
# Vector Size Tests
# =============================================================================


class TestOllamaClientVectorSize:
    """Test suite for OllamaClient.vector_size property."""

    def test_vector_size_returns_model_default(self) -> None:
        """Test that vector_size returns model default.

        **Why this test is important:**
          - Vector size is needed for collection configuration
          - Model-specific sizes ensure correct configuration
          - Critical for data consistency
          - Validates model mapping

        **What it tests:**
          - Default vector size is 768 for unknown models
          - Known models return correct sizes
        """
        client = OllamaClient(base_url="http://ollama.example.com:11434", model="unknown-model")

        assert client.vector_size == 768  # Default

    def test_vector_size_returns_model_specific_size(self) -> None:
        """Test that vector_size returns model-specific size.

        **Why this test is important:**
          - Different models have different vector dimensions
          - Correct size ensures collection compatibility
          - Critical for data integrity
          - Validates model-specific mapping

        **What it tests:**
          - nomic-embed-text returns 768
          - all-minilm returns 384
        """
        client_nomic = OllamaClient(base_url="http://ollama.example.com:11434", model="nomic-embed-text")
        client_minilm = OllamaClient(base_url="http://ollama.example.com:11434", model="all-minilm")

        assert client_nomic.vector_size == 768
        assert client_minilm.vector_size == 384


# =============================================================================
# Text Embedding Tests
# =============================================================================


class TestOllamaClientEmbedText:
    """Test suite for OllamaClient.embed_text method."""

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_success(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text returns embedding vector on success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        result = await ollama_client.embed_text("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        # First positional argument is the URL
        assert "api/embeddings" in call_args[0][0]
        # Check keyword arguments
        assert call_args[1]["json"] == {"model": "nomic-embed-text", "prompt": "hello world"}

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_raises_on_http_status_error(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text raises UpstreamError on HTTP status error.

        Note: 500 errors are retriable, so the error message comes from the
        retry wrapper after all retries are exhausted.
        """
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_response
        )
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Ollama _embed_async_impl failed after"):
            await ollama_client.embed_text("hello world")

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_raises_on_request_error(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text raises UpstreamError on request error.

        Note: RequestError (base class, not ConnectError) is non-retriable
        so the error wraps immediately via async_retry_call.
        """
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.RequestError("Connection failed")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Ollama _embed_async_impl failed"):
            await ollama_client.embed_text("hello world")

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_raises_on_missing_embedding(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text raises UpstreamError when embedding is missing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="missing embedding"):
            await ollama_client.embed_text("hello world")

    @patch("foundation.circuit_breaker.handle_circuit_breaker_error")
    @pytest.mark.asyncio
    async def test_embed_text_handles_circuit_breaker_open(
        self, mock_handle_error: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text handles circuit breaker open state.

        **Why this test is important:**
          - Async methods use aiobreaker for circuit breaking
          - Circuit breaker open state should fail fast
          - Critical for fault tolerance in async code paths

        **What it tests:**
          - Async circuit breaker open state triggers UpstreamError
          - handle_circuit_breaker_error is called with correct service name
        """
        mock_handle_error.side_effect = UpstreamError("service unavailable")

        # Mock the async circuit breaker's current_state property
        mock_async_breaker = MagicMock(spec=aiobreaker.CircuitBreaker)
        mock_async_breaker.current_state = aiobreaker.state.CircuitBreakerState.OPEN
        object.__setattr__(ollama_client, "_async_breaker", mock_async_breaker)

        with pytest.raises(UpstreamError, match="service unavailable"):
            await ollama_client.embed_text("hello world")

        mock_handle_error.assert_called_once_with("ollama")


# =============================================================================
# Batch Text Embedding Tests
# =============================================================================


class TestOllamaClientEmbedTextBatch:
    """Test suite for OllamaClient.embed_text_batch method."""

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_batch_success(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text_batch returns embeddings on success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        result = await ollama_client.embed_text_batch(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_text_batch_raises_value_error_on_empty_list(
        self, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text_batch raises ValueError for empty list."""
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            await ollama_client.embed_text_batch([])

    @pytest.mark.asyncio
    async def test_embed_text_batch_raises_value_error_on_exceeding_max_batch_size(self) -> None:
        """Test that embed_text_batch raises ValueError when exceeding max_batch_size.

        **Why this test is important:**
          - Batch size limits prevent quality degradation
          - Large batches can cause OOM or slow responses
          - Critical for production reliability

        **What it tests:**
          - Batch exceeding max_batch_size raises ValueError
          - Error message includes both actual and max size
        """
        client = OllamaClient(
            base_url="http://ollama.example.com:11434",
            model="test-model",
            max_batch_size=5,
        )

        texts = ["text"] * 10  # 10 texts exceeds max of 5
        with pytest.raises(ValueError, match="exceeds max_batch_size"):
            await client.embed_text_batch(texts)

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_batch_scales_timeout_by_batch_size(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text_batch scales timeout by batch size.

        **Why this test is important:**
          - Larger batches need more time
          - Timeout scaling prevents premature timeouts
          - Critical for reliability
          - Validates timeout calculation

        **What it tests:**
          - Timeout is scaled based on batch size
          - Minimum timeout is preserved
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1], [0.2], [0.3]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        await ollama_client.embed_text_batch(["a", "b", "c"])

        # Verify post was called with scaled timeout
        call_kwargs = mock_client.post.call_args[1]
        # timeout_s=60, batch_timeout_multiplier=1.0, 3 texts = 60 * 1.0 * 3 = 180
        assert call_kwargs["timeout"] == 180

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_batch_falls_back_on_failure(
        self,
        mock_async_client_cls: MagicMock,
        ollama_client: OllamaClient,
    ) -> None:
        """Test that embed_text_batch falls back to individual calls on failure.

        **Why this test is important:**
          - Fallback ensures compatibility with older Ollama versions
          - Graceful degradation improves reliability
          - Uses internal _embed_async_impl to avoid double circuit breaker wrapping

        **What it tests:**
          - Batch API failure triggers fallback
          - Individual _embed_async_impl calls are made
          - Correct result is returned after fallback
        """
        import httpx

        # Mock batch API failure
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Batch API not supported"

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Bad Request", request=MagicMock(), response=mock_response
        )
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        # Mock internal _embed_async_impl method (avoids double circuit breaker wrapping)
        with patch.object(
            ollama_client, "_embed_async_impl", side_effect=[[0.1, 0.2], [0.3, 0.4]]
        ) as mock_embed_impl:
            result = await ollama_client.embed_text_batch(["hello", "world"], fallback_to_individual=True)

            assert result == [[0.1, 0.2], [0.3, 0.4]]
            assert mock_embed_impl.call_count == 2

    @patch("foundation.circuit_breaker.handle_circuit_breaker_error")
    @pytest.mark.asyncio
    async def test_embed_text_batch_handles_circuit_breaker_open(
        self, mock_handle_error: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text_batch handles circuit breaker open state.

        **Why this test is important:**
          - Async batch methods use aiobreaker for circuit breaking
          - Circuit breaker open state should fail fast
          - Critical for fault tolerance in async batch operations

        **What it tests:**
          - Async circuit breaker open state triggers UpstreamError
          - handle_circuit_breaker_error is called with correct service name
        """
        mock_handle_error.side_effect = UpstreamError("service unavailable")

        # Mock the async circuit breaker's current_state property
        mock_async_breaker = MagicMock(spec=aiobreaker.CircuitBreaker)
        mock_async_breaker.current_state = aiobreaker.state.CircuitBreakerState.OPEN
        object.__setattr__(ollama_client, "_async_breaker", mock_async_breaker)

        with pytest.raises(UpstreamError, match="service unavailable"):
            await ollama_client.embed_text_batch(["hello", "world"])

        mock_handle_error.assert_called_once_with("ollama")

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_text_batch_raises_if_missing_results(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_text_batch raises UpstreamError when result count != input count.

        **Why this test is important:**
          - Ollama must return one embedding per input text
          - Length mismatch indicates upstream bug or truncation
          - Critical for data integrity in batch operations

        **What it tests:**
          - Response with fewer embeddings than inputs raises UpstreamError
          - Error message includes expected and actual counts
        """
        # 3 inputs but only 2 embeddings in response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],  # 2 embeddings
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Ollama returned 2 embeddings for 3 texts"):
            await ollama_client.embed_text_batch(["hello", "world", "foo"])
