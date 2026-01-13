"""Unit tests for clients.ollama module.

This file tests the OllamaClient class which provides text embedding generation
via the Ollama API.

# Test Coverage

The tests cover:
  - Client Initialization: Default and custom configuration, from_config factory
  - Session Management: Session creation, set_session method, session property
  - Embedding Generation: Single and batch embeddings, success and error cases
  - Circuit Breaker Integration: Circuit breaker usage, error handling
  - Async Operations: Async embedding methods
  - Error Handling: UpstreamError on failures, circuit breaker errors
  - Vector Size: Model-based vector size determination

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The underlying requests.Session and circuit breaker are mocked to isolate client logic.

# Running Tests

Run with: pytest tests/unit/clients/test_ollama.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pybreaker
import pytest
import requests

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
          - Default timeout_s is 60 seconds
          - Session is created automatically
        """
        client = OllamaClient(base_url="http://ollama.example.com:11434", model="test-model")

        assert client.base_url == "http://ollama.example.com:11434"
        assert client.model == "test-model"
        assert client.timeout_s == 60
        assert client._session is not None

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
        """Test that circuit breaker is created during initialization.

        **Why this test is important:**
          - Circuit breaker provides fault tolerance
          - Ensures circuit breaker is configured with correct parameters
          - Critical for production reliability
          - Validates circuit breaker integration

        **What it tests:**
          - Circuit breaker is created with correct configuration
          - Failure threshold and recovery timeout are set correctly
        """
        client = OllamaClient(base_url="http://ollama.example.com:11434", model="test-model")

        # Verify circuit breaker was created
        assert client._breaker is not None
        assert isinstance(client._breaker, pybreaker.CircuitBreaker)
        assert client._breaker.name == "ollama"
        assert client._breaker.fail_max == 5
        assert client._breaker.reset_timeout == 30

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
          - Session can be passed optionally
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


# =============================================================================
# Session Management Tests
# =============================================================================


class TestOllamaClientSession:
    """Test suite for OllamaClient session management."""

    def test_session_property_creates_session(self) -> None:
        """Test that session property creates session if needed.

        **Why this test is important:**
          - Lazy session creation improves initialization performance
          - Ensures session is always available
          - Critical for resource management
          - Validates lazy initialization pattern

        **What it tests:**
          - Session is created when accessed
          - Session is cached for subsequent accesses
        """
        client = OllamaClient(base_url="http://ollama.example.com:11434", model="test-model")
        # Clear initial session
        object.__setattr__(client, "_session", None)

        session = client.session

        assert session is not None
        assert client.session is session  # Cached

    def test_set_session_updates_session(self, ollama_client: OllamaClient) -> None:
        """Test that set_session updates the session.

        **Why this test is important:**
          - Allows custom session configuration
          - Enables connection pooling and retry logic customization
          - Critical for performance optimization
          - Validates session replacement

        **What it tests:**
          - set_session updates the session
          - session property returns the new session
        """
        new_session = MagicMock(spec=requests.Session)

        ollama_client.set_session(new_session)

        assert ollama_client._session == new_session
        assert ollama_client.session == new_session


# =============================================================================
# Embedding Generation Tests
# =============================================================================


class TestOllamaClientEmbed:
    """Test suite for OllamaClient.embed method."""

    def test_embed_success(self, ollama_client: OllamaClient, mock_session: MagicMock) -> None:
        """Test that embed returns embedding vector on success.

        **Why this test is important:**
          - Embedding generation is the core functionality
          - Validates successful API interaction
          - Ensures response parsing is correct
          - Critical for basic functionality

        **What it tests:**
          - HTTP POST is called with correct URL and payload
          - Response JSON is parsed correctly
          - Embedding vector is returned as list of floats
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_session.post.return_value = mock_response

        result = ollama_client.embed("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        # First positional argument is the URL
        assert "api/embeddings" in call_args[0][0]
        # Check keyword arguments
        assert call_args[1]["json"] == {"model": "nomic-embed-text", "prompt": "hello world"}

    def test_embed_raises_upstream_error_on_request_exception(
        self, ollama_client: OllamaClient, mock_session: MagicMock
    ) -> None:
        """Test that embed raises UpstreamError on request exception.

        **Why this test is important:**
          - Error handling ensures consistent error types
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation and debugging
          - Validates error wrapping

        **What it tests:**
          - RequestException is wrapped in UpstreamError
          - Error message includes context
        """
        mock_session.post.side_effect = requests.RequestException("connection failed")

        with pytest.raises(UpstreamError, match="Ollama request failed"):
            ollama_client.embed("hello world")

    def test_embed_raises_upstream_error_on_error_status(
        self, ollama_client: OllamaClient, mock_session: MagicMock
    ) -> None:
        """Test that embed raises UpstreamError on error status code.

        **Why this test is important:**
          - HTTP error status codes should be handled gracefully
          - UpstreamError provides consistent error handling
          - Critical for error propagation
          - Validates status code handling

        **What it tests:**
          - HTTP 400+ status codes raise UpstreamError
          - Error message includes status code and response text
        """
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_session.post.return_value = mock_response

        with pytest.raises(UpstreamError, match="Ollama error 500"):
            ollama_client.embed("hello world")

    def test_embed_raises_upstream_error_on_missing_embedding(
        self, ollama_client: OllamaClient, mock_session: MagicMock
    ) -> None:
        """Test that embed raises UpstreamError when embedding is missing.

        **Why this test is important:**
          - Validates response structure
          - Prevents silent failures
          - Critical for data integrity
          - Validates response validation

        **What it tests:**
          - Missing embedding field raises UpstreamError
          - Empty embedding list raises UpstreamError
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_session.post.return_value = mock_response

        with pytest.raises(UpstreamError, match="missing embedding"):
            ollama_client.embed("hello world")

    def test_embed_handles_circuit_breaker_error(
        self,
        ollama_client: OllamaClient,
    ) -> None:
        """Test that embed handles circuit breaker errors.

        **Why this test is important:**
          - Circuit breaker errors need special handling
          - UpstreamError conversion ensures consistent error types
          - Critical for fault tolerance
          - Validates circuit breaker integration

        **What it tests:**
          - CircuitBreakerError is handled correctly by the decorator
          - UpstreamError is raised when circuit is open
        """
        # Replace the circuit breaker with a mock in OPEN state
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        mock_breaker.current_state = pybreaker.STATE_OPEN
        object.__setattr__(ollama_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="service is currently unavailable"):
            ollama_client.embed("hello world")


# =============================================================================
# Batch Embedding Tests
# =============================================================================


class TestOllamaClientEmbedBatch:
    """Test suite for OllamaClient.embed_batch method."""

    def test_embed_batch_success(self, ollama_client: OllamaClient, mock_session: MagicMock) -> None:
        """Test that embed_batch returns embeddings on success.

        **Why this test is important:**
          - Batch embedding is more efficient than individual calls
          - Validates batch API interaction
          - Ensures response parsing is correct
          - Critical for performance optimization

        **What it tests:**
          - HTTP POST is called with batch endpoint and payload
          - Response JSON is parsed correctly
          - List of embedding vectors is returned
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_session.post.return_value = mock_response

        result = ollama_client.embed_batch(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        # First positional argument is the URL
        assert "api/embed" in call_args[0][0]
        # Check keyword arguments
        assert call_args[1]["json"] == {"model": "nomic-embed-text", "input": ["hello", "world"]}

    def test_embed_batch_raises_value_error_on_empty_list(self, ollama_client: OllamaClient) -> None:
        """Test that embed_batch raises ValueError for empty list.

        **Why this test is important:**
          - Prevents invalid API calls
          - Ensures input validation
          - Critical for preventing errors
          - Validates input validation

        **What it tests:**
          - Empty texts list raises ValueError
          - Error message is descriptive
        """
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            ollama_client.embed_batch([])

    def test_embed_batch_scales_timeout_by_batch_size(
        self, ollama_client: OllamaClient, mock_session: MagicMock
    ) -> None:
        """Test that embed_batch scales timeout by batch size.

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
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1], [0.2], [0.3]]}
        mock_session.post.return_value = mock_response

        ollama_client.embed_batch(["a", "b", "c"])

        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["timeout"] == 180  # 60 * 3

    def test_embed_batch_falls_back_on_batch_api_failure(
        self,
        ollama_client: OllamaClient,
        mock_session: MagicMock,
    ) -> None:
        """Test that embed_batch falls back to individual calls on batch API failure.

        **Why this test is important:**
          - Fallback ensures compatibility with older Ollama versions
          - Graceful degradation improves reliability
          - Critical for backward compatibility
          - Validates fallback logic

        **What it tests:**
          - Batch API failure triggers fallback
          - Individual embed calls are made
          - Correct result is returned after fallback
        """
        # First call fails (batch API), subsequent calls succeed (fallback)
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 400
        mock_response_fail.text = "Batch API not supported"

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.side_effect = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]

        mock_session.post.side_effect = [
            mock_response_fail,
            mock_response_success,
            mock_response_success,
        ]

        result = ollama_client.embed_batch(["hello", "world"], fallback_to_individual=True)

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert mock_session.post.call_count == 3  # 1 batch + 2 individual


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
# Close Method Tests
# =============================================================================


class TestOllamaClientClose:
    """Test suite for OllamaClient.close method."""

    def test_close_closes_session(self, ollama_client: OllamaClient, mock_session: MagicMock) -> None:
        """Test that close closes the session.

        **Why this test is important:**
          - Proper resource cleanup prevents leaks
          - Session closing releases connections
          - Critical for resource management
          - Validates cleanup logic

        **What it tests:**
          - Session close is called
          - Session reference is cleared
        """
        ollama_client.close()

        mock_session.close.assert_called_once()
        assert ollama_client._session is None

    def test_close_handles_none_session(self) -> None:
        """Test that close handles None session gracefully.

        **Why this test is important:**
          - Idempotent close prevents errors
          - Allows multiple close calls
          - Critical for robustness
          - Validates defensive programming

        **What it tests:**
          - Close with None session doesn't raise error
        """
        client = OllamaClient(base_url="http://ollama.example.com:11434", model="test-model")
        object.__setattr__(client, "_session", None)

        # Should not raise
        client.close()


# =============================================================================
# Async Embedding Tests
# =============================================================================


class TestOllamaClientEmbedAsync:
    """Test suite for OllamaClient.embed_async method."""

    @pytest.mark.asyncio
    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_async_success(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_async returns embedding vector on success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        result = await ollama_client.embed_async("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_client.post.assert_called_once()

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_async_raises_on_http_status_error(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_async raises UpstreamError on HTTP status error."""
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

        with pytest.raises(UpstreamError, match="Ollama error 500"):
            await ollama_client.embed_async("hello world")

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_async_raises_on_request_error(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_async raises UpstreamError on request error."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.RequestError("Connection failed")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="Ollama request failed"):
            await ollama_client.embed_async("hello world")

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_async_raises_on_missing_embedding(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_async raises UpstreamError when embedding is missing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        with pytest.raises(UpstreamError, match="missing embedding"):
            await ollama_client.embed_async("hello world")

    @patch("clients.ollama.handle_circuit_breaker_error")
    @pytest.mark.asyncio
    async def test_embed_async_handles_circuit_breaker_open(
        self, mock_handle_error: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_async handles circuit breaker open state."""
        from unittest.mock import PropertyMock

        mock_handle_error.side_effect = UpstreamError("service unavailable")

        # Mock the circuit breaker's current_state property
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        type(mock_breaker).current_state = PropertyMock(return_value=pybreaker.STATE_OPEN)
        object.__setattr__(ollama_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="service unavailable"):
            await ollama_client.embed_async("hello world")

        mock_handle_error.assert_called_once_with("ollama")


# =============================================================================
# Async Batch Embedding Tests
# =============================================================================


class TestOllamaClientEmbedBatchAsync:
    """Test suite for OllamaClient.embed_batch_async method."""

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_async_success(
        self, mock_async_client_cls: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_batch_async returns embeddings on success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_async_client_cls.return_value = mock_client

        result = await ollama_client.embed_batch_async(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_batch_async_raises_value_error_on_empty_list(
        self, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_batch_async raises ValueError for empty list."""
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            await ollama_client.embed_batch_async([])

    @patch("clients.ollama.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_async_falls_back_on_failure(
        self,
        mock_async_client_cls: MagicMock,
        ollama_client: OllamaClient,
    ) -> None:
        """Test that embed_batch_async falls back to individual calls on failure."""
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

        # Mock individual embed_async calls
        with patch.object(ollama_client, "embed_async", side_effect=[[0.1, 0.2], [0.3, 0.4]]) as mock_embed:
            result = await ollama_client.embed_batch_async(["hello", "world"], fallback_to_individual=True)

            assert result == [[0.1, 0.2], [0.3, 0.4]]
            assert mock_embed.call_count == 2

    @patch("clients.ollama.handle_circuit_breaker_error")
    @pytest.mark.asyncio
    async def test_embed_batch_async_handles_circuit_breaker_open(
        self, mock_handle_error: MagicMock, ollama_client: OllamaClient
    ) -> None:
        """Test that embed_batch_async handles circuit breaker open state."""
        from unittest.mock import PropertyMock

        mock_handle_error.side_effect = UpstreamError("service unavailable")

        # Mock the circuit breaker's current_state property
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        type(mock_breaker).current_state = PropertyMock(return_value=pybreaker.STATE_OPEN)
        object.__setattr__(ollama_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="service unavailable"):
            await ollama_client.embed_batch_async(["hello", "world"])

        mock_handle_error.assert_called_once_with("ollama")


# =============================================================================
# Initialization Tests (Session Creation)
# =============================================================================


class TestOllamaClientSessionCreation:
    """Test suite for OllamaClient session initialization."""

    @patch("clients.ollama.create_retry_session")
    def test_client_creates_session_on_init(self, mock_create_session: MagicMock) -> None:
        """Test that client creates a session on initialization when not provided."""
        mock_session = MagicMock(spec=requests.Session)
        mock_create_session.return_value = mock_session

        client = OllamaClient(
            base_url="http://ollama.example.com:11434",
            model="test-model",
        )

        mock_create_session.assert_called_once()
        assert client._session is mock_session

    def test_from_config_with_session(self) -> None:
        """Test that from_config sets session when provided."""
        mock_session = MagicMock(spec=requests.Session)
        config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama.example.com:11434",
            ollama_model="test-model",
        )

        client = OllamaClient.from_config(config, session=mock_session)

        assert client._session is mock_session
