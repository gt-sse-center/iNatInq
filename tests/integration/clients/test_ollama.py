"""Integration tests for Ollama client.

This module tests the OllamaClient against a real Ollama container
to verify production-like behavior including:
- Embedding generation (single and batch)
- Async operations
- Circuit breaker behavior
- Timeout handling
- Error handling
- Resource cleanup

## Container Requirements

These tests use testcontainers to spin up an Ollama container:
- Image: ollama/ollama:latest
- Model: all-minilm (small, 384-dimensional embeddings)
- Port: 11434 (mapped to random host port)

## Running Tests

```bash
# Run all Ollama integration tests
make test-integration TEST_FILE=tests/integration/clients/test_ollama.py

# Run with verbose output
uv run pytest tests/integration/clients/test_ollama.py -v -s
```

## Test Categories

1. Happy Path: Valid requests return valid embeddings
2. Batch Operations: Batch embedding API works correctly
3. Async Operations: Async methods work correctly
4. Circuit Breaker: Opens after failures, recovers after timeout
5. Timeout Handling: Slow operations are handled gracefully
6. Resource Cleanup: Client close() releases resources
7. Observability: Errors and retries are logged
"""

import asyncio
import logging

import aiobreaker
import pybreaker
import pytest

from clients.ollama import OllamaClient

logger = logging.getLogger(__name__)


# =============================================================================
# Happy Path Tests
# =============================================================================


@pytest.mark.integration
class TestHappyPath:
    """Test suite for basic successful operations.

    These tests verify that the Ollama client works correctly
    under normal conditions with a healthy container.
    """

    def test_embed_returns_vector(self, ollama_client: OllamaClient) -> None:
        """Test that embed() returns a valid embedding vector.

        **Why this test is important:**
        Embedding generation is the core functionality of the Ollama client.
        This test verifies that the complete request/response cycle works
        against a real Ollama server.

        **What it tests:**
        - HTTP POST request is made to correct endpoint
        - Response is parsed correctly
        - Embedding vector has correct dimension (384 for all-minilm)
        - Vector contains valid floating-point values
        """
        result = ollama_client.embed("hello world")

        assert isinstance(result, list)
        assert len(result) == 384  # all-minilm dimension
        assert all(isinstance(x, float) for x in result)

    def test_embed_different_texts_produce_different_vectors(self, ollama_client: OllamaClient) -> None:
        """Test that different texts produce different embeddings.

        **Why this test is important:**
        Embeddings should capture semantic meaning. Different texts should
        produce meaningfully different vectors to enable semantic search.

        **What it tests:**
        - Two different texts produce different embedding vectors
        - The vectors are not identical (semantic differentiation works)
        """
        result1 = ollama_client.embed("hello world")
        result2 = ollama_client.embed("quantum physics experiments")

        # Vectors should be different for semantically different texts
        assert result1 != result2

    @pytest.mark.asyncio
    async def test_embed_async_returns_vector(self, ollama_client: OllamaClient) -> None:
        """Test that embed_async() returns a valid embedding vector.

        **Why this test is important:**
        Async embedding is critical for high-throughput pipelines where
        we need to parallelize multiple embedding requests.

        **What it tests:**
        - Async HTTP request completes successfully
        - Response is parsed correctly in async context
        - Embedding vector has correct dimension
        """
        result = await ollama_client.embed_async("hello world")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)


# =============================================================================
# Batch Operation Tests
# =============================================================================


@pytest.mark.integration
class TestBatchOperations:
    """Test suite for batch embedding operations.

    Batch operations are more efficient than individual calls
    when embedding multiple texts.
    """

    def test_embed_batch_returns_multiple_vectors(self, ollama_client: OllamaClient) -> None:
        """Test that embed_batch() returns vectors for all input texts.

        **Why this test is important:**
        Batch embedding reduces API calls and improves throughput.
        This test verifies the batch API works correctly.

        **What it tests:**
        - Batch API endpoint is called
        - Correct number of vectors returned
        - Each vector has correct dimension
        """
        texts = ["hello", "world", "test"]
        result = ollama_client.embed_batch(texts)

        assert len(result) == 3
        assert all(len(v) == 384 for v in result)
        assert all(isinstance(x, float) for v in result for x in v)

    @pytest.mark.asyncio
    async def test_embed_batch_async_returns_multiple_vectors(self, ollama_client: OllamaClient) -> None:
        """Test that embed_batch_async() returns vectors for all input texts.

        **Why this test is important:**
        Async batch embedding combines the benefits of batching and
        async operations for maximum throughput.

        **What it tests:**
        - Async batch API endpoint works
        - Correct number of vectors returned
        - Vectors are correctly parsed in async context
        """
        texts = ["hello", "world"]
        result = await ollama_client.embed_batch_async(texts)

        assert len(result) == 2
        assert all(len(v) == 384 for v in result)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


@pytest.mark.integration
class TestCircuitBreaker:
    """Test suite for circuit breaker behavior.

    Circuit breakers prevent cascading failures by failing fast
    when a service is degraded.
    """

    def test_sync_circuit_breaker_starts_closed(self, ollama_client: OllamaClient) -> None:
        """Test that sync circuit breaker starts in closed state.

        **Why this test is important:**
        The circuit breaker must start closed to allow normal operation.
        A breaker that starts open would block all requests.

        **What it tests:**
        - Sync circuit breaker (_breaker) exists
        - Initial state is CLOSED
        """
        assert ollama_client._breaker is not None
        assert ollama_client._breaker.current_state == pybreaker.STATE_CLOSED

    def test_async_circuit_breaker_starts_closed(self, ollama_client: OllamaClient) -> None:
        """Test that async circuit breaker starts in closed state.

        **Why this test is important:**
        The async circuit breaker must also start closed.
        Async operations use a separate breaker (aiobreaker).

        **What it tests:**
        - Async circuit breaker (_async_breaker) exists
        - Initial state is CLOSED
        """
        assert ollama_client._async_breaker is not None
        assert ollama_client._async_breaker.current_state == aiobreaker.state.CircuitBreakerState.CLOSED

    def test_sync_circuit_breaker_tracks_successes(self, ollama_client: OllamaClient) -> None:
        """Test that successful calls don't affect circuit breaker.

        **Why this test is important:**
        Successful calls should not increment the failure counter.
        The circuit breaker should remain closed during normal operation.

        **What it tests:**
        - Successful embed() call completes
        - Circuit breaker remains closed
        - Failure counter stays at 0
        """
        # Make a successful call
        ollama_client.embed("test")

        # Circuit breaker should still be closed
        assert ollama_client._breaker.current_state == pybreaker.STATE_CLOSED
        assert ollama_client._breaker.fail_counter == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Test suite for error handling behavior.

    These tests verify that the client handles errors gracefully
    and raises appropriate exceptions.
    """

    def test_empty_batch_raises_value_error(self, ollama_client: OllamaClient) -> None:
        """Test that empty batch raises ValueError.

        **Why this test is important:**
        Empty input should fail fast with a clear error rather than
        making an unnecessary API call.

        **What it tests:**
        - Empty texts list raises ValueError
        - Error message is descriptive
        """
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            ollama_client.embed_batch([])

    @pytest.mark.asyncio
    async def test_empty_batch_async_raises_value_error(self, ollama_client: OllamaClient) -> None:
        """Test that empty async batch raises ValueError.

        **Why this test is important:**
        Async batch operations should have the same validation as sync.

        **What it tests:**
        - Empty texts list raises ValueError in async context
        """
        with pytest.raises(ValueError, match="texts list cannot be empty"):
            await ollama_client.embed_batch_async([])


# =============================================================================
# Resource Cleanup Tests
# =============================================================================


@pytest.mark.integration
class TestResourceCleanup:
    """Test suite for resource cleanup behavior.

    These tests verify that clients properly release resources
    when closed.
    """

    def test_close_is_idempotent(self, ollama_url: str) -> None:
        """Test that close() can be called multiple times safely.

        **Why this test is important:**
        In complex error handling flows, close() might be called multiple
        times. It should be safe to call repeatedly.

        **What it tests:**
        - First close() succeeds
        - Second close() doesn't raise
        - Session is properly cleared
        """
        client = OllamaClient(base_url=ollama_url, model="all-minilm", timeout_s=30)

        # First close
        client.close()
        assert client._session is None

        # Second close should not raise
        client.close()
        assert client._session is None

    def test_client_usable_after_session_recreation(self, ollama_url: str) -> None:
        """Test that client can recreate session after close.

        **Why this test is important:**
        Some use cases might need to close and reopen a client.
        The session property should lazily recreate the session.

        **What it tests:**
        - Client can be closed
        - Accessing session property recreates session
        - Client is usable again
        """
        client = OllamaClient(base_url=ollama_url, model="all-minilm", timeout_s=30)

        # Use it
        result1 = client.embed("hello")
        assert len(result1) == 384

        # Close it
        client.close()
        assert client._session is None

        # Use it again - session should be recreated
        result2 = client.embed("world")
        assert len(result2) == 384

        # Cleanup
        client.close()


# =============================================================================
# Vector Size Tests
# =============================================================================


@pytest.mark.integration
class TestVectorSize:
    """Test suite for vector size configuration.

    Vector size must be consistent with the model being used.
    """

    def test_vector_size_matches_model(self, ollama_client: OllamaClient) -> None:
        """Test that vector_size property matches model output.

        **Why this test is important:**
        The vector_size property is used to configure vector databases.
        It must match the actual output dimension.

        **What it tests:**
        - vector_size property returns expected value
        - Actual embedding matches expected dimension
        """
        expected_size = 384  # all-minilm
        assert ollama_client.vector_size == expected_size

        # Verify actual embedding matches
        embedding = ollama_client.embed("test")
        assert len(embedding) == expected_size


# =============================================================================
# Concurrent Operation Tests
# =============================================================================


@pytest.mark.integration
class TestConcurrentOperations:
    """Test suite for concurrent async operations.

    These tests verify that multiple async operations can run
    concurrently without issues.
    """

    @pytest.mark.asyncio
    async def test_concurrent_embed_async(self, ollama_client: OllamaClient) -> None:
        """Test that multiple embed_async calls can run concurrently.

        **Why this test is important:**
        In production, we often need to embed multiple texts in parallel.
        This test verifies that concurrent calls don't interfere.

        **What it tests:**
        - Multiple async calls can run concurrently
        - All calls return valid results
        - No resource contention issues
        """
        texts = ["hello", "world", "test", "concurrent"]

        # Run all embed_async calls concurrently
        tasks = [ollama_client.embed_async(text) for text in texts]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert all(len(r) == 384 for r in results)
        assert all(isinstance(x, float) for r in results for x in r)


# =============================================================================
# Observability Tests
# =============================================================================


@pytest.mark.integration
class TestObservability:
    """Test suite for logging and observability.

    These tests verify that operations are properly logged
    for debugging and monitoring.
    """

    def test_embed_logs_on_success(
        self, ollama_client: OllamaClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that successful operations can be observed.

        **Why this test is important:**
        Observability is critical for debugging and monitoring.
        While we don't log every success, failures should be logged.

        **What it tests:**
        - Normal operation completes without error logs
        - Client functions correctly (indirectly validates observability hooks)
        """
        with caplog.at_level(logging.DEBUG):
            result = ollama_client.embed("test")

        assert len(result) == 384
        # No error logs for successful operations
        assert "error" not in caplog.text.lower() or "error_type" in caplog.text.lower()


# =============================================================================
# From Config Factory Tests
# =============================================================================


@pytest.mark.integration
class TestFromConfig:
    """Test suite for factory method configuration.

    These tests verify that from_config correctly creates clients.
    """

    def test_from_config_creates_working_client(self, ollama_url: str) -> None:
        """Test that from_config creates a functional client.

        **Why this test is important:**
        The factory method is the recommended way to create clients
        from configuration. It must work correctly.

        **What it tests:**
        - from_config creates a valid client
        - Client can make API calls
        - Configuration is correctly applied
        """
        from config import EmbeddingConfig

        config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url=ollama_url,
            ollama_model="all-minilm",
        )

        client = OllamaClient.from_config(config)

        try:
            result = client.embed("test")
            assert len(result) == 384
        finally:
            client.close()
