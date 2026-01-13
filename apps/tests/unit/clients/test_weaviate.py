"""Unit tests for clients.weaviate module.

This file tests the WeaviateClientWrapper class which provides vector database operations
via the Weaviate client.

# Test Coverage

The tests cover:
  - Client Initialization: Default and custom configuration, from_config factory
  - Collection Management: ensure_collection, collection existence checking
  - Vector Operations: search, batch_upsert
  - Circuit Breaker Integration: Circuit breaker usage, error handling
  - Error Handling: UpstreamError on failures, circuit breaker errors

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The underlying WeaviateAsyncClient and circuit breaker are mocked to isolate client logic.

# Running Tests

Run with: pytest tests/unit/clients/test_weaviate.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pybreaker
import pytest

from clients.weaviate import WeaviateClientWrapper, WeaviateDataObject
from config import VectorDBConfig
from core.exceptions import UpstreamError
from core.models import SearchResults

# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestWeaviateClientWrapperInit:
    """Test suite for WeaviateClientWrapper initialization."""

    @patch("clients.weaviate.WeaviateAsyncClient")
    def test_creates_client_with_config(self, mock_weaviate_async_client: MagicMock) -> None:
        """Test that client is created with configuration.

        **Why this test is important:**
          - Client initialization is the foundation for all operations
          - Ensures configuration is applied correctly
          - Validates that Weaviate client is created with correct parameters
          - Critical for basic functionality

        **What it tests:**
          - WeaviateAsyncClient is created with correct connection params
          - Client attributes are set correctly
          - Circuit breaker is created
        """
        mock_client = AsyncMock()
        mock_weaviate_async_client.return_value = mock_client

        client = WeaviateClientWrapper(url="http://weaviate.example.com:8080")

        mock_weaviate_async_client.assert_called_once()
        assert client.url == "http://weaviate.example.com:8080"
        assert client.api_key is None
        assert client._client == mock_client

    @patch("clients.weaviate.WeaviateAsyncClient")
    def test_creates_client_with_api_key(self, mock_weaviate_async_client: MagicMock) -> None:
        """Test that client is created with API key.

        **Why this test is important:**
          - API key authentication is supported
          - Validates authentication configuration
          - Critical for secure deployments
          - Validates parameter handling

        **What it tests:**
          - API key is stored correctly
          - Client is created with auth configuration
        """
        mock_client = AsyncMock()
        mock_weaviate_async_client.return_value = mock_client

        client = WeaviateClientWrapper(url="http://weaviate.example.com:8080", api_key="test-key")

        assert client.api_key == "test-key"
        mock_weaviate_async_client.assert_called_once()

    @patch("clients.weaviate.WeaviateAsyncClient")
    def test_creates_circuit_breaker(self, mock_weaviate_async_client: MagicMock) -> None:
        """Test that circuit breaker is created during initialization.

        **Why this test is important:**
          - Circuit breaker provides fault tolerance
          - Ensures circuit breaker is configured with correct parameters
          - Critical for production reliability
          - Validates circuit breaker integration

        **What it tests:**
          - Circuit breaker is created with correct name
          - Failure threshold and recovery timeout are set correctly
        """
        mock_weaviate_async_client.return_value = AsyncMock()

        client = WeaviateClientWrapper(url="http://weaviate.example.com:8080")

        # Verify circuit breaker was created
        assert client._breaker is not None
        assert isinstance(client._breaker, pybreaker.CircuitBreaker)
        assert client._breaker.name == "weaviate"
        assert client._breaker.fail_max == 3
        assert client._breaker.reset_timeout == 60

    def test_from_config_creates_client(self) -> None:
        """Test that from_config factory creates client correctly.

        **Why this test is important:**
          - Factory method provides convenient client creation
          - Validates configuration integration
          - Critical for configuration-driven initialization
          - Ensures proper config validation

        **What it tests:**
          - Client is created from VectorDBConfig
          - Config values are correctly applied
          - API key is passed if provided
        """
        config = VectorDBConfig(
            provider_type="weaviate",
            collection="test-collection",
            weaviate_url="http://weaviate.example.com:8080",
            weaviate_api_key="test-key",
        )

        with patch("clients.weaviate.WeaviateAsyncClient"):
            client = WeaviateClientWrapper.from_config(config)

        assert client.url == "http://weaviate.example.com:8080"
        assert client.api_key == "test-key"

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
        config = VectorDBConfig(
            provider_type="qdrant",
            collection="test-collection",
            qdrant_url="http://qdrant.example.com",
        )

        with pytest.raises(ValueError, match="provider_type must be 'weaviate'"):
            WeaviateClientWrapper.from_config(config)

    def test_from_config_validates_required_fields(self) -> None:
        """Test that from_config validates required fields.

        **Why this test is important:**
          - Prevents configuration errors
          - Ensures required fields are present
          - Critical for preventing runtime errors
          - Validates error handling

        **What it tests:**
          - ValueError is raised for missing weaviate_url
        """
        config = VectorDBConfig(provider_type="weaviate", collection="test-collection", weaviate_url=None)

        with pytest.raises(ValueError, match="requires: weaviate_url"):
            WeaviateClientWrapper.from_config(config)


# =============================================================================
# Collection Management Tests
# =============================================================================


class TestWeaviateClientWrapperEnsureCollection:
    """Test suite for WeaviateClientWrapper.ensure_collection method."""

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_if_missing(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that ensure_collection creates collection if missing.

        **Why this test is important:**
          - Collection creation is essential for storage operations
          - Ensures collections exist before use
          - Critical for dev convenience functions
          - Validates collection creation logic

        **What it tests:**
          - exists is called to check existence
          - create is called if collection doesn't exist
          - Collection is created with correct configuration
        """
        mock_weaviate_client.collections.exists.return_value = False
        mock_weaviate_client.collections.create.return_value = None
        mock_weaviate_client.__aenter__ = AsyncMock(return_value=mock_weaviate_client)
        mock_weaviate_client.__aexit__ = AsyncMock(return_value=None)

        await weaviate_client.ensure_collection_async(collection="test-collection", vector_size=768)

        mock_weaviate_client.collections.exists.assert_called_once_with("test-collection")
        mock_weaviate_client.collections.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_if_exists(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that ensure_collection skips creation if collection exists.

        **Why this test is important:**
          - Idempotent operations prevent errors
          - Avoids unnecessary API calls
          - Critical for efficiency
          - Validates existence checking

        **What it tests:**
          - exists is called to check existence
          - create is not called if collection exists
        """
        mock_weaviate_client.collections.exists.return_value = True
        mock_weaviate_client.__aenter__ = AsyncMock(return_value=mock_weaviate_client)
        mock_weaviate_client.__aexit__ = AsyncMock(return_value=None)

        await weaviate_client.ensure_collection_async(collection="test-collection", vector_size=768)

        mock_weaviate_client.collections.exists.assert_called_once_with("test-collection")
        mock_weaviate_client.collections.create.assert_not_called()


# =============================================================================
# Search Tests
# =============================================================================


class TestWeaviateClientWrapperSearch:
    """Test suite for WeaviateClientWrapper.search method."""

    @pytest.mark.asyncio
    async def test_search_success(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that search returns results on success.

        **Why this test is important:**
          - Vector search is the core functionality
          - Validates successful API interaction
          - Ensures response parsing is correct
          - Critical for basic functionality

        **What it tests:**
          - search is called with correct parameters
          - Results are converted to SearchResults
          - Items are correctly formatted with scores and payloads
        """
        mock_object1 = MagicMock()
        mock_object1.uuid = "uuid-1"
        mock_object1.properties = {"text": "hello"}
        mock_metadata1 = MagicMock()
        mock_metadata1.certainty = 0.95
        mock_metadata1.distance = None
        mock_object1.metadata = mock_metadata1

        mock_object2 = MagicMock()
        mock_object2.uuid = "uuid-2"
        mock_object2.properties = {"text": "world"}
        mock_metadata2 = MagicMock()
        mock_metadata2.certainty = None
        mock_metadata2.distance = 0.2
        mock_object2.metadata = mock_metadata2

        mock_response = MagicMock()
        mock_response.objects = [mock_object1, mock_object2]

        mock_collection = MagicMock()
        mock_query = MagicMock()
        mock_query.near_vector = AsyncMock(return_value=mock_response)
        mock_collection.query = mock_query
        mock_weaviate_client.collections.get.return_value = mock_collection
        mock_weaviate_client.__aenter__ = AsyncMock(return_value=mock_weaviate_client)
        mock_weaviate_client.__aexit__ = AsyncMock(return_value=None)

        result = await weaviate_client.search_async(
            collection="test-collection", query_vector=[0.1, 0.2, 0.3], limit=10
        )

        assert isinstance(result, SearchResults)
        assert len(result.items) == 2
        assert result.items[0].point_id == "uuid-1"
        assert result.items[0].score == 0.95  # Uses certainty
        assert result.items[0].payload == {"text": "hello"}
        assert result.items[1].point_id == "uuid-2"
        assert result.items[1].score == pytest.approx(0.8)  # 1.0 - 0.2 distance
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_search_raises_upstream_error_on_failure(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that search raises UpstreamError on failure.

        **Why this test is important:**
          - Error handling ensures consistent error types
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation and debugging
          - Validates error wrapping

        **What it tests:**
          - Exception is wrapped in UpstreamError
          - Error message includes context
        """
        mock_weaviate_client.__aenter__ = AsyncMock(return_value=mock_weaviate_client)
        mock_weaviate_client.__aexit__ = AsyncMock(return_value=None)
        mock_weaviate_client.collections.get.side_effect = Exception("Search failed")

        with pytest.raises(UpstreamError, match="Weaviate search failed"):
            await weaviate_client.search_async(
                collection="test-collection", query_vector=[0.1, 0.2], limit=10
            )

    @pytest.mark.asyncio
    async def test_search_handles_circuit_breaker_open(
        self,
        weaviate_client: WeaviateClientWrapper,
    ) -> None:
        """Test that search handles circuit breaker open state.

        **Why this test is important:**
          - Circuit breaker errors need special handling
          - UpstreamError conversion ensures consistent error types
          - Critical for fault tolerance
          - Validates circuit breaker integration

        **What it tests:**
          - Open circuit breaker state is checked
          - handle_circuit_breaker_error is called
        """
        # Replace the circuit breaker with a mock in OPEN state
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        mock_breaker.current_state = pybreaker.STATE_OPEN
        object.__setattr__(weaviate_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="weaviate service is currently unavailable"):
            await weaviate_client.search_async(
                collection="test-collection", query_vector=[0.1, 0.2], limit=10
            )


# =============================================================================
# Batch Upsert Tests
# =============================================================================


class TestWeaviateClientWrapperBatchUpsert:
    """Test suite for WeaviateClientWrapper.batch_upsert method."""

    @pytest.mark.asyncio
    async def test_batch_upsert_success(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that batch_upsert succeeds on valid input.

        **Why this test is important:**
          - Batch upsert is essential for bulk operations
          - Validates successful API interaction
          - Ensures collection is created if needed
          - Critical for performance optimization

        **What it tests:**
          - ensure_collection is called first
          - insert_many is called with correct parameters
          - Empty points list is handled
        """
        mock_weaviate_client.collections.exists.return_value = True
        mock_weaviate_client.__aenter__ = AsyncMock(return_value=mock_weaviate_client)
        mock_weaviate_client.__aexit__ = AsyncMock(return_value=None)

        mock_collection = MagicMock()
        mock_data = MagicMock()
        mock_data.insert_many = AsyncMock(return_value=None)
        mock_collection.data = mock_data
        mock_weaviate_client.collections.get.return_value = mock_collection

        points = [
            WeaviateDataObject(uuid="uuid-1", properties={"text": "hello"}, vector=[0.1, 0.2, 0.3]),
            WeaviateDataObject(uuid="uuid-2", properties={"text": "world"}, vector=[0.4, 0.5, 0.6]),
        ]

        await weaviate_client.batch_upsert_async(collection="test-collection", points=points, vector_size=768)

        mock_data.insert_many.assert_called_once()
        call_args = mock_data.insert_many.call_args[0]
        assert len(call_args[0]) == 2  # Two objects

    @pytest.mark.asyncio
    async def test_batch_upsert_skips_empty_list(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that batch_upsert skips empty points list.

        **Why this test is important:**
          - Empty lists should be handled gracefully
          - Prevents unnecessary API calls
          - Critical for efficiency
          - Validates edge case handling

        **What it tests:**
          - Empty points list returns without API calls
        """
        await weaviate_client.batch_upsert_async(collection="test-collection", points=[], vector_size=768)

        mock_weaviate_client.collections.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_upsert_raises_upstream_error_on_failure(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that batch_upsert raises UpstreamError on failure.

        **Why this test is important:**
          - Error handling ensures consistent error types
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation and debugging
          - Validates error wrapping

        **What it tests:**
          - Exception is wrapped in UpstreamError
          - Error message includes context
        """
        mock_weaviate_client.collections.exists.return_value = True
        mock_weaviate_client.__aenter__ = AsyncMock(return_value=mock_weaviate_client)
        mock_weaviate_client.__aexit__ = AsyncMock(return_value=None)
        mock_weaviate_client.collections.get.side_effect = Exception("Upsert failed")

        points = [WeaviateDataObject(uuid="uuid-1", properties={"text": "hello"}, vector=[0.1, 0.2])]

        with pytest.raises(UpstreamError, match="Weaviate batch upsert failed"):
            await weaviate_client.batch_upsert_async(
                collection="test-collection", points=points, vector_size=768
            )

    @pytest.mark.asyncio
    async def test_batch_upsert_handles_circuit_breaker_open(
        self,
        weaviate_client: WeaviateClientWrapper,
    ) -> None:
        """Test that batch_upsert handles circuit breaker open state.

        **Why this test is important:**
          - Circuit breaker errors need special handling
          - UpstreamError conversion ensures consistent error types
          - Critical for fault tolerance
          - Validates circuit breaker integration

        **What it tests:**
          - Open circuit breaker state is checked
          - handle_circuit_breaker_error is called
        """
        # Replace the circuit breaker with a mock in OPEN state
        mock_breaker = MagicMock(spec=pybreaker.CircuitBreaker)
        mock_breaker.current_state = pybreaker.STATE_OPEN
        object.__setattr__(weaviate_client, "_breaker", mock_breaker)

        points = [WeaviateDataObject(uuid="uuid-1", properties={"text": "hello"}, vector=[0.1, 0.2])]

        with pytest.raises(UpstreamError, match="weaviate service is currently unavailable"):
            await weaviate_client.batch_upsert_async(
                collection="test-collection", points=points, vector_size=768
            )


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestWeaviateClientWrapperAdditional:
    """Test suite for additional WeaviateClientWrapper coverage."""

    def test_client_property(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that client property returns the async client."""
        assert weaviate_client.client is mock_weaviate_client

    @pytest.mark.asyncio
    async def test_ensure_collection_handles_already_exists_error(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that ensure_collection handles 'already exists' error gracefully."""
        mock_weaviate_client.collections.exists.return_value = False
        mock_weaviate_client.collections.create.side_effect = Exception("Collection already exists")

        # Should not raise
        await weaviate_client.ensure_collection_async(collection="TestCollection", vector_size=768)

    @pytest.mark.asyncio
    async def test_ensure_collection_handles_duplicate_error(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that ensure_collection handles 'duplicate' error gracefully."""
        mock_weaviate_client.collections.exists.return_value = False
        mock_weaviate_client.collections.create.side_effect = Exception("Duplicate class name")

        # Should not raise
        await weaviate_client.ensure_collection_async(collection="TestCollection", vector_size=768)

    @pytest.mark.asyncio
    async def test_ensure_collection_raises_on_other_errors(
        self, weaviate_client: WeaviateClientWrapper, mock_weaviate_client: AsyncMock
    ) -> None:
        """Test that ensure_collection raises UpstreamError on other exceptions."""
        mock_weaviate_client.collections.exists.return_value = False
        mock_weaviate_client.collections.create.side_effect = Exception("API Error")

        with pytest.raises(UpstreamError, match="Weaviate collection creation failed"):
            await weaviate_client.ensure_collection_async(collection="TestCollection", vector_size=768)

    def test_close_with_client(self, weaviate_client: WeaviateClientWrapper) -> None:
        """Test that close clears the client reference."""
        weaviate_client.close()

        assert weaviate_client._client is None

    def test_close_with_none_client(self) -> None:
        """Test that close handles None client gracefully."""
        with patch("clients.weaviate.WeaviateAsyncClient") as mock_client_cls:
            mock_client_cls.return_value = None
            client = WeaviateClientWrapper(url="http://weaviate.example.com:8080")
            object.__setattr__(client, "_client", None)

            # Should not raise
            client.close()
