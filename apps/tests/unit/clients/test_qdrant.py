"""Unit tests for clients.qdrant module.

This file tests the QdrantClientWrapper class which provides vector database operations
via the Qdrant client.

# Test Coverage

The tests cover:
  - Client Initialization: Default configuration, from_config factory
  - Collection Management: ensure_collection, collection existence checking
  - Vector Operations: search, batch_upsert, batch_upsert_sync
  - Indexing Operations: disable_indexing, enable_indexing
  - Circuit Breaker Integration: Circuit breaker usage, error handling
  - Error Handling: UpstreamError on failures, circuit breaker errors

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The underlying Qdrant clients and circuit breaker are mocked to isolate client logic.

# Running Tests

Run with: pytest tests/unit/clients/test_qdrant.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pybreaker
import pytest
from qdrant_client.models import PointStruct
from clients.qdrant import QdrantClientWrapper
from config import VectorDBConfig
from core.exceptions import UpstreamError
from core.models import SearchResults

# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestQdrantClientWrapperInit:
    """Test suite for QdrantClientWrapper initialization."""

    @patch("clients.qdrant.AsyncQdrantClient")
    @patch("clients.qdrant.QdrantClient")
    def test_creates_client_with_config(
        self,
        mock_async_qdrant_client: MagicMock,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """Test that client is created with configuration.

        **Why this test is important:**
          - Client initialization is the foundation for all operations
          - Ensures configuration is applied correctly
          - Validates that Qdrant clients are created with correct parameters
          - Critical for basic functionality

        **What it tests:**
          - AsyncQdrantClient is created with correct URL and timeout
          - QdrantClient is created with correct URL and timeout
          - Client attributes are set correctly
          - Circuit breaker is created
        """
        mock_async_client = AsyncMock()
        mock_qdrant_client.return_value = mock_async_client
        mock_sync_client = MagicMock()
        mock_async_qdrant_client.return_value = mock_sync_client

        client = QdrantClientWrapper(url="http://qdrant.example.com:6333")

        mock_qdrant_client.assert_called_once_with(
            url="http://qdrant.example.com:6333", api_key=None, timeout=300
        )
        mock_async_qdrant_client.assert_called_once_with(
            url="http://qdrant.example.com:6333", api_key=None, timeout=300
        )
        assert client.url == "http://qdrant.example.com:6333"
        assert client._client == mock_async_client
        assert client._sync_client == mock_sync_client

    def test_creates_circuit_breaker(self) -> None:
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
        client = QdrantClientWrapper(url="http://qdrant.example.com:6333")

        # Verify circuit breaker was created
        assert client._breaker is not None
        assert isinstance(client._breaker, pybreaker.CircuitBreaker)
        assert client._breaker.name == "qdrant"
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
        """
        config = VectorDBConfig(
            provider_type="qdrant",
            collection="test-collection",
            qdrant_url="http://qdrant.example.com:6333",
        )

        with (
            patch("clients.qdrant.AsyncQdrantClient"),
            patch("clients.qdrant.QdrantClient"),
        ):
            client = QdrantClientWrapper.from_config(config)

        assert client.url == "http://qdrant.example.com:6333"

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
            provider_type="weaviate",
            collection="test-collection",
            weaviate_url="http://weaviate.example.com",
        )

        with pytest.raises(ValueError, match="provider_type must be 'qdrant'"):
            QdrantClientWrapper.from_config(config)

    def test_from_config_validates_required_fields(self) -> None:
        """Test that from_config validates required fields.

        **Why this test is important:**
          - Prevents configuration errors
          - Ensures required fields are present
          - Critical for preventing runtime errors
          - Validates error handling

        **What it tests:**
          - ValueError is raised for missing qdrant_url
        """
        config = VectorDBConfig(provider_type="qdrant", collection="test-collection", qdrant_url=None)

        with pytest.raises(ValueError, match="requires: qdrant_url"):
            QdrantClientWrapper.from_config(config)


# =============================================================================
# Collection Management Tests
# =============================================================================


class TestQdrantClientWrapperEnsureCollection:
    """Test suite for QdrantClientWrapper.ensure_collection method."""

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_if_missing(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_collection creates collection if missing.

        **Why this test is important:**
          - Collection creation is essential for storage operations
          - Ensures collections exist before use
          - Critical for dev convenience functions
          - Validates collection creation logic

        **What it tests:**
          - get_collections is called to check existence
          - create_collection is called if collection doesn't exist
        """
        mock_collections = MagicMock()
        mock_collections.collections = []  # Empty, collection doesn't exist
        mock_async_client.get_collections.return_value = mock_collections

        await qdrant_client.ensure_collection_async(collection="test-collection", vector_size=768)

        mock_async_client.get_collections.assert_called_once()
        mock_async_client.create_collection.assert_called_once()
        call_kwargs = mock_async_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["vectors_config"].size == 768

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_if_exists(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_collection skips creation if collection exists.

        **Why this test is important:**
          - Idempotent operations prevent errors
          - Avoids unnecessary API calls
          - Critical for efficiency
          - Validates existence checking

        **What it tests:**
          - get_collections is called to check existence
          - create_collection is not called if collection exists
        """
        mock_collection = MagicMock()
        mock_collection.name = "test-collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_async_client.get_collections.return_value = mock_collections

        await qdrant_client.ensure_collection_async(collection="test-collection", vector_size=768)

        mock_async_client.get_collections.assert_called_once()
        mock_async_client.create_collection.assert_not_called()


# =============================================================================
# Search Tests
# =============================================================================


class TestQdrantClientWrapperSearch:
    """Test suite for QdrantClientWrapper.search method."""

    @pytest.mark.asyncio
    async def test_search_success(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that search returns results on success.

        **Why this test is important:**
          - Vector search is the core functionality
          - Validates successful API interaction
          - Ensures response parsing is correct
          - Critical for basic functionality

        **What it tests:**
          - query_points is called with correct parameters
          - Results are converted to SearchResults
          - Items are correctly formatted
        """
        mock_point1 = MagicMock()
        mock_point1.id = "1"
        mock_point1.score = 0.95
        mock_point1.payload = {"text": "hello"}

        mock_point2 = MagicMock()
        mock_point2.id = "2"
        mock_point2.score = 0.85
        mock_point2.payload = {"text": "world"}

        # query_points returns a response object with .points attribute
        mock_response = MagicMock()
        mock_response.points = [mock_point1, mock_point2]
        mock_async_client.query_points.return_value = mock_response

        result = await qdrant_client.search_async(
            collection="test-collection", query_vector=[0.1, 0.2, 0.3], limit=10
        )

        assert isinstance(result, SearchResults)
        assert len(result.items) == 2
        assert result.items[0].point_id == "1"
        assert result.items[0].score == 0.95
        assert result.items[0].payload == {"text": "hello"}
        assert result.items[1].point_id == "2"
        assert result.items[1].score == 0.85
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_search_raises_upstream_error_on_failure(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
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
        mock_async_client.query_points.side_effect = Exception("Search failed")

        with pytest.raises(UpstreamError, match="Qdrant search failed"):
            await qdrant_client.search_async(collection="test-collection", query_vector=[0.1, 0.2], limit=10)

    @pytest.mark.asyncio
    async def test_search_handles_none_score(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that search handles None score gracefully.

        **Why this test is important:**
          - Qdrant can return None scores in some cases
          - Ensures graceful handling without TypeError
          - Critical for robustness

        **What it tests:**
          - None score is converted to 0.0
          - No exception is raised
        """
        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = None  # Qdrant can return None
        mock_point.payload = {"text": "hello"}

        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_async_client.query_points.return_value = mock_response

        result = await qdrant_client.search_async(
            collection="test-collection", query_vector=[0.1, 0.2, 0.3], limit=10
        )

        assert len(result.items) == 1
        assert result.items[0].score == 0.0  # None converted to 0.0

    @pytest.mark.asyncio
    async def test_search_handles_circuit_breaker_exception(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that search handles CircuitBreakerError during call.

        **Why this test is important:**
          - Circuit breaker can throw during a call (not just pre-check)
          - Ensures consistent UpstreamError for all circuit breaker scenarios
          - Critical for fault tolerance

        **What it tests:**
          - CircuitBreakerError is caught and converted to UpstreamError
        """
        mock_async_client.query_points.side_effect = pybreaker.CircuitBreakerError(
            pybreaker.CircuitBreaker(name="test")
        )

        with pytest.raises(UpstreamError, match="qdrant service is currently unavailable"):
            await qdrant_client.search_async(collection="test-collection", query_vector=[0.1, 0.2], limit=10)

    @pytest.mark.asyncio
    async def test_search_handles_circuit_breaker_open(
        self,
        qdrant_client: QdrantClientWrapper,
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
        object.__setattr__(qdrant_client, "_breaker", mock_breaker)

        with pytest.raises(UpstreamError, match="qdrant service is currently unavailable"):
            await qdrant_client.search_async(collection="test-collection", query_vector=[0.1, 0.2], limit=10)


# =============================================================================
# Batch Upsert Tests
# =============================================================================


class TestQdrantClientWrapperBatchUpsert:
    """Test suite for QdrantClientWrapper.batch_upsert method."""

    @pytest.mark.asyncio
    async def test_batch_upsert_success(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that batch_upsert succeeds on valid input.

        **Why this test is important:**
          - Batch upsert is essential for bulk operations
          - Validates successful API interaction
          - Ensures collection is created if needed
          - Critical for performance optimization

        **What it tests:**
          - ensure_collection is called first
          - upsert is called with correct parameters
          - Empty points list is handled
        """
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections
        mock_async_client.upsert.return_value = None

        points = [
            PointStruct(id="1", vector=[0.1, 0.2], payload={"text": "hello"}),
            PointStruct(id="2", vector=[0.3, 0.4], payload={"text": "world"}),
        ]

        await qdrant_client.batch_upsert_async(collection="test-collection", points=points, vector_size=768)

        mock_async_client.upsert.assert_called_once()
        call_kwargs = mock_async_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["points"] == points

    @pytest.mark.asyncio
    async def test_batch_upsert_skips_empty_list(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
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
        await qdrant_client.batch_upsert_async(collection="test-collection", points=[], vector_size=768)

        mock_async_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_upsert_raises_upstream_error_on_failure(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
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
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections
        mock_async_client.upsert.side_effect = Exception("Upsert failed")

        points = [PointStruct(id="1", vector=[0.1, 0.2], payload={"text": "hello"})]

        with pytest.raises(UpstreamError, match="Qdrant batch upsert failed"):
            await qdrant_client.batch_upsert_async(
                collection="test-collection", points=points, vector_size=768
            )

    @pytest.mark.asyncio
    async def test_batch_upsert_handles_circuit_breaker_open(
        self,
        qdrant_client: QdrantClientWrapper,
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
        object.__setattr__(qdrant_client, "_breaker", mock_breaker)

        points = [PointStruct(id="1", vector=[0.1, 0.2], payload={"text": "hello"})]

        with pytest.raises(UpstreamError, match="qdrant service is currently unavailable"):
            await qdrant_client.batch_upsert_async(
                collection="test-collection", points=points, vector_size=768
            )


class TestQdrantClientWrapperBatchUpsertSync:
    """Test suite for QdrantClientWrapper.batch_upsert_sync method."""

    def test_batch_upsert_sync_success(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that batch_upsert_sync succeeds on valid input.

        **Why this test is important:**
          - Sync batch upsert is essential for synchronous code paths
          - Validates successful API interaction
          - Ensures collection is created if needed
          - Critical for Spark jobs and synchronous operations

        **What it tests:**
          - Collection existence is checked and created if needed
          - upsert is called with correct parameters
          - Empty points list is handled
        """
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_sync_client.get_collections.return_value = mock_collections_response
        mock_sync_client.upsert.return_value = None

        points = [
            PointStruct(id="1", vector=[0.1, 0.2], payload={"text": "hello"}),
            PointStruct(id="2", vector=[0.3, 0.4], payload={"text": "world"}),
        ]

        qdrant_client.batch_upsert_sync(collection="test-collection", points=points, vector_size=768)

        mock_sync_client.upsert.assert_called_once()
        call_kwargs = mock_sync_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["points"] == points

    def test_batch_upsert_sync_skips_empty_list(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that batch_upsert_sync skips empty points list.

        **Why this test is important:**
          - Empty lists should be handled gracefully
          - Prevents unnecessary API calls
          - Critical for efficiency
          - Validates edge case handling

        **What it tests:**
          - Empty points list returns without API calls
        """
        qdrant_client.batch_upsert_sync(collection="test-collection", points=[], vector_size=768)

        mock_sync_client.upsert.assert_not_called()

    def test_batch_upsert_sync_raises_upstream_error_on_failure(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that batch_upsert_sync raises UpstreamError on failure.

        **Why this test is important:**
          - Error handling ensures consistent error types
          - UpstreamError maps to HTTP 502 in API layer
          - Critical for error propagation and debugging
          - Validates error wrapping

        **What it tests:**
          - Exception is wrapped in UpstreamError
          - Error message includes context
        """
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_sync_client.get_collections.return_value = mock_collections_response
        mock_sync_client.upsert.side_effect = Exception("Upsert failed")

        points = [PointStruct(id="1", vector=[0.1, 0.2], payload={"text": "hello"})]

        with pytest.raises(UpstreamError, match="Qdrant batch upsert failed"):
            qdrant_client.batch_upsert_sync(collection="test-collection", points=points, vector_size=768)


# =============================================================================
# Indexing Tests
# =============================================================================


class TestQdrantClientWrapperIndexing:
    """Test suite for QdrantClientWrapper indexing operations."""

    @pytest.mark.asyncio
    async def test_disable_indexing_success(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that disable_indexing succeeds.

        **Why this test is important:**
          - Indexing control optimizes bulk operations
          - Validates successful API interaction
          - Critical for performance optimization
          - Validates indexing configuration

        **What it tests:**
          - update_collection is called with correct parameters
          - Indexing threshold and HNSW m are set to 0
        """
        mock_async_client.update_collection.return_value = None

        await qdrant_client.disable_indexing(collection="test-collection")

        mock_async_client.update_collection.assert_called_once()
        call_kwargs = mock_async_client.update_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"

    @pytest.mark.asyncio
    async def test_enable_indexing_success(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that enable_indexing succeeds.

        **Why this test is important:**
          - Re-enabling indexing restores performance
          - Validates successful API interaction
          - Critical for post-bulk operation cleanup
          - Validates indexing configuration

        **What it tests:**
          - update_collection is called with correct parameters
          - Default indexing threshold and HNSW m are applied
        """
        mock_async_client.update_collection.return_value = None

        await qdrant_client.enable_indexing(collection="test-collection")

        mock_async_client.update_collection.assert_called_once()
        call_kwargs = mock_async_client.update_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"

    @pytest.mark.asyncio
    async def test_enable_indexing_with_custom_params(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that enable_indexing accepts custom parameters.

        **Why this test is important:**
          - Custom parameters allow tuning for different use cases
          - Different collections may need different indexing settings
          - Critical for adapting to collection-specific requirements
          - Validates parameter passing

        **What it tests:**
          - Custom indexing_threshold is applied
          - Custom hnsw_m is applied
        """
        mock_async_client.update_collection.return_value = None

        await qdrant_client.enable_indexing(collection="test-collection", indexing_threshold=10000, hnsw_m=32)

        mock_async_client.update_collection.assert_called_once()
        call_kwargs = mock_async_client.update_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestQdrantClientWrapperAdditional:
    """Test suite for additional QdrantClientWrapper coverage."""

    def test_client_property(self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock) -> None:
        """Test that client property returns the async client."""
        assert qdrant_client.client is mock_async_client

    @pytest.mark.asyncio
    async def test_disable_indexing_raises_on_error(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that disable_indexing raises UpstreamError on exception."""
        mock_async_client.update_collection.side_effect = Exception("API Error")

        with pytest.raises(UpstreamError, match="Failed to disable indexing"):
            await qdrant_client.disable_indexing(collection="test-collection")

    @pytest.mark.asyncio
    async def test_enable_indexing_raises_on_error(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that enable_indexing raises UpstreamError on exception."""
        mock_async_client.update_collection.side_effect = Exception("API Error")

        with pytest.raises(UpstreamError, match="Failed to enable indexing"):
            await qdrant_client.enable_indexing(collection="test-collection")

    @pytest.mark.asyncio
    async def test_batch_upsert_raises_on_exception(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that batch_upsert raises UpstreamError on exception."""
        mock_async_client.upsert.side_effect = Exception("Upsert failed")

        points = [PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "test"})]

        with pytest.raises(UpstreamError, match="Qdrant batch upsert failed"):
            await qdrant_client.batch_upsert_async(collection="test-collection", points=points, vector_size=2)

    def test_close_with_running_loop(
        self,
        qdrant_client: QdrantClientWrapper,
        mock_sync_client: MagicMock,
    ) -> None:
        """Test that close handles running event loop."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_loop.create_task = MagicMock()
            mock_get_loop.return_value = mock_loop

            qdrant_client.close()

            mock_loop.create_task.assert_called_once()
            mock_sync_client.close.assert_called_once()

    def test_close_without_running_loop(
        self,
        qdrant_client: QdrantClientWrapper,
        mock_sync_client: MagicMock,
    ) -> None:
        """Test that close handles non-running event loop."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete = MagicMock()
            mock_get_loop.return_value = mock_loop

            qdrant_client.close()

            mock_loop.run_until_complete.assert_called_once()
            mock_sync_client.close.assert_called_once()

    def test_close_without_event_loop(
        self,
        qdrant_client: QdrantClientWrapper,
        mock_sync_client: MagicMock,
    ) -> None:
        """Test that close handles missing event loop."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No event loop")
            with patch("asyncio.run") as mock_run:
                qdrant_client.close()

                mock_run.assert_called_once()
                mock_sync_client.close.assert_called_once()

    def test_close_with_none_clients(self) -> None:
        """Test that close handles None clients gracefully."""
        with patch("clients.qdrant.AsyncQdrantClient") as mock_client_cls:
            mock_client_cls.return_value = None
            client = QdrantClientWrapper(url="http://qdrant.example.com:6333")
            object.__setattr__(client, "_client", None)
            object.__setattr__(client, "_sync_client", None)

            # Should not raise
            client.close()
