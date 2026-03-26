# pyright: reportPrivateUsage=false

"""Unit tests for clients.qdrant module.

This file tests the QdrantClientWrapper class which provides vector database operations
via the Qdrant client.

# Test Coverage

The tests cover:
  - Client Initialization: Default configuration, from_config factory
  - Collection Management: ensure_collection, collection existence checking
  - Vector Operations: search, batch_upsert_async
  - Indexing Operations: disable_indexing, enable_indexing
  - Circuit Breaker Integration: Circuit breaker usage, error handling
  - Error Handling: UpstreamError on failures, circuit breaker errors

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The underlying Qdrant clients and circuit breaker are mocked to isolate client logic.

# Running Tests

Run with: pytest tests/unit/clients/test_qdrant.py
"""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiobreaker
import aiobreaker.state as aio_state
import pybreaker
import pytest
from qdrant_client.http import models as qmodels
from qdrant_client.models import PointStruct

from clients.qdrant import QdrantClientWrapper, _DISTANCE_METRIC_MAP
from config import VectorDBConfig
from core.models import SearchResults
from foundation.exceptions import UpstreamError

# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestQdrantClientWrapperInit:
    """Test suite for QdrantClientWrapper initialization."""

    @patch("clients.qdrant.AsyncQdrantClient")
    def test_creates_client_with_config(
        self,
        mock_async_qdrant_client: MagicMock,
    ) -> None:
        """Test that client is created with configuration.

        **Why this test is important:**
          - Client initialization is the foundation for all operations
          - Ensures configuration is applied correctly
          - Validates that Qdrant client is created with correct parameters
          - Critical for basic functionality

        **What it tests:**
          - AsyncQdrantClient is created with correct URL and timeout
          - Client attributes are set correctly
          - Circuit breaker is created
        """
        mock_async_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_async_client

        client = QdrantClientWrapper(url="http://qdrant.example.com:6333")

        mock_async_qdrant_client.assert_called_once_with(
            url="http://qdrant.example.com:6333", api_key=None, timeout=300, pool_size=10
        )
        assert client.url == "http://qdrant.example.com:6333"
        assert client._client == mock_async_client

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

    def test_creates_circuit_breaker_with_custom_config(self) -> None:
        """Test that circuit breaker uses custom configuration.

        **Why this test is important:**
          - Custom circuit breaker settings are needed for different environments
          - Production may need different thresholds than development
          - Validates that configuration is properly applied

        **What it tests:**
          - Custom failure threshold is applied
          - Custom recovery timeout is applied
        """
        client = QdrantClientWrapper(
            url="http://qdrant.example.com:6333",
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=120,
        )

        assert client._breaker.fail_max == 10
        assert client._breaker.reset_timeout == 120

    @patch("clients.qdrant.AsyncQdrantClient")
    def test_creates_client_with_custom_timeout(
        self,
        mock_async_qdrant_client: MagicMock,
    ) -> None:
        """Test that custom timeout is passed to Qdrant client.

        **Why this test is important:**
          - Timeout configuration is critical for reliability
          - Different environments may need different timeouts
          - Validates that timeout setting propagates correctly

        **What it tests:**
          - AsyncQdrantClient receives custom timeout
        """
        mock_async_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_async_client

        QdrantClientWrapper(
            url="http://qdrant.example.com:6333",
            timeout_s=600,
        )

        mock_async_qdrant_client.assert_called_once_with(
            url="http://qdrant.example.com:6333",
            api_key=None,
            timeout=600,
            pool_size=10,
        )

    @patch("clients.qdrant.AsyncQdrantClient")
    def test_creates_client_with_api_key(
        self,
        mock_async_qdrant_client: MagicMock,
    ) -> None:
        """Test that api_key is passed to Qdrant client.

        **Why this test is important:**
          - API key is required for cloud Qdrant instances
          - Ensures authentication credentials are passed correctly
          - Critical for cloud deployment security

        **What it tests:**
          - AsyncQdrantClient is created with api_key
          - Client stores the api_key attribute
        """
        mock_async_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_async_client

        client = QdrantClientWrapper(
            url="https://qdrant.cloud.example.com:6333",
            api_key="test-api-key-12345",
        )

        mock_async_qdrant_client.assert_called_once_with(
            url="https://qdrant.cloud.example.com:6333",
            api_key="test-api-key-12345",
            timeout=300,
            pool_size=10,
        )
        assert client.api_key == "test-api-key-12345"

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

        with patch("clients.qdrant.AsyncQdrantClient"):
            client = QdrantClientWrapper.from_config(config)

        assert client.url == "http://qdrant.example.com:6333"

    @patch("clients.qdrant.AsyncQdrantClient")
    def test_from_config_passes_api_key(
        self,
        mock_async_qdrant_client: MagicMock,
    ) -> None:
        """Test that from_config passes api_key from configuration.

        **Why this test is important:**
          - API key from config must be passed to client
          - Validates configuration integration for cloud deployments
          - Critical for secure cloud Qdrant access

        **What it tests:**
          - qdrant_api_key from VectorDBConfig is passed to client
          - Client is created with correct api_key attribute
        """
        config = VectorDBConfig(
            provider_type="qdrant",
            collection="test-collection",
            qdrant_url="https://qdrant.cloud.example.com:6333",
            qdrant_api_key="config-api-key-67890",
        )

        client = QdrantClientWrapper.from_config(config)

        mock_async_qdrant_client.assert_called_once_with(
            url="https://qdrant.cloud.example.com:6333",
            api_key="config-api-key-67890",
            timeout=300,
            pool_size=10,
        )
        assert client.api_key == "config-api-key-67890"

    @patch("clients.qdrant.AsyncQdrantClient")
    def test_from_config_passes_resilience_settings(
        self,
        mock_async_qdrant_client: MagicMock,
    ) -> None:
        """Test that from_config passes all resilience settings from config.

        **Why this test is important:**
          - Resilience settings must propagate from config to client
          - Ensures timeout, circuit breaker threshold, and timeout are applied
          - Critical for environment-specific configuration

        **What it tests:**
          - qdrant_timeout is passed as timeout_s
          - qdrant_circuit_breaker_threshold is passed correctly
          - qdrant_circuit_breaker_timeout is passed correctly
          - Circuit breaker is configured with custom values
        """
        mock_async_client = AsyncMock()
        mock_async_qdrant_client.return_value = mock_async_client

        config = VectorDBConfig(
            provider_type="qdrant",
            collection="test-collection",
            qdrant_url="http://qdrant.example.com:6333",
            qdrant_timeout=600,
            qdrant_circuit_breaker_threshold=10,
            qdrant_circuit_breaker_timeout=120,
        )

        client = QdrantClientWrapper.from_config(config)

        # Verify timeout was passed to client
        mock_async_qdrant_client.assert_called_once_with(
            url="http://qdrant.example.com:6333",
            api_key=None,
            timeout=600,
            pool_size=10,
        )

        # Verify client attributes
        assert client.timeout_s == 600
        assert client.circuit_breaker_threshold == 10
        assert client.circuit_breaker_timeout == 120

        # Verify circuit breaker was configured with custom values
        assert client._breaker.fail_max == 10
        assert client._breaker.reset_timeout == 120

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
        # Use a MagicMock since VectorDBConfig now only allows "qdrant"
        config = MagicMock(spec=VectorDBConfig)
        config.provider_type = "other"
        config.collection = "test-collection"
        config.qdrant_url = None
        config.qdrant_api_key = None

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

    @pytest.mark.asyncio
    async def test_ensure_image_collection_creates_if_missing(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_image_collection creates collection if missing.

        **Why this test is important:**
          - Image collection creation is essential for image embeddings
          - Ensures collections exist before use with correct naming pattern
          - Critical for dev convenience functions
          - Validates image collection creation logic

        **What it tests:**
          - get_collections is called to check existence
          - create_collection is called with collection
          - Default vector_size is 512 (CLIP default)
        """
        mock_collections = MagicMock()
        mock_collections.collections = []  # Empty, collection doesn't exist
        mock_async_client.get_collections.return_value = mock_collections

        await qdrant_client.ensure_image_collection_async(collection="documents")

        mock_async_client.get_collections.assert_called_once()
        mock_async_client.create_collection.assert_called_once()
        call_kwargs = mock_async_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "documents"
        assert call_kwargs["vectors_config"].size == 512  # CLIP default

    @pytest.mark.asyncio
    async def test_ensure_image_collection_creates_with_custom_vector_size(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_image_collection accepts custom vector_size.

        **Why this test is important:**
          - Different embedding models have different vector dimensions
          - Custom vector_size allows flexibility for different models
          - Critical for supporting multiple embedding backends
          - Validates parameter passing

        **What it tests:**
          - Custom vector_size is applied correctly
        """
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections

        await qdrant_client.ensure_image_collection_async(collection="photos", vector_size=768)

        call_kwargs = mock_async_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "photos"
        assert call_kwargs["vectors_config"].size == 768

    @pytest.mark.asyncio
    async def test_ensure_image_collection_creates_with_custom_distance_metric(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_image_collection accepts custom distance_metric.

        **Why this test is important:**
          - Different use cases may require different distance metrics
          - Supports cosine, euclidean, and dot product similarity
          - Critical for flexibility in vector similarity comparisons

        **What it tests:**
          - create_collection is called with the mapped distance metric
          - distance_metric parameter correctly maps to qmodels.Distance enum
        """
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections

        await qdrant_client.ensure_image_collection_async(collection="photos", distance_metric="euclidean")

        call_kwargs = mock_async_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "photos"
        assert call_kwargs["vectors_config"].distance == qmodels.Distance.EUCLID

    @pytest.mark.asyncio
    async def test_ensure_image_collection_distance_metric_mapping(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that distance_metric mapping covers all supported values.

        **Why this test is important:**
          - Ensures all advertised distance metrics are properly mapped
          - Validates the _DISTANCE_METRIC_MAP constant

        **What it tests:**
          - "cosine" maps to qmodels.Distance.COSINE
          - "euclidean" maps to qmodels.Distance.EUCLID
          - "dot" maps to qmodels.Distance.DOT
        """
        assert _DISTANCE_METRIC_MAP["cosine"] == qmodels.Distance.COSINE
        assert _DISTANCE_METRIC_MAP["euclidean"] == qmodels.Distance.EUCLID
        assert _DISTANCE_METRIC_MAP["dot"] == qmodels.Distance.DOT

    @pytest.mark.asyncio
    async def test_ensure_image_collection_skips_if_exists(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_image_collection skips creation if collection exists.

        **Why this test is important:**
          - Idempotent operations prevent errors
          - Avoids unnecessary API calls
          - Critical for efficiency

        **What it tests:**
          - get_collections is called to check existence
          - create_collection is not called if collection exists
        """
        mock_collection = MagicMock()
        mock_collection.name = "documents"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_async_client.get_collections.return_value = mock_collections

        await qdrant_client.ensure_image_collection_async(collection="documents")

        mock_async_client.get_collections.assert_called_once()
        mock_async_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_image_collection_naming_pattern(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_image_collection uses correct naming pattern.

        **Why this test is important:**
          - Ensures image collections are clearly distinguished from text collections
          - Critical for collection organization and management
          - Validates naming convention

        **What it tests:**
          - Different base collection names produce correct image collection names
        """
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections

        # Test with different collection names
        test_cases = [
            ("documents", "documents"),
            ("photos", "photos"),
            ("test", "test"),
        ]

        for base_name, expected_name in test_cases:
            mock_async_client.get_collections.return_value = mock_collections
            await qdrant_client.ensure_image_collection_async(collection=base_name)
            call_kwargs = mock_async_client.create_collection.call_args[1]
            assert call_kwargs["collection_name"] == expected_name

    @pytest.mark.asyncio
    async def test_ensure_collection_passes_quantization_config(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_collection_async passes quantization_config to create_collection.

        **Why this test is important:**
          - Quantization config must be set at collection creation time
          - Qdrant applies quantization on write when configured
          - Enables INT8/binary quantization without a separate setup step

        **What it tests:**
          - quantization_config is forwarded to create_collection
        """
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections

        scalar_config = qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        )

        await qdrant_client.ensure_collection_async(
            collection="test-quantized",
            vector_size=1152,
            quantization_config=scalar_config,
        )

        call_kwargs = mock_async_client.create_collection.call_args[1]
        assert call_kwargs["quantization_config"] is scalar_config

    @pytest.mark.asyncio
    async def test_ensure_collection_defaults_to_no_quantization(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_collection_async defaults to no quantization.

        **Why this test is important:**
          - Backward compatibility: existing callers must not break
          - Default behavior should match pre-quantization behavior

        **What it tests:**
          - quantization_config defaults to None when not specified
        """
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections

        await qdrant_client.ensure_collection_async(collection="test-default", vector_size=768)

        call_kwargs = mock_async_client.create_collection.call_args[1]
        assert call_kwargs["quantization_config"] is None

    @pytest.mark.asyncio
    async def test_ensure_image_collection_passes_quantization_config(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that ensure_image_collection_async passes quantization_config.

        **Why this test is important:**
          - Image collections are the primary ingestion path
          - Quantization config must be threaded through to create_collection

        **What it tests:**
          - quantization_config is forwarded to create_collection for image collections
        """
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_async_client.get_collections.return_value = mock_collections

        binary_config = qmodels.BinaryQuantization(
            binary=qmodels.BinaryQuantizationConfig(always_ram=True),
        )

        await qdrant_client.ensure_image_collection_async(
            collection="images",
            vector_size=1152,
            quantization_config=binary_config,
        )

        call_kwargs = mock_async_client.create_collection.call_args[1]
        assert call_kwargs["quantization_config"] is binary_config


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
    async def test_search_async_passes_search_params(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that search_async passes search_params to query_points.

        **Why this test is important:**
          - Quantization benchmarking requires passing Qdrant-specific
            SearchParams (rescore, oversampling) through to query_points.
          - Validates that the search_params kwarg is forwarded correctly.

        **What it tests:**
          - query_points receives the search_params kwarg.
          - None default works when search_params is omitted.
        """
        mock_response = MagicMock()
        mock_response.points = []
        mock_async_client.query_points.return_value = mock_response

        search_params = qmodels.SearchParams(
            quantization=qmodels.QuantizationSearchParams(rescore=True, oversampling=3.0)
        )

        await qdrant_client.search_async(
            collection="test-collection",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
            search_params=search_params,
        )

        mock_async_client.query_points.assert_called_once_with(
            collection_name="test-collection",
            query=[0.1, 0.2, 0.3],
            limit=10,
            with_payload=True,
            search_params=search_params,
        )

    @pytest.mark.asyncio
    async def test_search_async_default_search_params_is_none(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that search_async defaults search_params to None.

        **Why this test is important:**
          - Ensures backward compatibility: callers that omit search_params
            get None passed through (Qdrant SDK default behavior).

        **What it tests:**
          - query_points receives search_params=None when not specified.
        """
        mock_response = MagicMock()
        mock_response.points = []
        mock_async_client.query_points.return_value = mock_response

        await qdrant_client.search_async(
            collection="test-collection",
            query_vector=[0.1, 0.2],
            limit=5,
        )

        mock_async_client.query_points.assert_called_once_with(
            collection_name="test-collection",
            query=[0.1, 0.2],
            limit=5,
            with_payload=True,
            search_params=None,
        )

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

        with pytest.raises(UpstreamError, match="Qdrant search_async failed"):
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
          - aiobreaker.CircuitBreakerError is caught and converted to UpstreamError
        """
        # aiobreaker.CircuitBreakerError requires message and reopen_time
        mock_async_client.query_points.side_effect = aiobreaker.CircuitBreakerError(
            "Circuit is open", datetime.datetime.now(datetime.timezone.utc)
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
          - Open circuit breaker state (via aiobreaker) triggers fail-fast
          - handle_circuit_breaker_error is called
        """
        # Create a mock async breaker in OPEN state
        mock_breaker = MagicMock()
        # aiobreaker uses .current_state for state checking (returns enum)
        mock_breaker.current_state = aio_state.CircuitBreakerState.OPEN
        object.__setattr__(qdrant_client, "_async_breaker", mock_breaker)

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

        with pytest.raises(UpstreamError, match="Qdrant batch_upsert_async failed"):
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
          - Open circuit breaker state (via aiobreaker) triggers fail-fast
          - handle_circuit_breaker_error is called
        """
        # Create a mock async breaker in OPEN state
        mock_breaker = MagicMock()
        # aiobreaker uses .current_state for state checking (returns enum)
        mock_breaker.current_state = aio_state.CircuitBreakerState.OPEN
        object.__setattr__(qdrant_client, "_async_breaker", mock_breaker)

        points = [PointStruct(id="1", vector=[0.1, 0.2], payload={"text": "hello"})]

        with pytest.raises(UpstreamError, match="qdrant service is currently unavailable"):
            await qdrant_client.batch_upsert_async(
                collection="test-collection", points=points, vector_size=768
            )


# =============================================================================
# Indexing Tests
# =============================================================================


class TestQdrantClientWrapperIndexing:
    """Test suite for QdrantClientWrapper indexing operations."""

    def test_disable_indexing_success(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that disable_indexing_sync succeeds.

        **Why this test is important:**
          - Indexing control optimizes bulk operations
          - Validates successful API interaction
          - Critical for performance optimization
          - Validates indexing configuration

        **What it tests:**
          - update_collection is called with correct parameters
          - Indexing threshold and HNSW m are set to 0
        """
        mock_sync_client.get_collection.return_value = MagicMock()
        mock_sync_client.update_collection.return_value = None

        result = qdrant_client.disable_indexing_sync(collection="test-collection")

        assert result is not None
        mock_sync_client.update_collection.assert_called_once()
        call_kwargs = mock_sync_client.update_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["optimizer_config"].indexing_threshold == 0
        assert call_kwargs["hnsw_config"].m == 0

    def test_enable_indexing_success(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that enable_indexing_sync succeeds.

        **Why this test is important:**
          - Re-enabling indexing restores performance
          - Validates successful API interaction
          - Critical for post-bulk operation cleanup
          - Validates indexing configuration

        **What it tests:**
          - update_collection is called with correct parameters
          - Default indexing threshold and HNSW m are applied
        """
        mock_sync_client.update_collection.return_value = None

        qdrant_client.enable_indexing_sync(collection="test-collection")

        mock_sync_client.update_collection.assert_called_once()
        call_kwargs = mock_sync_client.update_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["optimizer_config"].indexing_threshold == 20_000
        assert call_kwargs["hnsw_config"].m == 16

    def test_enable_indexing_with_custom_params(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that enable_indexing_sync accepts custom parameters.

        **Why this test is important:**
          - Custom parameters allow tuning for different use cases
          - Different collections may need different indexing settings
          - Critical for adapting to collection-specific requirements
          - Validates parameter passing

        **What it tests:**
          - Custom indexing_threshold is applied
          - Custom hnsw_m is applied
        """
        mock_sync_client.update_collection.return_value = None

        qdrant_client.enable_indexing_sync(collection="test-collection", indexing_threshold=10000, hnsw_m=32)

        mock_sync_client.update_collection.assert_called_once()
        call_kwargs = mock_sync_client.update_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test-collection"
        assert call_kwargs["optimizer_config"].indexing_threshold == 10000
        assert call_kwargs["hnsw_config"].m == 32


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestQdrantClientWrapperAdditional:
    """Test suite for additional QdrantClientWrapper coverage."""

    def test_client_property(self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock) -> None:
        """Test that client property returns the async client."""
        assert qdrant_client.client is mock_async_client

    def test_disable_indexing_raises_on_error(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that disable_indexing_sync raises UpstreamError on exception."""
        mock_sync_client.get_collection.side_effect = Exception("API Error")

        with pytest.raises(UpstreamError, match="Failed to disable indexing for collection"):
            qdrant_client.disable_indexing_sync(collection="test-collection")

    def test_enable_indexing_raises_on_error(
        self, qdrant_client: QdrantClientWrapper, mock_sync_client: MagicMock
    ) -> None:
        """Test that enable_indexing_sync raises UpstreamError on exception."""
        mock_sync_client.update_collection.side_effect = Exception("API Error")

        with pytest.raises(UpstreamError, match="Failed to enable indexing for collection"):
            qdrant_client.enable_indexing_sync(collection="test-collection")

    @pytest.mark.asyncio
    async def test_batch_upsert_raises_on_exception(
        self, qdrant_client: QdrantClientWrapper, mock_async_client: AsyncMock
    ) -> None:
        """Test that batch_upsert raises UpstreamError on exception."""
        mock_async_client.upsert.side_effect = Exception("Upsert failed")

        points = [PointStruct(id=1, vector=[0.1, 0.2], payload={"text": "test"})]

        with pytest.raises(UpstreamError, match="Qdrant batch_upsert_async failed"):
            await qdrant_client.batch_upsert_async(collection="test-collection", points=points, vector_size=2)

    def test_close_calls_asyncio_run(
        self,
        qdrant_client: QdrantClientWrapper,
    ) -> None:
        """Test that close calls asyncio.run with close_async_resource."""
        with patch("clients.qdrant.asyncio.run") as mock_run:
            qdrant_client.close()

            mock_run.assert_called_once()

    def test_close_sets_client_to_none(
        self,
        qdrant_client: QdrantClientWrapper,
    ) -> None:
        """Test that close sets _client to None."""
        with patch("clients.qdrant.asyncio.run"):
            qdrant_client.close()

        assert qdrant_client._client is None

    def test_close_is_idempotent(
        self,
        qdrant_client: QdrantClientWrapper,
    ) -> None:
        """Test that close is safe to call multiple times."""
        with patch("clients.qdrant.asyncio.run") as mock_run:
            qdrant_client.close()
            qdrant_client.close()  # second call is a no-op

            mock_run.assert_called_once()

    def test_close_with_none_client(self) -> None:
        """Test that close handles None client gracefully."""
        with patch("clients.qdrant.AsyncQdrantClient") as mock_client_cls:
            mock_client_cls.return_value = None
            client = QdrantClientWrapper(url="http://qdrant.example.com:6333")
            object.__setattr__(client, "_client", None)

            # Should not raise
            client.close()
            assert client._client is None
