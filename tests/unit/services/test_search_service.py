"""Unit tests for core.services.search_service module.

This file tests the SearchService and ImageSearchService classes which provide
semantic search orchestration by coordinating embedding generation and vector
database queries.

# Test Coverage

The tests cover:

ImageSearchService:
  - Service Initialization: CLIP client and vector DB provider injection
  - Search Images: Query validation, CLIP text embedding, image collection search
  - Async Search: Async operations, error handling
  - Input Validation: Empty queries, invalid limits
  - Error Handling: BadRequestError on validation, UpstreamError propagation
  - Cache Integration: Cache hit/miss/error scenarios

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The embedding provider and vector DB provider are mocked to isolate
service logic.

# Running Tests

Run with: pytest tests/unit/services/test_search_service.py
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import attrs.exceptions
import pytest

from foundation.exceptions import UpstreamError
from core.exceptions import BadRequestError
from core.models import SearchResultItem as SearchItem
from core.models import SearchResults
from core.services.search_service import ImageSearchService


# =============================================================================
# ImageSearchService Initialization Tests
# =============================================================================


class TestImageSearchServiceInit:
    """Test suite for ImageSearchService initialization."""

    def test_creates_service_with_providers(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test that service is created with embedding provider and vector DB provider.

        **Why this test is important:**
          - Service requires both embedding provider and vector DB provider
          - Validates dependency injection
          - Critical for initialization
          - Validates attrs integration

        **What it tests:**
          - Service is created with embedding provider
          - Service is created with vector DB provider
          - Providers are accessible as attributes
        """
        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
        )

        assert service.embedding_provider is mock_embedding_provider
        assert service.vector_db_provider is mock_image_vector_db_provider

    def test_service_is_frozen(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test that service is immutable (frozen=True).

        **Why this test is important:**
          - Immutability prevents accidental modification
          - Ensures thread safety
          - Critical for service reliability
          - Validates attrs frozen configuration

        **What it tests:**
          - Attributes cannot be modified after creation
          - FrozenInstanceError is raised on modification attempt
        """
        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            service.embedding_provider = MagicMock()


# =============================================================================
# ImageSearchService Search Images Tests
# =============================================================================


class TestImageSearchServiceSearchImages:
    """Test suite for ImageSearchService.search_images_async method."""

    @pytest.mark.asyncio
    async def test_search_success(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async performs text-to-image search successfully.

        **Why this test is important:**
          - Image search is the core functionality
          - Validates CLIP text embedding + vector DB orchestration
          - Ensures proper result formatting
          - Critical for basic functionality

        **What it tests:**
          - CLIP client embed_text is called with query
          - Vector DB provider search is called with embedding
          - Search results are returned correctly
        """
        result = await image_search_service.search_images_async(
            collection="documents",
            query="sunset over ocean",
            limit=10,
        )

        # Verify CLIP text embedding was generated
        image_search_service.embedding_provider.embed_text.assert_called_once_with("sunset over ocean")

        # Verify vector DB was searched with image collection name
        image_search_service.vector_db_provider.search_async.assert_called_once()
        call_kwargs = image_search_service.vector_db_provider.search_async.call_args[1]
        assert call_kwargs["collection"] == "documents"
        assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["limit"] == 10

        # Verify results
        assert isinstance(result, SearchResults)
        assert len(result.items) == 2
        assert result.total == 2
        assert result.items[0].score == 0.92
        assert result.items[1].score == 0.85

    @pytest.mark.asyncio
    async def test_search_strips_whitespace(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async strips whitespace from query.

        **Why this test is important:**
          - Whitespace can affect embeddings
          - Query normalization improves consistency
          - Critical for search quality
          - Validates input preprocessing

        **What it tests:**
          - Query is stripped before embedding generation
          - Leading/trailing whitespace is removed
        """
        await image_search_service.search_images_async(
            collection="documents",
            query="  sunset over ocean  ",
            limit=10,
        )

        image_search_service.embedding_provider.embed_text.assert_called_once_with("sunset over ocean")

    @pytest.mark.asyncio
    async def test_search_raises_on_empty_query(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async raises BadRequestError for empty query.

        **Why this test is important:**
          - Empty queries are invalid
          - Validation prevents wasted API calls
          - Critical for error prevention
          - Validates input validation

        **What it tests:**
          - BadRequestError is raised for empty string
          - Error message is descriptive
        """
        with pytest.raises(BadRequestError, match="Query string cannot be empty"):
            await image_search_service.search_images_async(collection="documents", query="", limit=10)

    @pytest.mark.asyncio
    async def test_search_raises_on_whitespace_only_query(
        self, image_search_service: ImageSearchService
    ) -> None:
        """Test that search_images_async raises BadRequestError for whitespace-only query.

        **Why this test is important:**
          - Whitespace-only queries are effectively empty
          - Validation catches edge cases
          - Critical for error prevention
          - Validates input validation

        **What it tests:**
          - BadRequestError is raised for whitespace-only string
          - Query.strip() is used for validation
        """
        with pytest.raises(BadRequestError, match="Query string cannot be empty"):
            await image_search_service.search_images_async(collection="documents", query="   ", limit=10)

    @pytest.mark.asyncio
    async def test_search_raises_on_invalid_limit_too_small(
        self, image_search_service: ImageSearchService
    ) -> None:
        """Test that search_images_async raises BadRequestError for limit < 1.

        **Why this test is important:**
          - Limit must be positive
          - Validation prevents invalid API calls
          - Critical for error prevention
          - Validates input validation

        **What it tests:**
          - BadRequestError is raised for limit=0
          - Error message is descriptive
        """
        with pytest.raises(BadRequestError, match="Limit must be between 1 and 100"):
            await image_search_service.search_images_async(collection="documents", query="test", limit=0)

    @pytest.mark.asyncio
    async def test_search_raises_on_invalid_limit_too_large(
        self, image_search_service: ImageSearchService
    ) -> None:
        """Test that search_images_async raises BadRequestError for limit > 100.

        **Why this test is important:**
          - Limit must be reasonable
          - Prevents resource exhaustion
          - Critical for service protection
          - Validates input validation

        **What it tests:**
          - BadRequestError is raised for limit=101
          - Upper bound is enforced
        """
        with pytest.raises(BadRequestError, match="Limit must be between 1 and 100"):
            await image_search_service.search_images_async(collection="documents", query="test", limit=101)

    @pytest.mark.asyncio
    async def test_search_accepts_valid_limit_range(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async accepts valid limit values.

        **Why this test is important:**
          - Valid limits should work
          - Validates boundary conditions
          - Critical for functionality
          - Validates validation logic

        **What it tests:**
          - Limit=1 is accepted (lower boundary)
          - Limit=100 is accepted (upper boundary)
          - Limit=50 is accepted (mid-range)
        """
        # Lower boundary
        result_1 = await image_search_service.search_images_async(
            collection="documents", query="test", limit=1
        )
        assert result_1 is not None

        # Upper boundary
        result_100 = await image_search_service.search_images_async(
            collection="documents", query="test", limit=100
        )
        assert result_100 is not None

        # Mid-range
        result_50 = await image_search_service.search_images_async(
            collection="documents", query="test", limit=50
        )
        assert result_50 is not None

    @pytest.mark.asyncio
    async def test_search_propagates_embedding_error(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async propagates embedding provider errors.

        **Why this test is important:**
          - embedding errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from embedding provider is propagated
          - Error is not swallowed
        """
        image_search_service.embedding_provider.embed_text.side_effect = UpstreamError(
            "embedding provider failed"
        )

        with pytest.raises(UpstreamError, match="embedding provider failed"):
            await image_search_service.search_images_async(collection="documents", query="test", limit=10)

    @pytest.mark.asyncio
    async def test_search_propagates_vector_db_error(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async propagates vector DB provider errors.

        **Why this test is important:**
          - Vector DB errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from vector DB provider is propagated
          - Error is not swallowed
        """
        image_search_service.vector_db_provider.search_async.side_effect = UpstreamError(
            "Qdrant connection failed"
        )

        with pytest.raises(UpstreamError, match="Qdrant connection failed"):
            await image_search_service.search_images_async(collection="documents", query="test", limit=10)


# =============================================================================
# ImageSearchService Async Search Images Tests
# =============================================================================


class TestImageSearchServiceSearchImagesAsync:
    """Test suite for ImageSearchService.search_images_async method."""

    @pytest.mark.asyncio
    async def test_search_async_success(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async performs text-to-image search successfully.

        **Why this test is important:**
          - Async image search enables non-blocking operations
          - Validates async orchestration
          - Ensures proper result formatting
          - Critical for API performance

        **What it tests:**
          - EmbeddingProvider embed_text is called with query
          - Vector DB provider search is called with embedding
          - Search results are returned correctly
        """
        result = await image_search_service.search_images_async(
            collection="photos",
            query="fluffy cat",
            limit=10,
        )

        # Verify text embedding was generated
        image_search_service.embedding_provider.embed_text.assert_called_once_with("fluffy cat")

        # Verify vector DB was searched with image collection name
        image_search_service.vector_db_provider.search_async.assert_called_once()
        call_kwargs = image_search_service.vector_db_provider.search_async.call_args[1]
        assert call_kwargs["collection"] == "photos"
        assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["limit"] == 10

        # Verify results
        assert isinstance(result, SearchResults)
        assert len(result.items) == 2
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_search_async_strips_whitespace(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async strips whitespace from query.

        **Why this test is important:**
          - Whitespace can affect embeddings
          - Query normalization improves consistency
          - Critical for search quality
          - Validates input preprocessing

        **What it tests:**
          - Query is stripped before embedding generation
          - Leading/trailing whitespace is removed
        """
        await image_search_service.search_images_async(
            collection="photos",
            query="  fluffy cat  ",
            limit=10,
        )

        image_search_service.embedding_provider.embed_text.assert_called_once_with("fluffy cat")

    @pytest.mark.asyncio
    async def test_search_async_raises_on_empty_query(self, image_search_service: ImageSearchService) -> None:
        """Test that search_images_async raises BadRequestError for empty query.

        **Why this test is important:**
          - Empty queries are invalid
          - Validation prevents wasted API calls
          - Critical for error prevention
          - Validates input validation

        **What it tests:**
          - BadRequestError is raised for empty string
          - Error message is descriptive
        """
        with pytest.raises(BadRequestError, match="Query string cannot be empty"):
            await image_search_service.search_images_async(collection="photos", query="", limit=10)

    @pytest.mark.asyncio
    async def test_search_async_raises_on_invalid_limit(
        self, image_search_service: ImageSearchService
    ) -> None:
        """Test that search_images_async validates limit parameter.

        **Why this test is important:**
          - Limit validation prevents invalid requests
          - Same validation as sync version
          - Critical for consistency
          - Validates input validation

        **What it tests:**
          - BadRequestError is raised for invalid limits
          - Validation logic matches sync version
        """
        with pytest.raises(BadRequestError, match="Limit must be between 1 and 100"):
            await image_search_service.search_images_async(collection="photos", query="test", limit=0)

        with pytest.raises(BadRequestError, match="Limit must be between 1 and 100"):
            await image_search_service.search_images_async(collection="photos", query="test", limit=101)

    @pytest.mark.asyncio
    async def test_search_async_propagates_embedding_error(
        self, image_search_service: ImageSearchService
    ) -> None:
        """Test that search_images_async propagates client errors.

        **Why this test is important:**
          - EmbeddingProvider errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from embedding provider is propagated
          - Error is not swallowed
        """
        image_search_service.embedding_provider.embed_text.side_effect = UpstreamError(
            "embedding provider failed"
        )

        with pytest.raises(UpstreamError, match="embedding provider failed"):
            await image_search_service.search_images_async(collection="photos", query="test", limit=10)

    @pytest.mark.asyncio
    async def test_search_async_propagates_vector_db_error(
        self, image_search_service: ImageSearchService
    ) -> None:
        """Test that search_images_async propagates vector DB provider errors.

        **Why this test is important:**
          - Vector DB errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from vector DB provider is propagated
          - Error is not swallowed
        """
        image_search_service.vector_db_provider.search_async.side_effect = UpstreamError(
            "Qdrant connection failed"
        )

        with pytest.raises(UpstreamError, match="Qdrant connection failed"):
            await image_search_service.search_images_async(collection="photos", query="test", limit=10)


# =============================================================================
# ImageSearchService Integration Tests
# =============================================================================


class TestImageSearchServiceIntegration:
    """Test suite for end-to-end ImageSearchService integration."""

    @pytest.mark.asyncio
    async def test_full_image_search_workflow(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test complete image search workflow: validate -> embed -> search -> format.

        **Why this test is important:**
          - Validates end-to-end workflow
          - Ensures all steps work together
          - Critical for real-world usage
          - Validates integration

        **What it tests:**
          - Input validation passes for valid query
          - Text embedding is generated correctly
          - Vector DB search is performed on image collection
          - Results include image metadata
        """
        # Setup mock responses
        mock_embedding_provider.embed_text.return_value = [0.5, 0.6, 0.7, 0.8]
        mock_image_vector_db_provider.search_async.return_value = SearchResults(
            items=[
                SearchItem(
                    point_id="sunset-001",
                    score=0.95,
                    payload={
                        "s3_key": "images/sunset-001.jpg",
                        "s3_uri": "s3://pipeline/images/sunset-001.jpg",
                        "format": "jpeg",
                        "width": 1920,
                        "height": 1080,
                        "thumbnail_key": "thumbnails/sunset-001.jpg",
                    },
                ),
                SearchItem(
                    point_id="beach-002",
                    score=0.88,
                    payload={
                        "s3_key": "images/beach-002.png",
                        "s3_uri": "s3://pipeline/images/beach-002.png",
                        "format": "png",
                        "width": 1280,
                        "height": 720,
                    },
                ),
            ],
            total=2,
        )

        # Create service
        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
        )

        # Perform search
        results = await service.search_images_async(
            collection="vacation",
            query="beautiful sunset over the ocean",
            limit=5,
        )

        # Verify text embedding
        mock_embedding_provider.embed_text.assert_called_once_with("beautiful sunset over the ocean")

        # Verify vector DB search with image collection
        mock_image_vector_db_provider.search_async.assert_called_once()
        call_kwargs = mock_image_vector_db_provider.search_async.call_args[1]
        assert call_kwargs["collection"] == "vacation"
        assert call_kwargs["query_vector"] == [0.5, 0.6, 0.7, 0.8]
        assert call_kwargs["limit"] == 5

        # Verify results
        assert results.total == 2
        assert len(results.items) == 2
        assert results.items[0].point_id == "sunset-001"
        assert results.items[0].score == 0.95
        assert results.items[0].payload["s3_key"] == "images/sunset-001.jpg"
        assert results.items[0].payload["format"] == "jpeg"
        assert results.items[1].point_id == "beach-002"
        assert results.items[1].score == 0.88

    @pytest.mark.asyncio
    async def test_full_image_search_workflow_async(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Test complete async image search workflow.

        **Why this test is important:**
          - Validates end-to-end async workflow
          - Ensures all async steps work together
          - Critical for API performance
          - Validates async integration

        **What it tests:**
          - Input validation passes for valid query
          - Async text embedding is generated correctly
          - Vector DB search is performed
          - Results are formatted correctly
        """
        # Setup mock responses
        mock_embedding_provider.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_image_vector_db_provider.search_async.return_value = SearchResults(
            items=[
                SearchItem(
                    point_id="cat-001",
                    score=0.91,
                    payload={
                        "s3_key": "images/cat-001.jpg",
                        "s3_uri": "s3://pipeline/images/cat-001.jpg",
                        "format": "jpeg",
                        "width": 800,
                        "height": 600,
                    },
                ),
            ],
            total=1,
        )

        # Create service
        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
        )

        # Perform async search
        results = await service.search_images_async(
            collection="pets",
            query="fluffy cat",
            limit=10,
        )

        # Verify text embedding
        mock_embedding_provider.embed_text.assert_called_once_with("fluffy cat")

        # Verify vector DB search
        mock_image_vector_db_provider.search_async.assert_called_once()
        call_kwargs = mock_image_vector_db_provider.search_async.call_args[1]
        assert call_kwargs["collection"] == "pets"

        # Verify results
        assert results.total == 1
        assert len(results.items) == 1
        assert results.items[0].point_id == "cat-001"
        assert results.items[0].payload["s3_key"] == "images/cat-001.jpg"


# =============================================================================
# ImageSearchService Cache Integration Tests
# =============================================================================


class TestImageSearchServiceCacheIntegration:
    """Test suite for semantic cache integration in ImageSearchService."""

    @staticmethod
    def _make_mock_cache() -> MagicMock:
        """Create a mock CacheClient with AsyncMock methods."""
        cache = MagicMock()
        cache.lookup = AsyncMock(return_value=None)
        cache.store = AsyncMock()
        return cache

    @staticmethod
    def _make_cached_results() -> SearchResults:
        """Return a SearchResults instance representing a cache hit."""
        return SearchResults(
            items=[
                SearchItem(
                    point_id="cached-1",
                    score=0.99,
                    payload={"s3_key": "images/cached.jpg", "s3_uri": "s3://b/cached.jpg"},
                ),
            ],
            total=1,
        )

    @pytest.mark.asyncio
    async def test_no_cache_preserves_existing_behavior(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """When cache is None, the service behaves identically to the uncached path."""
        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
            cache=None,
        )

        result = await service.search_images_async(collection="documents", query="sunset", limit=10)

        mock_embedding_provider.embed_text.assert_called_once()
        mock_image_vector_db_provider.search_async.assert_called_once()
        assert isinstance(result, SearchResults)

    @pytest.mark.asyncio
    async def test_cache_hit_skips_vector_db(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """On cache hit, vector DB search is NOT called and store is NOT called."""
        cache = self._make_mock_cache()
        cached = self._make_cached_results()
        cache.lookup = AsyncMock(return_value=cached)

        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
            cache=cache,
        )

        result = await service.search_images_async(collection="documents", query="sunset", limit=10)

        assert result is cached
        cache.lookup.assert_called_once()
        mock_image_vector_db_provider.search_async.assert_not_called()
        cache.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_queries_db_and_stores(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """On cache miss, vector DB is queried and results are stored in cache."""
        cache = self._make_mock_cache()
        cache.lookup = AsyncMock(return_value=None)

        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
            cache=cache,
        )

        result = await service.search_images_async(collection="documents", query="sunset", limit=10)

        cache.lookup.assert_called_once()
        mock_image_vector_db_provider.search_async.assert_called_once()
        cache.store.assert_called_once_with("documents", [0.1, 0.2, 0.3], "sunset", result, 10)

    @pytest.mark.asyncio
    async def test_cache_store_failure_is_nonfatal(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If cache.store raises, the search still returns DB results and a warning is logged."""
        cache = self._make_mock_cache()
        cache.lookup = AsyncMock(return_value=None)
        cache.store = AsyncMock(side_effect=RuntimeError("Redis down"))

        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
            cache=cache,
        )

        with caplog.at_level(logging.WARNING):
            result = await service.search_images_async(collection="documents", query="sunset", limit=10)

        assert isinstance(result, SearchResults)
        assert len(result.items) == 2
        mock_image_vector_db_provider.search_async.assert_called_once()
        assert any("Failed to store results in semantic cache" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_cache_lookup_passes_correct_args(
        self,
        mock_embedding_provider: MagicMock,
        mock_image_vector_db_provider: MagicMock,
    ) -> None:
        """Verify lookup receives the collection, query_vector, and limit."""
        cache = self._make_mock_cache()

        service = ImageSearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_image_vector_db_provider,
            cache=cache,
        )

        await service.search_images_async(collection="photos", query="  fluffy cat  ", limit=5)

        cache.lookup.assert_called_once_with("photos", [0.1, 0.2, 0.3], 5)
