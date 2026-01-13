"""Unit tests for core.services.search_service module.

This file tests the SearchService class which provides semantic search orchestration
by coordinating embedding generation and vector database queries.

# Test Coverage

The tests cover:
  - Service Initialization: Provider injection via attrs
  - Search Documents: Query validation, embedding generation, vector search
  - Async Search: Async operations, error handling
  - Input Validation: Empty queries, invalid limits
  - Error Handling: BadRequestError on validation, UpstreamError propagation
  - Integration: End-to-end search workflow

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
The embedding provider and vector DB provider are mocked to isolate service logic.

# Running Tests

Run with: pytest tests/unit/services/test_search_service.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import attrs.exceptions
import pytest

from core.exceptions import BadRequestError, UpstreamError
from core.models import SearchResultItem as SearchItem
from core.models import SearchResults
from core.services.search_service import SearchService

# =============================================================================
# Service Initialization Tests
# =============================================================================


class TestSearchServiceInit:
    """Test suite for SearchService initialization."""

    def test_creates_service_with_providers(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_db_provider: MagicMock,
    ) -> None:
        """Test that service is created with providers.

        **Why this test is important:**
          - Service requires both providers
          - Validates dependency injection
          - Critical for initialization
          - Validates attrs integration

        **What it tests:**
          - Service is created with embedding provider
          - Service is created with vector DB provider
          - Providers are accessible as attributes
        """
        service = SearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_vector_db_provider,
        )

        assert service.embedding_provider is mock_embedding_provider
        assert service.vector_db_provider is mock_vector_db_provider

    def test_service_is_frozen(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_db_provider: MagicMock,
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
        service = SearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_vector_db_provider,
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            service.embedding_provider = MagicMock()


# =============================================================================
# Search Documents Tests
# =============================================================================


class TestSearchServiceSearchDocuments:
    """Test suite for SearchService.search_documents method."""

    def test_search_success(self, search_service: SearchService) -> None:
        """Test that search_documents performs semantic search successfully.

        **Why this test is important:**
          - Search is the core functionality
          - Validates embedding + vector DB orchestration
          - Ensures proper result formatting
          - Critical for basic functionality

        **What it tests:**
          - Embedding provider embed is called with query
          - Vector DB provider search is called with embedding
          - Search results are returned correctly
        """
        result = search_service.search_documents(
            collection="test_collection",
            query="machine learning",
            limit=10,
        )

        # Verify embedding was generated
        search_service.embedding_provider.embed.assert_called_once_with("machine learning")

        # Verify vector DB was searched (note: asyncio.run was used)
        search_service.vector_db_provider.search_async.assert_called_once()
        call_kwargs = search_service.vector_db_provider.search_async.call_args[1]
        assert call_kwargs["collection"] == "test_collection"
        assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["limit"] == 10

        # Verify results
        assert isinstance(result, SearchResults)
        assert len(result.items) == 2
        assert result.total == 2
        assert result.items[0].score == 0.95
        assert result.items[1].score == 0.85

    def test_search_strips_whitespace(self, search_service: SearchService) -> None:
        """Test that search_documents strips whitespace from query.

        **Why this test is important:**
          - Whitespace can affect embeddings
          - Query normalization improves consistency
          - Critical for search quality
          - Validates input preprocessing

        **What it tests:**
          - Query is stripped before embedding generation
          - Leading/trailing whitespace is removed
        """
        search_service.search_documents(
            collection="test_collection",
            query="  machine learning  ",
            limit=10,
        )

        search_service.embedding_provider.embed.assert_called_once_with("machine learning")

    def test_search_raises_on_empty_query(self, search_service: SearchService) -> None:
        """Test that search_documents raises BadRequestError for empty query.

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
            search_service.search_documents(collection="test_collection", query="", limit=10)

    def test_search_raises_on_whitespace_only_query(self, search_service: SearchService) -> None:
        """Test that search_documents raises BadRequestError for whitespace-only query.

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
            search_service.search_documents(collection="test_collection", query="   ", limit=10)

    def test_search_raises_on_invalid_limit_too_small(self, search_service: SearchService) -> None:
        """Test that search_documents raises BadRequestError for limit < 1.

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
            search_service.search_documents(collection="test_collection", query="test", limit=0)

    def test_search_raises_on_invalid_limit_too_large(self, search_service: SearchService) -> None:
        """Test that search_documents raises BadRequestError for limit > 100.

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
            search_service.search_documents(collection="test_collection", query="test", limit=101)

    def test_search_accepts_valid_limit_range(self, search_service: SearchService) -> None:
        """Test that search_documents accepts valid limit values.

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
        search_service.search_documents(collection="test_collection", query="test", limit=1)

        # Upper boundary
        search_service.search_documents(collection="test_collection", query="test", limit=100)

        # Mid-range
        search_service.search_documents(collection="test_collection", query="test", limit=50)

    def test_search_propagates_embedding_error(self, search_service: SearchService) -> None:
        """Test that search_documents propagates embedding provider errors.

        **Why this test is important:**
          - Embedding errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from embedding provider is propagated
          - Error is not swallowed
        """
        search_service.embedding_provider.embed.side_effect = UpstreamError("Ollama connection failed")

        with pytest.raises(UpstreamError, match="Ollama connection failed"):
            search_service.search_documents(collection="test_collection", query="test", limit=10)

    def test_search_propagates_vector_db_error(self, search_service: SearchService) -> None:
        """Test that search_documents propagates vector DB provider errors.

        **Why this test is important:**
          - Vector DB errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from vector DB provider is propagated
          - Error is not swallowed
        """
        search_service.vector_db_provider.search_async.side_effect = UpstreamError("Qdrant connection failed")

        with pytest.raises(UpstreamError, match="Qdrant connection failed"):
            search_service.search_documents(collection="test_collection", query="test", limit=10)


# =============================================================================
# Async Search Documents Tests
# =============================================================================


class TestSearchServiceSearchDocumentsAsync:
    """Test suite for SearchService.search_documents_async method."""

    @pytest.mark.asyncio
    async def test_search_async_success(self, search_service: SearchService) -> None:
        """Test that search_documents_async performs semantic search successfully.

        **Why this test is important:**
          - Async search enables non-blocking operations
          - Validates async orchestration
          - Ensures proper result formatting
          - Critical for API performance

        **What it tests:**
          - Embedding provider embed_async is called with query
          - Vector DB provider search is called with embedding
          - Search results are returned correctly
        """
        result = await search_service.search_documents_async(
            collection="test_collection",
            query="machine learning",
            limit=10,
        )

        # Verify embedding was generated
        search_service.embedding_provider.embed_async.assert_called_once_with("machine learning")

        # Verify vector DB was searched
        search_service.vector_db_provider.search_async.assert_called_once()
        call_kwargs = search_service.vector_db_provider.search_async.call_args[1]
        assert call_kwargs["collection"] == "test_collection"
        assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3]
        assert call_kwargs["limit"] == 10

        # Verify results
        assert isinstance(result, SearchResults)
        assert len(result.items) == 2
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_search_async_strips_whitespace(self, search_service: SearchService) -> None:
        """Test that search_documents_async strips whitespace from query.

        **Why this test is important:**
          - Whitespace can affect embeddings
          - Query normalization improves consistency
          - Critical for search quality
          - Validates input preprocessing

        **What it tests:**
          - Query is stripped before embedding generation
          - Leading/trailing whitespace is removed
        """
        await search_service.search_documents_async(
            collection="test_collection",
            query="  machine learning  ",
            limit=10,
        )

        search_service.embedding_provider.embed_async.assert_called_once_with("machine learning")

    @pytest.mark.asyncio
    async def test_search_async_raises_on_empty_query(self, search_service: SearchService) -> None:
        """Test that search_documents_async raises BadRequestError for empty query.

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
            await search_service.search_documents_async(collection="test_collection", query="", limit=10)

    @pytest.mark.asyncio
    async def test_search_async_raises_on_invalid_limit(self, search_service: SearchService) -> None:
        """Test that search_documents_async validates limit parameter.

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
            await search_service.search_documents_async(collection="test_collection", query="test", limit=0)

        with pytest.raises(BadRequestError, match="Limit must be between 1 and 100"):
            await search_service.search_documents_async(collection="test_collection", query="test", limit=101)

    @pytest.mark.asyncio
    async def test_search_async_propagates_embedding_error(self, search_service: SearchService) -> None:
        """Test that search_documents_async propagates embedding provider errors.

        **Why this test is important:**
          - Embedding errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from embedding provider is propagated
          - Error is not swallowed
        """
        search_service.embedding_provider.embed_async.side_effect = UpstreamError("Ollama connection failed")

        with pytest.raises(UpstreamError, match="Ollama connection failed"):
            await search_service.search_documents_async(collection="test_collection", query="test", limit=10)

    @pytest.mark.asyncio
    async def test_search_async_propagates_vector_db_error(self, search_service: SearchService) -> None:
        """Test that search_documents_async propagates vector DB provider errors.

        **Why this test is important:**
          - Vector DB errors need to propagate
          - UpstreamError is expected error type
          - Critical for error handling
          - Validates error propagation

        **What it tests:**
          - UpstreamError from vector DB provider is propagated
          - Error is not swallowed
        """
        search_service.vector_db_provider.search_async.side_effect = UpstreamError("Qdrant connection failed")

        with pytest.raises(UpstreamError, match="Qdrant connection failed"):
            await search_service.search_documents_async(collection="test_collection", query="test", limit=10)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSearchServiceIntegration:
    """Test suite for end-to-end service integration."""

    def test_full_search_workflow(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_db_provider: MagicMock,
    ) -> None:
        """Test complete search workflow: validate -> embed -> search -> format.

        **Why this test is important:**
          - Validates end-to-end workflow
          - Ensures all steps work together
          - Critical for real-world usage
          - Validates integration

        **What it tests:**
          - Input validation passes for valid query
          - Embedding is generated correctly
          - Vector DB search is performed
          - Results are formatted correctly
        """
        # Setup mock responses
        mock_embedding_provider.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_vector_db_provider.search_async.return_value = SearchResults(
            items=[
                SearchItem(
                    point_id="doc1",
                    score=0.98,
                    payload={
                        "text": "Machine learning is awesome",
                        "author": "Alice",
                        "date": "2026-01-12",
                    },
                ),
                SearchItem(
                    point_id="doc2",
                    score=0.87,
                    payload={"text": "Deep learning basics", "author": "Bob", "date": "2026-01-11"},
                ),
                SearchItem(
                    point_id="doc3",
                    score=0.75,
                    payload={
                        "text": "Neural networks explained",
                        "author": "Charlie",
                        "date": "2026-01-10",
                    },
                ),
            ],
            total=3,
        )

        # Create service
        service = SearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_vector_db_provider,
        )

        # Perform search
        results = service.search_documents(
            collection="ml_docs",
            query="machine learning tutorial",
            limit=5,
        )

        # Verify embedding generation
        mock_embedding_provider.embed.assert_called_once_with("machine learning tutorial")

        # Verify vector DB search
        mock_vector_db_provider.search_async.assert_called_once()
        call_kwargs = mock_vector_db_provider.search_async.call_args[1]
        assert call_kwargs["collection"] == "ml_docs"
        assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert call_kwargs["limit"] == 5

        # Verify results
        assert results.total == 3
        assert len(results.items) == 3
        assert results.items[0].point_id == "doc1"
        assert results.items[0].score == 0.98
        assert results.items[0].payload["author"] == "Alice"
        assert results.items[1].point_id == "doc2"
        assert results.items[1].score == 0.87
        assert results.items[2].point_id == "doc3"
        assert results.items[2].score == 0.75

    @pytest.mark.asyncio
    async def test_full_search_workflow_async(
        self,
        mock_embedding_provider: MagicMock,
        mock_vector_db_provider: MagicMock,
    ) -> None:
        """Test complete async search workflow.

        **Why this test is important:**
          - Validates end-to-end async workflow
          - Ensures all async steps work together
          - Critical for API performance
          - Validates async integration

        **What it tests:**
          - Input validation passes for valid query
          - Async embedding is generated correctly
          - Vector DB search is performed
          - Results are formatted correctly
        """
        # Setup mock responses
        mock_embedding_provider.embed_async.return_value = [0.1, 0.2, 0.3]
        mock_vector_db_provider.search_async.return_value = SearchResults(
            items=[
                SearchItem(
                    point_id="doc1",
                    score=0.95,
                    payload={"text": "Test document", "source": "test.txt"},
                ),
            ],
            total=1,
        )

        # Create service
        service = SearchService(
            embedding_provider=mock_embedding_provider,
            vector_db_provider=mock_vector_db_provider,
        )

        # Perform async search
        results = await service.search_documents_async(
            collection="test_docs",
            query="test query",
            limit=10,
        )

        # Verify embedding generation
        mock_embedding_provider.embed_async.assert_called_once_with("test query")

        # Verify vector DB search
        mock_vector_db_provider.search_async.assert_called_once()

        # Verify results
        assert results.total == 1
        assert len(results.items) == 1
        assert results.items[0].point_id == "doc1"
