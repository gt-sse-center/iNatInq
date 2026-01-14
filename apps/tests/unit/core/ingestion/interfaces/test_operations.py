"""Unit tests for core.ingestion.interfaces.operations module.

This file tests the operation classes that encapsulate processing logic
for S3 fetching, embedding generation, and vector DB upserts.

# Test Coverage

The tests cover:
  - S3ContentFetcher: Single/batch fetching, error handling
  - EmbeddingGenerator: Single/batch generation, rate limiting
  - VectorDBUpserter: Parallel upserts, partial failures
  - BatchProcessor: Complete batch processing, dynamic sizing

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.

# Running Tests

Run with: pytest tests/unit/core/ingestion/interfaces/test_operations.py
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError
from core.exceptions import UpstreamError
from core.ingestion.interfaces.operations import (
    BatchProcessor,
    EmbeddingGenerator,
    S3ContentFetcher,
    VectorDBUpserter,
)
from core.ingestion.interfaces.types import (
    BatchEmbeddingResult,
    ContentResult,
)
from core.models import VectorPoint

# =============================================================================
# S3ContentFetcher Tests
# =============================================================================


class TestS3ContentFetcher:
    """Test suite for S3ContentFetcher."""

    def test_creates_fetcher(self) -> None:
        """Test fetcher initialization.

        **Why this test is important:**
          - Fetcher is core operation
          - Validates constructor

        **What it tests:**
          - Fetcher stores client and bucket
        """
        mock_s3 = MagicMock()
        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")

        assert fetcher.s3 == mock_s3
        assert fetcher.bucket == "pipeline"

    def test_fetch_one_success(self) -> None:
        """Test successful single object fetch.

        **Why this test is important:**
          - Core S3 operation
          - Validates return type

        **What it tests:**
          - fetch_one returns ContentResult on success
        """
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = b"Hello world"

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("doc.txt")

        assert result is not None
        assert result.s3_key == "doc.txt"
        assert result.content == "Hello world"

    def test_fetch_one_with_unicode(self) -> None:
        """Test fetching unicode content.

        **Why this test is important:**
          - International text support
          - UTF-8 decoding

        **What it tests:**
          - Unicode content decoded correctly
        """
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = "Hello 世界".encode()

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("doc.txt")

        assert result is not None
        assert result.content == "Hello 世界"

    def test_fetch_one_returns_none_on_client_error(self) -> None:
        """Test that fetch returns None on ClientError.

        **Why this test is important:**
          - Error handling is critical
          - Graceful degradation

        **What it tests:**
          - None returned on error
        """
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("missing.txt")

        assert result is None

    def test_fetch_one_returns_none_on_upstream_error(self) -> None:
        """Test that fetch returns None on UpstreamError.

        **Why this test is important:**
          - Pipeline error handling
          - Validates error type

        **What it tests:**
          - None returned on UpstreamError
        """
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = UpstreamError("S3 unavailable")

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("doc.txt")

        assert result is None

    def test_fetch_all_success(self) -> None:
        """Test fetching all objects successfully.

        **Why this test is important:**
          - Batch fetching common
          - Validates tuple return

        **What it tests:**
          - All contents returned, no failures
        """
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = b"content"

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        contents, failures = fetcher.fetch_all(["doc1.txt", "doc2.txt"])

        assert len(contents) == 2
        assert len(failures) == 0

    def test_fetch_all_mixed_results(self) -> None:
        """Test fetching with mixed success/failure.

        **Why this test is important:**
          - Partial failures happen
          - Must track both

        **What it tests:**
          - Successful and failed separated correctly
        """
        mock_s3 = MagicMock()

        def side_effect(bucket, key):
            if key == "doc1.txt":
                return b"content"
            else:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        mock_s3.get_object.side_effect = side_effect

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        contents, failures = fetcher.fetch_all(["doc1.txt", "doc2.txt"])

        assert len(contents) == 1
        assert len(failures) == 1
        assert contents[0].s3_key == "doc1.txt"
        assert failures[0].s3_key == "doc2.txt"
        assert failures[0].success is False

    def test_fetch_all_empty_list(self) -> None:
        """Test fetching empty list.

        **Why this test is important:**
          - Edge case handling
          - No errors on empty

        **What it tests:**
          - Empty lists returned
        """
        mock_s3 = MagicMock()
        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")

        contents, failures = fetcher.fetch_all([])

        assert len(contents) == 0
        assert len(failures) == 0


# =============================================================================
# EmbeddingGenerator Tests
# =============================================================================


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""

    def test_creates_generator(self) -> None:
        """Test generator initialization.

        **Why this test is important:**
          - Generator is core operation
          - Validates constructor

        **What it tests:**
          - Generator stores embedder
        """
        mock_embedder = MagicMock()
        generator = EmbeddingGenerator(mock_embedder)

        assert generator.embedder == mock_embedder

    def test_vector_size_property(self) -> None:
        """Test vector_size property.

        **Why this test is important:**
          - Vector size needed for DB operations
          - Delegates to embedder

        **What it tests:**
          - Returns embedder's vector_size
        """
        mock_embedder = MagicMock()
        mock_embedder.vector_size = 768

        generator = EmbeddingGenerator(mock_embedder)

        assert generator.vector_size == 768

    @pytest.mark.asyncio
    async def test_generate_one_async_success(self) -> None:
        """Test successful single embedding.

        **Why this test is important:**
          - Core embedding operation
          - Validates async behavior

        **What it tests:**
          - Returns vector on success
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_async = AsyncMock(return_value=[0.1, 0.2, 0.3])

        generator = EmbeddingGenerator(mock_embedder)
        result = await generator.generate_one_async("Hello world")

        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_one_async_failure(self) -> None:
        """Test embedding failure handling.

        **Why this test is important:**
          - Error handling critical
          - Returns None on failure

        **What it tests:**
          - None returned on error
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_async = AsyncMock(side_effect=UpstreamError("Service unavailable"))

        generator = EmbeddingGenerator(mock_embedder)
        result = await generator.generate_one_async("Hello")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_batch_async_success(self) -> None:
        """Test successful batch embedding.

        **Why this test is important:**
          - Batch embedding common
          - Validates multiple vectors

        **What it tests:**
          - Returns list of vectors
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_async = AsyncMock(side_effect=[[0.1, 0.2], [0.3, 0.4]])

        generator = EmbeddingGenerator(mock_embedder)
        batch = [
            ContentResult("doc1.txt", "Hello"),
            ContentResult("doc2.txt", "World"),
        ]
        result = await generator.generate_batch_async(batch)

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_generate_batch_async_empty(self) -> None:
        """Test batch with empty list.

        **Why this test is important:**
          - Edge case handling
          - Should return empty list

        **What it tests:**
          - Empty list returned for empty input
        """
        mock_embedder = MagicMock()
        generator = EmbeddingGenerator(mock_embedder)

        result = await generator.generate_batch_async([])

        assert result == []

    @pytest.mark.asyncio
    async def test_generate_batch_async_with_rate_limiter(self) -> None:
        """Test batch with rate limiting.

        **Why this test is important:**
          - Rate limiting prevents overload
          - Validates limiter called

        **What it tests:**
          - Rate limiter acquire called
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_async = AsyncMock(return_value=[0.1, 0.2])

        mock_limiter = MagicMock()
        mock_limiter.acquire = AsyncMock()

        generator = EmbeddingGenerator(mock_embedder, rate_limiter=mock_limiter)
        batch = [ContentResult("doc.txt", "Hello")]
        await generator.generate_batch_async(batch)

        mock_limiter.acquire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_batch_async_with_semaphore(self) -> None:
        """Test batch with concurrency semaphore.

        **Why this test is important:**
          - Concurrency control needed
          - Validates semaphore used

        **What it tests:**
          - Processing completes with semaphore
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_async = AsyncMock(return_value=[0.1, 0.2])

        generator = EmbeddingGenerator(mock_embedder)
        semaphore = asyncio.Semaphore(1)
        batch = [ContentResult("doc.txt", "Hello")]

        result = await generator.generate_batch_async(batch, semaphore)

        assert result == [[0.1, 0.2]]


# =============================================================================
# VectorDBUpserter Tests
# =============================================================================


class TestVectorDBUpserter:
    """Test suite for VectorDBUpserter."""

    def test_creates_upserter(self) -> None:
        """Test upserter initialization.

        **Why this test is important:**
          - Upserter is core operation
          - Validates constructor

        **What it tests:**
          - Upserter stores both DBs
        """
        mock_qdrant = MagicMock()
        mock_weaviate = MagicMock()

        upserter = VectorDBUpserter(mock_qdrant, mock_weaviate)

        assert upserter.qdrant_db == mock_qdrant
        assert upserter.weaviate_db == mock_weaviate

    @pytest.mark.asyncio
    async def test_upsert_batch_async_success(self) -> None:
        """Test successful batch upsert.

        **Why this test is important:**
          - Core upsert operation
          - Validates both DBs called

        **What it tests:**
          - Returns True on success
        """
        mock_qdrant = MagicMock()
        mock_qdrant.batch_upsert_async = AsyncMock(return_value=None)
        mock_weaviate = MagicMock()
        mock_weaviate.batch_upsert_async = AsyncMock(return_value=None)

        upserter = VectorDBUpserter(mock_qdrant, mock_weaviate)

        mock_point = MagicMock(spec=VectorPoint)
        mock_point.to_qdrant.return_value = MagicMock()
        batch = BatchEmbeddingResult(
            qdrant_points=[mock_point],
            weaviate_objects=[MagicMock()],
        )

        result = await upserter.upsert_batch_async(batch, "documents", 768)

        assert result is True
        mock_qdrant.batch_upsert_async.assert_awaited_once()
        mock_weaviate.batch_upsert_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_batch_async_empty(self) -> None:
        """Test upsert with empty batch.

        **Why this test is important:**
          - Edge case handling
          - Should succeed without calling DBs

        **What it tests:**
          - Returns True for empty batch
        """
        mock_qdrant = MagicMock()
        mock_weaviate = MagicMock()

        upserter = VectorDBUpserter(mock_qdrant, mock_weaviate)
        batch = BatchEmbeddingResult(qdrant_points=[], weaviate_objects=[])

        result = await upserter.upsert_batch_async(batch, "documents", 768)

        assert result is True
        mock_qdrant.batch_upsert_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_batch_async_partial_failure(self) -> None:
        """Test partial failure handling.

        **Why this test is important:**
          - One DB may fail
          - Should return True if any succeeds

        **What it tests:**
          - Returns True when one DB succeeds
        """
        mock_qdrant = MagicMock()
        mock_qdrant.batch_upsert_async = AsyncMock(return_value=None)
        mock_weaviate = MagicMock()
        mock_weaviate.batch_upsert_async = AsyncMock(side_effect=Exception("DB error"))

        upserter = VectorDBUpserter(mock_qdrant, mock_weaviate)

        mock_point = MagicMock(spec=VectorPoint)
        mock_point.to_qdrant.return_value = MagicMock()
        batch = BatchEmbeddingResult(
            qdrant_points=[mock_point],
            weaviate_objects=[MagicMock()],
        )

        result = await upserter.upsert_batch_async(batch, "documents", 768)

        assert result is True  # Qdrant succeeded

    @pytest.mark.asyncio
    async def test_upsert_batch_async_both_fail(self) -> None:
        """Test both DBs failing.

        **Why this test is important:**
          - Complete failure case
          - Should return False

        **What it tests:**
          - Returns False when both fail
        """
        mock_qdrant = MagicMock()
        mock_qdrant.batch_upsert_async = AsyncMock(side_effect=Exception("Qdrant error"))
        mock_weaviate = MagicMock()
        mock_weaviate.batch_upsert_async = AsyncMock(side_effect=Exception("Weaviate error"))

        upserter = VectorDBUpserter(mock_qdrant, mock_weaviate)

        mock_point = MagicMock(spec=VectorPoint)
        mock_point.to_qdrant.return_value = MagicMock()
        batch = BatchEmbeddingResult(
            qdrant_points=[mock_point],
            weaviate_objects=[MagicMock()],
        )

        result = await upserter.upsert_batch_async(batch, "documents", 768)

        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_qdrant_only_async_success(self) -> None:
        """Test Qdrant-only upsert.

        **Why this test is important:**
          - Sometimes only one DB needed
          - Validates single-DB path

        **What it tests:**
          - Returns True on success
        """
        mock_qdrant = MagicMock()
        mock_qdrant.batch_upsert_async = AsyncMock(return_value=None)
        mock_weaviate = MagicMock()

        upserter = VectorDBUpserter(mock_qdrant, mock_weaviate)

        mock_point = MagicMock(spec=VectorPoint)
        mock_point.to_qdrant.return_value = MagicMock()
        batch = BatchEmbeddingResult(
            qdrant_points=[mock_point],
            weaviate_objects=[],
        )

        result = await upserter.upsert_qdrant_only_async(batch, "documents", 768)

        assert result is True
        mock_weaviate.batch_upsert_async.assert_not_called()


# =============================================================================
# BatchProcessor Tests
# =============================================================================


class TestBatchProcessor:
    """Test suite for BatchProcessor."""

    def test_creates_processor(self) -> None:
        """Test processor initialization.

        **Why this test is important:**
          - Processor orchestrates operations
          - Validates constructor

        **What it tests:**
          - Processor stores all components
        """
        mock_generator = MagicMock()
        mock_factory = MagicMock()
        mock_upserter = MagicMock()

        processor = BatchProcessor(
            embedding_generator=mock_generator,
            point_factory=mock_factory,
            upserter=mock_upserter,
            collection="documents",
        )

        assert processor.embedding_generator == mock_generator
        assert processor.collection == "documents"

    @pytest.mark.asyncio
    async def test_process_batch_async_success(self) -> None:
        """Test successful batch processing.

        **Why this test is important:**
          - Core processing flow
          - Validates all steps

        **What it tests:**
          - Returns success results
          - Batch size grows
        """
        mock_generator = MagicMock()
        mock_generator.generate_batch_async = AsyncMock(return_value=[[0.1, 0.2]])
        mock_generator.vector_size = 768

        mock_factory = MagicMock()
        mock_batch = BatchEmbeddingResult(
            qdrant_points=[MagicMock()],
            weaviate_objects=[MagicMock()],
        )
        mock_factory.create_batch.return_value = mock_batch

        mock_upserter = MagicMock()
        mock_upserter.upsert_batch_async = AsyncMock(return_value=True)

        processor = BatchProcessor(
            embedding_generator=mock_generator,
            point_factory=mock_factory,
            upserter=mock_upserter,
            collection="documents",
        )

        batch = [ContentResult("doc.txt", "Hello")]
        results, new_size = await processor.process_batch_async(
            batch, None, current_batch_size=4, min_batch_size=1, max_batch_size=8
        )

        assert len(results) == 1
        assert results[0].success is True
        assert new_size == 5  # Grew by 1

    @pytest.mark.asyncio
    async def test_process_batch_async_empty(self) -> None:
        """Test empty batch processing.

        **Why this test is important:**
          - Edge case handling
          - Should return empty

        **What it tests:**
          - Empty list returned
        """
        processor = BatchProcessor(
            embedding_generator=MagicMock(),
            point_factory=MagicMock(),
            upserter=MagicMock(),
            collection="documents",
        )

        results, new_size = await processor.process_batch_async(
            [], None, current_batch_size=4, min_batch_size=1, max_batch_size=8
        )

        assert len(results) == 0
        assert new_size == 4

    @pytest.mark.asyncio
    async def test_process_batch_async_embedding_failure(self) -> None:
        """Test batch with embedding failure.

        **Why this test is important:**
          - Error handling critical
          - Batch size should shrink

        **What it tests:**
          - Failure results returned
          - Batch size shrinks
        """
        mock_generator = MagicMock()
        mock_generator.generate_batch_async = AsyncMock(return_value=None)

        processor = BatchProcessor(
            embedding_generator=mock_generator,
            point_factory=MagicMock(),
            upserter=MagicMock(),
            collection="documents",
        )

        batch = [ContentResult("doc.txt", "Hello")]
        results, new_size = await processor.process_batch_async(
            batch, None, current_batch_size=4, min_batch_size=1, max_batch_size=8
        )

        assert len(results) == 1
        assert results[0].success is False
        assert new_size == 2  # Halved

    @pytest.mark.asyncio
    async def test_process_batch_async_upsert_failure(self) -> None:
        """Test batch with upsert failure.

        **Why this test is important:**
          - Upsert failures happen
          - Batch size should shrink

        **What it tests:**
          - Failure results returned
          - Batch size shrinks
        """
        mock_generator = MagicMock()
        mock_generator.generate_batch_async = AsyncMock(return_value=[[0.1, 0.2]])
        mock_generator.vector_size = 768

        mock_factory = MagicMock()
        mock_factory.create_batch.return_value = BatchEmbeddingResult(
            qdrant_points=[MagicMock()],
            weaviate_objects=[MagicMock()],
        )

        mock_upserter = MagicMock()
        mock_upserter.upsert_batch_async = AsyncMock(return_value=False)

        processor = BatchProcessor(
            embedding_generator=mock_generator,
            point_factory=mock_factory,
            upserter=mock_upserter,
            collection="documents",
        )

        batch = [ContentResult("doc.txt", "Hello")]
        results, new_size = await processor.process_batch_async(
            batch, None, current_batch_size=4, min_batch_size=1, max_batch_size=8
        )

        assert len(results) == 1
        assert results[0].success is False
        assert new_size == 2  # Halved

    @pytest.mark.asyncio
    async def test_process_batch_async_respects_min_batch_size(self) -> None:
        """Test that batch size doesn't go below minimum.

        **Why this test is important:**
          - Min batch size prevents zero
          - Validates floor

        **What it tests:**
          - Batch size stays at minimum
        """
        mock_generator = MagicMock()
        mock_generator.generate_batch_async = AsyncMock(return_value=None)

        processor = BatchProcessor(
            embedding_generator=mock_generator,
            point_factory=MagicMock(),
            upserter=MagicMock(),
            collection="documents",
        )

        batch = [ContentResult("doc.txt", "Hello")]
        _results, new_size = await processor.process_batch_async(
            batch, None, current_batch_size=2, min_batch_size=2, max_batch_size=8
        )

        assert new_size == 2  # Stays at min

    @pytest.mark.asyncio
    async def test_process_batch_async_respects_max_batch_size(self) -> None:
        """Test that batch size doesn't exceed maximum.

        **Why this test is important:**
          - Max batch size prevents OOM
          - Validates ceiling

        **What it tests:**
          - Batch size stays at maximum
        """
        mock_generator = MagicMock()
        mock_generator.generate_batch_async = AsyncMock(return_value=[[0.1, 0.2]])
        mock_generator.vector_size = 768

        mock_factory = MagicMock()
        mock_factory.create_batch.return_value = BatchEmbeddingResult(
            qdrant_points=[MagicMock()],
            weaviate_objects=[MagicMock()],
        )

        mock_upserter = MagicMock()
        mock_upserter.upsert_batch_async = AsyncMock(return_value=True)

        processor = BatchProcessor(
            embedding_generator=mock_generator,
            point_factory=mock_factory,
            upserter=mock_upserter,
            collection="documents",
        )

        batch = [ContentResult("doc.txt", "Hello")]
        _results, new_size = await processor.process_batch_async(
            batch, None, current_batch_size=8, min_batch_size=1, max_batch_size=8
        )

        assert new_size == 8  # Stays at max

