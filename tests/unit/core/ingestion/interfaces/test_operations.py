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
    ImageContentFetcher,
    S3ContentFetcher,
    VectorDBUpserter,
    detect_image_format,
)
from core.ingestion.interfaces.types import (
    BatchEmbeddingResult,
    ContentResult,
    UpsertResult,
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
# detect_image_format Tests
# =============================================================================


class TestDetectImageFormat:
    """Test suite for detect_image_format function."""

    def test_detects_jpeg_format(self) -> None:
        """Test JPEG detection from magic bytes.

        **Why this test is important:**
          - JPEG is most common image format
          - Must detect reliably from magic bytes

        **What it tests:**
          - Returns 'jpeg' for JPEG magic bytes
        """
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        result = detect_image_format(jpeg_bytes)

        assert result == "jpeg"

    def test_detects_png_format(self) -> None:
        """Test PNG detection from magic bytes.

        **Why this test is important:**
          - PNG widely used for lossless images
          - Must detect reliably

        **What it tests:**
          - Returns 'png' for PNG magic bytes
        """
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = detect_image_format(png_bytes)

        assert result == "png"

    def test_detects_gif87a_format(self) -> None:
        """Test GIF87a detection from magic bytes.

        **Why this test is important:**
          - GIF87a is older GIF version
          - Must support both GIF versions

        **What it tests:**
          - Returns 'gif' for GIF87a magic bytes
        """
        gif_bytes = b"GIF87a" + b"\x00" * 100

        result = detect_image_format(gif_bytes)

        assert result == "gif"

    def test_detects_gif89a_format(self) -> None:
        """Test GIF89a detection from magic bytes.

        **Why this test is important:**
          - GIF89a supports animation
          - Most common GIF version

        **What it tests:**
          - Returns 'gif' for GIF89a magic bytes
        """
        gif_bytes = b"GIF89a" + b"\x00" * 100

        result = detect_image_format(gif_bytes)

        assert result == "gif"

    def test_detects_webp_format(self) -> None:
        """Test WebP detection from magic bytes.

        **Why this test is important:**
          - WebP is modern efficient format
          - Requires RIFF + WEBP check

        **What it tests:**
          - Returns 'webp' for WebP magic bytes
        """
        # WebP: RIFF....WEBP where .... is file size
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100

        result = detect_image_format(webp_bytes)

        assert result == "webp"

    def test_returns_none_for_unknown_format(self) -> None:
        """Test that unknown formats return None.

        **Why this test is important:**
          - Unknown formats should be rejected
          - Returns None for validation

        **What it tests:**
          - Returns None for unknown magic bytes
        """
        unknown_bytes = b"UNKNOWN_FORMAT" + b"\x00" * 100

        result = detect_image_format(unknown_bytes)

        assert result is None

    def test_returns_none_for_too_short_data(self) -> None:
        """Test that short data returns None.

        **Why this test is important:**
          - Need minimum bytes for detection
          - Edge case handling

        **What it tests:**
          - Returns None if < 12 bytes
        """
        short_bytes = b"\xff\xd8\xff"  # Only 3 bytes

        result = detect_image_format(short_bytes)

        assert result is None

    def test_returns_none_for_empty_data(self) -> None:
        """Test that empty data returns None.

        **Why this test is important:**
          - Empty data is edge case
          - Should not raise

        **What it tests:**
          - Returns None for empty bytes
        """
        result = detect_image_format(b"")

        assert result is None


# =============================================================================
# ImageContentFetcher Tests
# =============================================================================


class TestImageContentFetcher:
    """Test suite for ImageContentFetcher."""

    # Sample magic bytes for different formats
    JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
    PNG_HEADER = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    GIF_HEADER = b"GIF89a\x00\x00\x00\x00\x00\x00"
    WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBP\x00\x00"

    def test_creates_fetcher(self) -> None:
        """Test fetcher initialization.

        **Why this test is important:**
          - ImageContentFetcher is core image operation
          - Validates constructor with all params

        **What it tests:**
          - Fetcher stores client, bucket, and size limits
        """
        mock_s3 = MagicMock()
        fetcher = ImageContentFetcher(
            mock_s3,
            bucket="pipeline",
            min_size_bytes=50,
            max_size_bytes=10_000_000,
        )

        assert fetcher.s3 == mock_s3
        assert fetcher.bucket == "pipeline"
        assert fetcher.min_size_bytes == 50
        assert fetcher.max_size_bytes == 10_000_000

    def test_creates_fetcher_with_defaults(self) -> None:
        """Test fetcher initialization with default size limits.

        **Why this test is important:**
          - Defaults should be sensible
          - Validates default values

        **What it tests:**
          - Default min/max sizes are set
        """
        mock_s3 = MagicMock()
        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")

        assert fetcher.min_size_bytes == 100
        assert fetcher.max_size_bytes == 50 * 1024 * 1024

    def test_fetch_one_success_jpeg(self) -> None:
        """Test successful JPEG image fetch.

        **Why this test is important:**
          - Core S3 image operation
          - JPEG is most common format

        **What it tests:**
          - fetch_one returns ImageContentResult for JPEG
        """
        mock_s3 = MagicMock()
        jpeg_data = self.JPEG_HEADER + b"\x00" * 100
        mock_s3.get_object.return_value = jpeg_data

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("images/photo.jpg")

        assert result is not None
        assert result.s3_key == "images/photo.jpg"
        assert result.format == "jpeg"
        assert result.size_bytes == len(jpeg_data)
        assert result.image_bytes == jpeg_data

    def test_fetch_one_success_png(self) -> None:
        """Test successful PNG image fetch.

        **Why this test is important:**
          - PNG widely used for lossless
          - Validates PNG detection

        **What it tests:**
          - fetch_one returns ImageContentResult for PNG
        """
        mock_s3 = MagicMock()
        png_data = self.PNG_HEADER + b"\x00" * 100
        mock_s3.get_object.return_value = png_data

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("images/photo.png")

        assert result is not None
        assert result.format == "png"

    def test_fetch_one_success_gif(self) -> None:
        """Test successful GIF image fetch.

        **Why this test is important:**
          - GIF used for animations
          - Validates GIF detection

        **What it tests:**
          - fetch_one returns ImageContentResult for GIF
        """
        mock_s3 = MagicMock()
        gif_data = self.GIF_HEADER + b"\x00" * 100
        mock_s3.get_object.return_value = gif_data

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("images/photo.gif")

        assert result is not None
        assert result.format == "gif"

    def test_fetch_one_success_webp(self) -> None:
        """Test successful WebP image fetch.

        **Why this test is important:**
          - WebP is modern efficient format
          - Validates WebP detection

        **What it tests:**
          - fetch_one returns ImageContentResult for WebP
        """
        mock_s3 = MagicMock()
        webp_data = self.WEBP_HEADER + b"\x00" * 100
        mock_s3.get_object.return_value = webp_data

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("images/photo.webp")

        assert result is not None
        assert result.format == "webp"

    def test_fetch_one_returns_none_on_client_error(self) -> None:
        """Test that fetch returns None on ClientError.

        **Why this test is important:**
          - Error handling is critical
          - Graceful degradation

        **What it tests:**
          - None returned on S3 error
        """
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("missing.jpg")

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

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("images/photo.jpg")

        assert result is None

    def test_fetch_one_returns_none_for_too_small(self) -> None:
        """Test that fetch returns None for images below min size.

        **Why this test is important:**
          - Tiny images often corrupt
          - Size validation critical

        **What it tests:**
          - None returned for images < min_size_bytes
        """
        mock_s3 = MagicMock()
        # Very small "image" that's below default min of 100 bytes
        mock_s3.get_object.return_value = self.JPEG_HEADER  # Only ~12 bytes

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("images/tiny.jpg")

        assert result is None

    def test_fetch_one_returns_none_for_too_large(self) -> None:
        """Test that fetch returns None for images above max size.

        **Why this test is important:**
          - Large images can cause OOM
          - Size limits protect pipeline

        **What it tests:**
          - None returned for images > max_size_bytes
        """
        mock_s3 = MagicMock()
        # Create image data larger than custom max
        large_data = self.JPEG_HEADER + b"\x00" * 2000
        mock_s3.get_object.return_value = large_data

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline", max_size_bytes=1000)
        result = fetcher.fetch_one("images/huge.jpg")

        assert result is None

    def test_fetch_one_returns_none_for_unknown_format(self) -> None:
        """Test that fetch returns None for unknown image formats.

        **Why this test is important:**
          - Only support known formats
          - Unknown formats rejected

        **What it tests:**
          - None returned for unsupported format
        """
        mock_s3 = MagicMock()
        # Random data that isn't a recognized image format
        mock_s3.get_object.return_value = b"UNKNOWN_FORMAT" + b"\x00" * 200

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("images/mystery.xyz")

        assert result is None

    def test_fetch_all_success(self) -> None:
        """Test fetching all images successfully.

        **Why this test is important:**
          - Batch fetching common
          - Validates tuple return

        **What it tests:**
          - All images returned, no failures
        """
        mock_s3 = MagicMock()
        jpeg_data = self.JPEG_HEADER + b"\x00" * 100
        mock_s3.get_object.return_value = jpeg_data

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        images, failures = fetcher.fetch_all(["img1.jpg", "img2.jpg"])

        assert len(images) == 2
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
        jpeg_data = self.JPEG_HEADER + b"\x00" * 100

        def side_effect(bucket, key):
            if key == "img1.jpg":
                return jpeg_data
            else:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        mock_s3.get_object.side_effect = side_effect

        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")
        images, failures = fetcher.fetch_all(["img1.jpg", "img2.jpg"])

        assert len(images) == 1
        assert len(failures) == 1
        assert images[0].s3_key == "img1.jpg"
        assert failures[0].s3_key == "img2.jpg"
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
        fetcher = ImageContentFetcher(mock_s3, bucket="pipeline")

        images, failures = fetcher.fetch_all([])

        assert len(images) == 0
        assert len(failures) == 0

    def test_filter_image_keys_filters_correctly(self) -> None:
        """Test that filter_image_keys filters correctly.

        **Why this test is important:**
          - Need to filter S3 keys by extension
          - Only process image files

        **What it tests:**
          - Only supported extensions returned
        """
        keys = [
            "images/photo.jpg",
            "images/image.jpeg",
            "images/logo.png",
            "images/animation.gif",
            "images/modern.webp",
            "docs/readme.txt",
            "docs/data.json",
            "images/config.yaml",
        ]

        result = ImageContentFetcher.filter_image_keys(keys)

        assert len(result) == 5
        assert "images/photo.jpg" in result
        assert "images/image.jpeg" in result
        assert "images/logo.png" in result
        assert "images/animation.gif" in result
        assert "images/modern.webp" in result
        assert "docs/readme.txt" not in result
        assert "docs/data.json" not in result

    def test_filter_image_keys_case_insensitive(self) -> None:
        """Test that filter_image_keys is case insensitive.

        **Why this test is important:**
          - Extensions may be uppercase
          - Must handle both cases

        **What it tests:**
          - Uppercase extensions filtered correctly
        """
        keys = [
            "photo.JPG",
            "image.PNG",
            "ANIMATION.GIF",
        ]

        result = ImageContentFetcher.filter_image_keys(keys)

        assert len(result) == 3

    def test_filter_image_keys_empty_list(self) -> None:
        """Test filter_image_keys with empty list.

        **Why this test is important:**
          - Edge case handling
          - Should not raise

        **What it tests:**
          - Empty list returned for empty input
        """
        result = ImageContentFetcher.filter_image_keys([])

        assert result == []

    def test_filter_image_keys_no_extension(self) -> None:
        """Test filter_image_keys with files without extensions.

        **Why this test is important:**
          - Some files have no extension
          - Should be excluded

        **What it tests:**
          - Files without extensions filtered out
        """
        keys = ["photo", "image", "Makefile"]

        result = ImageContentFetcher.filter_image_keys(keys)

        assert result == []


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

        assert isinstance(result, UpsertResult)
        assert result.all_success
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

        assert isinstance(result, UpsertResult)
        assert result.all_success  # Empty batch is considered successful
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

        assert isinstance(result, UpsertResult)
        assert result.any_success  # Qdrant succeeded
        assert result.qdrant_success
        assert not result.weaviate_success

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

        assert isinstance(result, UpsertResult)
        assert not result.any_success
        assert not result.qdrant_success
        assert not result.weaviate_success


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
        mock_upserter.upsert_batch_async = AsyncMock(
            return_value=UpsertResult(qdrant_success=True, weaviate_success=True, batch_size=1)
        )

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
        mock_upserter.upsert_batch_async = AsyncMock(
            return_value=UpsertResult(qdrant_success=False, weaviate_success=False, batch_size=1)
        )

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
        mock_upserter.upsert_batch_async = AsyncMock(
            return_value=UpsertResult(qdrant_success=True, weaviate_success=True, batch_size=1)
        )

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
