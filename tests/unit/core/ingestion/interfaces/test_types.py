"""Unit tests for core.ingestion.interfaces.types module.

This file tests the shared type classes used by both Ray and Spark
ingestion pipelines.

# Test Coverage

The tests cover:
  - ProcessingResult: Success/failure creation, tuple conversion, immutability
  - ContentResult: Content storage, immutability
  - BatchEmbeddingResult: Batch operations, length, empty checks
  - ProcessingConfig: Configuration, defaults, immutability
  - RateLimitConfig: Rate limit settings
  - ProcessingClients: Client bundle, close methods

# Test Structure

Tests use pytest class-based organization with direct model instantiation.
Uses mocking for client dependencies.

# Running Tests

Run with: pytest tests/unit/core/ingestion/interfaces/test_types.py
"""

from unittest.mock import AsyncMock, MagicMock

import attrs.exceptions
import pytest
from config import EmbeddingConfig
from core.ingestion.interfaces.types import (
    BatchEmbeddingResult,
    ContentResult,
    ImageContentResult,
    ProcessingClients,
    ProcessingConfig,
    ProcessingResult,
    RateLimitConfig,
)
from core.models import VectorPoint

# =============================================================================
# ProcessingResult Tests
# =============================================================================


class TestProcessingResult:
    """Test suite for ProcessingResult type class."""

    def test_creates_result_with_required_fields(self) -> None:
        """Test that ProcessingResult is created with required fields.

        **Why this test is important:**
          - ProcessingResult tracks individual object status
          - Validates attrs integration
          - Ensures required fields are enforced

        **What it tests:**
          - Result created with s3_key and success
          - error_message defaults to empty string
        """
        result = ProcessingResult(s3_key="inputs/doc.txt", success=True)

        assert result.s3_key == "inputs/doc.txt"
        assert result.success is True
        assert result.error_message == ""

    def test_creates_result_with_error_message(self) -> None:
        """Test that ProcessingResult accepts error_message.

        **Why this test is important:**
          - Error tracking is critical for debugging
          - Validates optional field handling

        **What it tests:**
          - Result stores error_message correctly
        """
        result = ProcessingResult(
            s3_key="inputs/doc.txt",
            success=False,
            error_message="S3 fetch failed",
        )

        assert result.success is False
        assert result.error_message == "S3 fetch failed"

    def test_success_result_factory(self) -> None:
        """Test the success_result factory method.

        **Why this test is important:**
          - Factory methods simplify common patterns
          - Reduces boilerplate code

        **What it tests:**
          - success_result creates correct object
        """
        result = ProcessingResult.success_result("doc.txt")

        assert result.s3_key == "doc.txt"
        assert result.success is True
        assert result.error_message == ""

    def test_failure_result_factory(self) -> None:
        """Test the failure_result factory method.

        **Why this test is important:**
          - Factory methods simplify error handling
          - Ensures consistent error creation

        **What it tests:**
          - failure_result creates correct object
        """
        result = ProcessingResult.failure_result("doc.txt", "Connection timeout")

        assert result.s3_key == "doc.txt"
        assert result.success is False
        assert result.error_message == "Connection timeout"

    def test_to_tuple_format(self) -> None:
        """Test that to_tuple returns correct format.

        **Why this test is important:**
          - Ray/Spark require tuple format for compatibility
          - Critical for distributed processing

        **What it tests:**
          - to_tuple returns (key, success, error)
        """
        result = ProcessingResult(s3_key="doc.txt", success=True, error_message="")

        assert result.to_tuple() == ("doc.txt", True, "")

    def test_to_tuple_with_error(self) -> None:
        """Test to_tuple with error message.

        **Why this test is important:**
          - Error messages must be preserved in tuple

        **What it tests:**
          - Error message included in tuple
        """
        result = ProcessingResult(s3_key="doc.txt", success=False, error_message="Failed")

        assert result.to_tuple() == ("doc.txt", False, "Failed")

    def test_result_is_immutable(self) -> None:
        """Test that ProcessingResult is immutable.

        **Why this test is important:**
          - Immutability ensures thread safety
          - Prevents accidental modification

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        result = ProcessingResult(s3_key="doc.txt", success=True)

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            result.success = False


# =============================================================================
# ContentResult Tests
# =============================================================================


class TestContentResult:
    """Test suite for ContentResult type class."""

    def test_creates_content_result(self) -> None:
        """Test that ContentResult is created correctly.

        **Why this test is important:**
          - ContentResult holds downloaded S3 content
          - Simple but critical type

        **What it tests:**
          - ContentResult stores key and content
        """
        content = ContentResult(s3_key="inputs/doc.txt", content="Hello world")

        assert content.s3_key == "inputs/doc.txt"
        assert content.content == "Hello world"

    def test_content_with_unicode(self) -> None:
        """Test that ContentResult handles unicode.

        **Why this test is important:**
          - Documents may contain unicode
          - Must handle international text

        **What it tests:**
          - Unicode content stored correctly
        """
        content = ContentResult(s3_key="doc.txt", content="Hello 世界 🌍")

        assert content.content == "Hello 世界 🌍"

    def test_content_with_multiline(self) -> None:
        """Test that ContentResult handles multiline text.

        **Why this test is important:**
          - Documents are typically multiline
          - Newlines must be preserved

        **What it tests:**
          - Multiline content preserved
        """
        text = "Line 1\nLine 2\nLine 3"
        content = ContentResult(s3_key="doc.txt", content=text)

        assert content.content == text
        assert content.content.count("\n") == 2

    def test_content_is_immutable(self) -> None:
        """Test that ContentResult is immutable.

        **Why this test is important:**
          - Immutability for thread safety

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        content = ContentResult(s3_key="doc.txt", content="test")

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            content.content = "modified"


# =============================================================================
# ImageContentResult Tests
# =============================================================================


class TestImageContentResult:
    """Test suite for ImageContentResult type class."""

    def test_creates_image_result_with_required_fields(self) -> None:
        """Test that ImageContentResult is created with required fields.

        **Why this test is important:**
          - ImageContentResult holds downloaded image data
          - Validates all required fields are stored

        **What it tests:**
          - Result created with s3_key, image_bytes, format, size_bytes
          - Optional width/height default to None
        """
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = ImageContentResult(
            s3_key="images/photo.png",
            image_bytes=image_bytes,
            format="png",
            size_bytes=len(image_bytes),
        )

        assert result.s3_key == "images/photo.png"
        assert result.image_bytes == image_bytes
        assert result.format == "png"
        assert result.size_bytes == len(image_bytes)
        assert result.width is None
        assert result.height is None

    def test_creates_image_result_with_dimensions(self) -> None:
        """Test that ImageContentResult accepts width/height.

        **Why this test is important:**
          - Dimensions are needed for image metadata
          - Validates optional fields work

        **What it tests:**
          - width and height stored correctly
        """
        result = ImageContentResult(
            s3_key="images/photo.jpg",
            image_bytes=b"\xff\xd8\xff",
            format="jpeg",
            size_bytes=1024,
            width=1920,
            height=1080,
        )

        assert result.width == 1920
        assert result.height == 1080

    def test_s3_uri_property(self) -> None:
        """Test that s3_uri property returns correct URI.

        **Why this test is important:**
          - S3 URIs used for referencing objects
          - Validates URI format

        **What it tests:**
          - s3_uri returns s3://<key> format
        """
        result = ImageContentResult(
            s3_key="images/photo.jpg",
            image_bytes=b"\xff\xd8\xff",
            format="jpeg",
            size_bytes=100,
        )

        assert result.s3_uri == "s3://images/photo.jpg"

    def test_mime_type_property_jpeg(self) -> None:
        """Test mime_type property for JPEG format.

        **Why this test is important:**
          - MIME types needed for HTTP responses
          - Validates JPEG mapping

        **What it tests:**
          - Returns image/jpeg for jpeg format
        """
        result = ImageContentResult(
            s3_key="photo.jpg",
            image_bytes=b"\xff\xd8\xff",
            format="jpeg",
            size_bytes=100,
        )

        assert result.mime_type == "image/jpeg"

    def test_mime_type_property_png(self) -> None:
        """Test mime_type property for PNG format.

        **Why this test is important:**
          - MIME types needed for HTTP responses
          - Validates PNG mapping

        **What it tests:**
          - Returns image/png for png format
        """
        result = ImageContentResult(
            s3_key="photo.png",
            image_bytes=b"\x89PNG",
            format="png",
            size_bytes=100,
        )

        assert result.mime_type == "image/png"

    def test_mime_type_property_webp(self) -> None:
        """Test mime_type property for WebP format.

        **Why this test is important:**
          - WebP is common modern format
          - Validates WebP mapping

        **What it tests:**
          - Returns image/webp for webp format
        """
        result = ImageContentResult(
            s3_key="photo.webp",
            image_bytes=b"RIFF",
            format="webp",
            size_bytes=100,
        )

        assert result.mime_type == "image/webp"

    def test_mime_type_property_gif(self) -> None:
        """Test mime_type property for GIF format.

        **Why this test is important:**
          - GIF support for animations
          - Validates GIF mapping

        **What it tests:**
          - Returns image/gif for gif format
        """
        result = ImageContentResult(
            s3_key="photo.gif",
            image_bytes=b"GIF89a",
            format="gif",
            size_bytes=100,
        )

        assert result.mime_type == "image/gif"

    def test_mime_type_property_unknown(self) -> None:
        """Test mime_type property for unknown format.

        **Why this test is important:**
          - Unknown formats may occur
          - Should return safe default

        **What it tests:**
          - Returns application/octet-stream for unknown
        """
        result = ImageContentResult(
            s3_key="photo.unknown",
            image_bytes=b"data",
            format="unknown",
            size_bytes=100,
        )

        assert result.mime_type == "application/octet-stream"

    def test_image_result_is_immutable(self) -> None:
        """Test that ImageContentResult is immutable.

        **Why this test is important:**
          - Immutability ensures thread safety
          - Prevents accidental modification

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        result = ImageContentResult(
            s3_key="photo.jpg",
            image_bytes=b"\xff\xd8\xff",
            format="jpeg",
            size_bytes=100,
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            result.format = "png"


# =============================================================================
# BatchEmbeddingResult Tests
# =============================================================================


class TestBatchEmbeddingResult:
    """Test suite for BatchEmbeddingResult type class."""

    def test_creates_empty_batch(self) -> None:
        """Test creating an empty batch.

        **Why this test is important:**
          - Empty batches are valid edge case
          - Must handle gracefully

        **What it tests:**
          - Empty batch created correctly
          - is_empty returns True
        """
        batch = BatchEmbeddingResult(qdrant_points=[], weaviate_objects=[])

        assert len(batch) == 0
        assert batch.is_empty() is True

    def test_creates_batch_with_points(self) -> None:
        """Test creating a batch with points.

        **Why this test is important:**
          - Normal use case
          - Validates both database lists

        **What it tests:**
          - Points stored correctly
          - Length matches
        """
        qdrant_point = VectorPoint(id="1", vector=[0.1, 0.2])
        weaviate_obj = MagicMock()

        batch = BatchEmbeddingResult(
            qdrant_points=[qdrant_point],
            weaviate_objects=[weaviate_obj],
        )

        assert len(batch) == 1
        assert batch.is_empty() is False
        assert batch.qdrant_points[0] == qdrant_point

    def test_batch_length_matches_qdrant_points(self) -> None:
        """Test that __len__ returns qdrant_points count.

        **Why this test is important:**
          - Length used for progress tracking
          - Must be consistent

        **What it tests:**
          - len() returns qdrant_points count
        """
        points = [VectorPoint(id=str(i), vector=[0.1]) for i in range(5)]
        batch = BatchEmbeddingResult(qdrant_points=points, weaviate_objects=[])

        assert len(batch) == 5

    def test_batch_is_immutable(self) -> None:
        """Test that BatchEmbeddingResult is immutable.

        **Why this test is important:**
          - Immutability for thread safety

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        batch = BatchEmbeddingResult(qdrant_points=[], weaviate_objects=[])

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            batch.qdrant_points = []


# =============================================================================
# ProcessingConfig Tests
# =============================================================================


class TestProcessingConfig:
    """Test suite for ProcessingConfig type class."""

    def test_creates_config_with_required_fields(self) -> None:
        """Test that ProcessingConfig is created with required fields.

        **Why this test is important:**
          - Configuration is foundation of pipeline
          - Validates all required fields

        **What it tests:**
          - Config created with required fields
          - Default values set correctly
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = ProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        assert config.s3_endpoint == "http://localhost:9000"
        assert config.s3_bucket == "documents"
        assert config.collection == "test-collection"
        assert config.embed_batch_size == 8
        assert config.upsert_batch_size == 200

    def test_config_with_custom_batch_sizes(self) -> None:
        """Test that ProcessingConfig accepts custom batch sizes.

        **Why this test is important:**
          - Batch sizes affect performance
          - Must support customization

        **What it tests:**
          - Custom values are stored
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = ProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
            embed_batch_size=16,
            upsert_batch_size=500,
        )

        assert config.embed_batch_size == 16
        assert config.upsert_batch_size == 500

    def test_config_is_immutable(self) -> None:
        """Test that ProcessingConfig is immutable.

        **Why this test is important:**
          - Immutability ensures thread safety

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = ProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.s3_bucket = "new-bucket"


# =============================================================================
# RateLimitConfig Tests
# =============================================================================


class TestRateLimitConfig:
    """Test suite for RateLimitConfig type class."""

    def test_creates_config_with_defaults(self) -> None:
        """Test that RateLimitConfig has sensible defaults.

        **Why this test is important:**
          - Defaults should be safe
          - Validates default values

        **What it tests:**
          - Default values set correctly
        """
        config = RateLimitConfig()

        assert config.requests_per_second == 5
        assert config.max_concurrency == 10

    def test_creates_config_with_custom_values(self) -> None:
        """Test that RateLimitConfig accepts custom values.

        **Why this test is important:**
          - Different APIs have different limits
          - Must support customization

        **What it tests:**
          - Custom values stored correctly
        """
        config = RateLimitConfig(requests_per_second=10, max_concurrency=20)

        assert config.requests_per_second == 10
        assert config.max_concurrency == 20

    def test_config_is_immutable(self) -> None:
        """Test that RateLimitConfig is immutable.

        **Why this test is important:**
          - Immutability for thread safety

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        config = RateLimitConfig()

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.requests_per_second = 100


# =============================================================================
# ProcessingClients Tests
# =============================================================================


class TestProcessingClients:
    """Test suite for ProcessingClients type class."""

    def test_creates_clients_bundle(self) -> None:
        """Test that ProcessingClients bundles all clients.

        **Why this test is important:**
          - Client bundle is core abstraction
          - Validates all fields stored

        **What it tests:**
          - All clients accessible
        """
        mock_s3 = MagicMock()
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_weaviate = MagicMock()
        mock_session = MagicMock()

        clients = ProcessingClients(
            s3=mock_s3,
            embedder=mock_embedder,
            qdrant_db=mock_qdrant,
            weaviate_db=mock_weaviate,
            session=mock_session,
        )

        assert clients.s3 == mock_s3
        assert clients.embedder == mock_embedder
        assert clients.qdrant_db == mock_qdrant
        assert clients.weaviate_db == mock_weaviate
        assert clients.session == mock_session

    def test_close_sync_calls_all_closes(self) -> None:
        """Test that close_sync closes all clients.

        **Why this test is important:**
          - Resource cleanup is critical
          - Must close all clients

        **What it tests:**
          - All close methods called
        """
        mock_s3 = MagicMock()
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_weaviate = MagicMock()
        mock_session = MagicMock()

        clients = ProcessingClients(
            s3=mock_s3,
            embedder=mock_embedder,
            qdrant_db=mock_qdrant,
            weaviate_db=mock_weaviate,
            session=mock_session,
        )

        clients.close_sync()

        mock_qdrant.close.assert_called_once()
        mock_weaviate.close.assert_called_once()
        mock_session.close.assert_called_once()

    def test_close_sync_handles_errors(self) -> None:
        """Test that close_sync handles errors gracefully.

        **Why this test is important:**
          - Cleanup must not raise
          - Should continue on error

        **What it tests:**
          - Errors are caught, other closes continue
        """
        mock_s3 = MagicMock()
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_qdrant.close.side_effect = Exception("Close failed")
        mock_weaviate = MagicMock()
        mock_session = MagicMock()

        clients = ProcessingClients(
            s3=mock_s3,
            embedder=mock_embedder,
            qdrant_db=mock_qdrant,
            weaviate_db=mock_weaviate,
            session=mock_session,
        )

        # Should not raise
        clients.close_sync()

        # Other closes still called
        mock_weaviate.close.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_async_closes_async_clients(self) -> None:
        """Test that close_async handles async clients.

        **Why this test is important:**
          - Spark uses async clients
          - Must close properly

        **What it tests:**
          - Async close methods called
        """
        mock_s3 = MagicMock()
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()
        mock_qdrant._client = MagicMock()
        mock_qdrant._client.close = AsyncMock()
        mock_weaviate = MagicMock()
        mock_weaviate._client = MagicMock()
        mock_weaviate._client.close = AsyncMock()
        mock_session = MagicMock()
        mock_session.close = AsyncMock()

        clients = ProcessingClients(
            s3=mock_s3,
            embedder=mock_embedder,
            qdrant_db=mock_qdrant,
            weaviate_db=mock_weaviate,
            session=mock_session,
        )

        await clients.close_async()

        mock_qdrant._client.close.assert_awaited_once()
        mock_weaviate._client.close.assert_awaited_once()
        mock_session.close.assert_awaited_once()
