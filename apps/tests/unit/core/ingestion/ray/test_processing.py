"""Unit tests for core.ingestion.ray.processing module.

This file tests the refactored Ray processing functions that handle S3 object
processing, embedding generation, and vector database upserts.

# Test Coverage

The tests cover:
  - Type Classes: ProcessingConfig, ProcessingResult, ContentResult, etc.
  - Operation Classes: S3ContentFetcher, EmbeddingGenerator, VectorDBUpserter
  - Factory Classes: VectorDBConfigFactory, ProcessingClientsFactory, VectorPointFactory
  - Pipeline: RayProcessingPipeline

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
Ray remote functions are not tested here (require integration tests).

# Running Tests

Run with: pytest tests/unit/core/ingestion/ray/test_processing.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import attrs.exceptions
import pytest
from botocore.exceptions import ClientError
from config import EmbeddingConfig
from core.exceptions import UpstreamError
from core.ingestion.interfaces import (
    BatchEmbeddingResult,
    ContentResult,
    EmbeddingGenerator,
    ProcessingClientsFactory,
    ProcessingConfig,
    ProcessingResult,
    S3ContentFetcher,
    UpsertResult,
    VectorDBConfigFactory,
    VectorDBUpserter,
    VectorPointFactory,
)
from core.ingestion.ray.processing import (
    RayProcessingConfig,
    RayProcessingPipeline,
)
from core.models import VectorPoint

# =============================================================================
# Type Classes Tests
# =============================================================================


class TestProcessingConfig:
    """Test suite for ProcessingConfig type class."""

    def test_creates_config_with_required_fields(self) -> None:
        """Test that ProcessingConfig is created with required fields.

        **Why this test is important:**
          - ProcessingConfig centralizes pipeline configuration
          - Validates attrs integration
          - Ensures required fields are enforced
          - Critical for type safety

        **What it tests:**
          - Config created with all required fields
          - Default values are set correctly
          - Fields are accessible as attributes
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
        assert config.embed_batch_size == 8  # Default
        assert config.upsert_batch_size == 200  # Default

    def test_config_with_custom_batch_sizes(self) -> None:
        """Test that ProcessingConfig accepts custom batch sizes.

        **Why this test is important:**
          - Batch sizes affect performance
          - Must support customization
          - Validates optional parameter handling
          - Critical for tuning

        **What it tests:**
          - Custom embed_batch_size is stored
          - Custom upsert_batch_size is stored
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
        """Test that ProcessingConfig is immutable (frozen).

        **Why this test is important:**
          - Immutability ensures thread safety
          - Prevents accidental modification
          - Validates attrs frozen=True setting
          - Critical for distributed processing

        **What it tests:**
          - Attempting to modify raises FrozenInstanceError
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


class TestProcessingResult:
    """Test suite for ProcessingResult type class."""

    def test_creates_success_result(self) -> None:
        """Test creating a successful result.

        **Why this test is important:**
          - ProcessingResult tracks individual object status
          - Success path is common
          - Validates factory method

        **What it tests:**
          - success_result creates correct object
          - success=True, error_message=""
        """
        result = ProcessingResult.success_result("inputs/doc.txt")

        assert result.s3_key == "inputs/doc.txt"
        assert result.success is True
        assert result.error_message == ""

    def test_creates_failure_result(self) -> None:
        """Test creating a failure result.

        **Why this test is important:**
          - Error tracking is critical
          - Failure path must be tested
          - Validates factory method

        **What it tests:**
          - failure_result creates correct object
          - success=False, error_message set
        """
        result = ProcessingResult.failure_result("inputs/doc.txt", "S3 fetch failed")

        assert result.s3_key == "inputs/doc.txt"
        assert result.success is False
        assert result.error_message == "S3 fetch failed"

    def test_to_tuple_format(self) -> None:
        """Test that to_tuple returns correct format.

        **Why this test is important:**
          - Ray/Spark require tuple format
          - Validates backward compatibility
          - Critical for distributed processing

        **What it tests:**
          - to_tuple returns (key, success, error)
        """
        result = ProcessingResult(s3_key="doc.txt", success=True, error_message="")
        assert result.to_tuple() == ("doc.txt", True, "")


class TestContentResult:
    """Test suite for ContentResult type class."""

    def test_creates_content_result(self) -> None:
        """Test creating a content result.

        **Why this test is important:**
          - ContentResult holds S3 content
          - Simple but critical type
          - Validates attrs integration

        **What it tests:**
          - ContentResult stores key and content
        """
        content = ContentResult(s3_key="inputs/doc.txt", content="Hello world")

        assert content.s3_key == "inputs/doc.txt"
        assert content.content == "Hello world"

    def test_content_result_is_immutable(self) -> None:
        """Test that ContentResult is immutable.

        **Why this test is important:**
          - Immutability for thread safety
          - Validates frozen=True

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        content = ContentResult(s3_key="doc.txt", content="test")

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            content.content = "modified"


# =============================================================================
# Factory Classes Tests
# =============================================================================


class TestVectorDBConfigFactory:
    """Test suite for VectorDBConfigFactory."""

    def test_creates_qdrant_config(self) -> None:
        """Test creating Qdrant configuration.

        **Why this test is important:**
          - Factory abstracts config creation
          - Validates environment detection

        **What it tests:**
          - create_qdrant_config returns valid config
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_qdrant_config()

        assert config.provider_type == "qdrant"

    def test_creates_weaviate_config(self) -> None:
        """Test creating Weaviate configuration.

        **Why this test is important:**
          - Dual database support
          - Validates config creation

        **What it tests:**
          - create_weaviate_config returns valid config
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        config = factory.create_weaviate_config()

        assert config.provider_type == "weaviate"

    def test_creates_both_configs(self) -> None:
        """Test creating both configurations.

        **Why this test is important:**
          - Common use case
          - Validates tuple return

        **What it tests:**
          - create_both returns (qdrant, weaviate) tuple
        """
        factory = VectorDBConfigFactory(namespace="ml-system")
        qdrant, weaviate = factory.create_both()

        assert qdrant.provider_type == "qdrant"
        assert weaviate.provider_type == "weaviate"


class TestVectorPointFactory:
    """Test suite for VectorPointFactory."""

    def test_creates_qdrant_point(self) -> None:
        """Test creating a Qdrant VectorPoint.

        **Why this test is important:**
          - VectorPoint is core data structure
          - Validates metadata creation

        **What it tests:**
          - create_qdrant_point returns VectorPoint
          - Metadata includes s3_key, bucket, uri, text
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        content = ContentResult(s3_key="doc.txt", content="Hello")
        vector = [0.1, 0.2, 0.3]

        point = factory.create_qdrant_point(content, vector)

        assert point.vector == vector
        assert point.payload["s3_key"] == "doc.txt"
        assert point.payload["s3_bucket"] == "pipeline"
        assert point.payload["text"] == "Hello"

    def test_creates_weaviate_object(self) -> None:
        """Test creating a Weaviate data object.

        **Why this test is important:**
          - Weaviate requires different format
          - Validates properties structure

        **What it tests:**
          - create_weaviate_object returns WeaviateDataObject
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        content = ContentResult(s3_key="doc.txt", content="Hello")
        vector = [0.1, 0.2, 0.3]

        obj = factory.create_weaviate_object(content, vector)

        assert obj.vector == vector
        assert obj.properties["s3_key"] == "doc.txt"

    def test_creates_matching_pair(self) -> None:
        """Test creating matching Qdrant and Weaviate points.

        **Why this test is important:**
          - Points must have same UUID
          - Critical for data consistency

        **What it tests:**
          - create_pair returns matching UUIDs
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        content = ContentResult(s3_key="doc.txt", content="Hello")
        vector = [0.1, 0.2, 0.3]

        qdrant_point, weaviate_obj = factory.create_pair(content, vector)

        assert qdrant_point.id == weaviate_obj.uuid

    def test_creates_batch(self) -> None:
        """Test creating batch of vector points.

        **Why this test is important:**
          - Batch creation is common
          - Validates length matching

        **What it tests:**
          - create_batch returns BatchEmbeddingResult
        """
        factory = VectorPointFactory(s3_bucket="pipeline")
        contents = [
            ContentResult(s3_key="doc1.txt", content="Hello"),
            ContentResult(s3_key="doc2.txt", content="World"),
        ]
        vectors = [[0.1, 0.2], [0.3, 0.4]]

        batch = factory.create_batch(contents, vectors)

        assert len(batch.qdrant_points) == 2
        assert len(batch.weaviate_objects) == 2


# =============================================================================
# Operation Classes Tests
# =============================================================================


class TestS3ContentFetcher:
    """Test suite for S3ContentFetcher."""

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

    def test_fetch_one_failure(self) -> None:
        """Test fetch failure handling.

        **Why this test is important:**
          - Error handling is critical
          - Validates None return

        **What it tests:**
          - fetch_one returns None on error
        """
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        result = fetcher.fetch_one("missing.txt")

        assert result is None

    def test_fetch_all_mixed_results(self) -> None:
        """Test fetching multiple objects with mixed results.

        **Why this test is important:**
          - Batch fetching is common
          - Validates success/failure separation

        **What it tests:**
          - fetch_all returns (contents, failures) tuple
        """
        mock_s3 = MagicMock()

        def side_effect(bucket, key):
            if key == "doc1.txt":
                return b"Content 1"
            else:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        mock_s3.get_object.side_effect = side_effect

        fetcher = S3ContentFetcher(mock_s3, bucket="pipeline")
        contents, failures = fetcher.fetch_all(["doc1.txt", "doc2.txt"])

        assert len(contents) == 1
        assert len(failures) == 1
        assert contents[0].s3_key == "doc1.txt"
        assert failures[0].s3_key == "doc2.txt"


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""

    @pytest.mark.asyncio
    async def test_generate_one_async_success(self) -> None:
        """Test successful single embedding generation.

        **Why this test is important:**
          - Core embedding operation
          - Validates async behavior

        **What it tests:**
          - generate_one_async returns vector on success
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
          - Error handling is critical
          - Validates None return

        **What it tests:**
          - generate_one_async returns None on error
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_async = AsyncMock(side_effect=UpstreamError("Service unavailable"))

        generator = EmbeddingGenerator(mock_embedder)
        result = await generator.generate_one_async("Hello world")

        assert result is None

    @pytest.mark.asyncio
    async def test_generate_batch_async_success(self) -> None:
        """Test successful batch embedding generation.

        **Why this test is important:**
          - Batch embedding is common
          - Validates multiple vectors

        **What it tests:**
          - generate_batch_async returns list of vectors
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


class TestVectorDBUpserter:
    """Test suite for VectorDBUpserter."""

    @pytest.mark.asyncio
    async def test_upsert_batch_async_success(self) -> None:
        """Test successful batch upsert.

        **Why this test is important:**
          - Upserting is critical operation
          - Validates dual database support

        **What it tests:**
          - upsert_batch_async returns True on success
        """
        mock_qdrant = MagicMock()
        mock_qdrant.batch_upsert_async = AsyncMock(return_value=None)
        mock_weaviate = MagicMock()
        mock_weaviate.batch_upsert_async = AsyncMock(return_value=None)

        upserter = VectorDBUpserter(mock_qdrant, mock_weaviate)

        # Create mock batch
        mock_point = MagicMock(spec=VectorPoint)
        mock_point.to_qdrant.return_value = MagicMock()
        batch = BatchEmbeddingResult(
            qdrant_points=[mock_point],
            weaviate_objects=[MagicMock()],
        )

        result = await upserter.upsert_batch_async(batch, "documents", 768)

        assert isinstance(result, UpsertResult)
        assert result.all_success
        mock_qdrant.batch_upsert_async.assert_called_once()
        mock_weaviate.batch_upsert_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_batch_async_partial_failure(self) -> None:
        """Test partial failure handling.

        **Why this test is important:**
          - One DB may fail while other succeeds
          - Validates partial success

        **What it tests:**
          - Returns True if at least one DB succeeds
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


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestRayProcessingPipeline:
    """Test suite for RayProcessingPipeline."""

    def test_creates_pipeline_with_config(self) -> None:
        """Test pipeline creation.

        **Why this test is important:**
          - Pipeline is main entry point
          - Validates config storage

        **What it tests:**
          - Pipeline stores config correctly
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = RayProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = RayProcessingPipeline(config)

        assert pipeline.config == config

    @patch.object(ProcessingClientsFactory, "create")
    def test_process_keys_sync_empty_list(self, mock_create) -> None:
        """Test processing empty key list.

        **Why this test is important:**
          - Edge case handling
          - Should return empty without creating clients

        **What it tests:**
          - Returns empty list for empty input
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = RayProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = RayProcessingPipeline(config)
        results = pipeline.process_keys_sync([])

        assert results == []
        mock_create.assert_not_called()
