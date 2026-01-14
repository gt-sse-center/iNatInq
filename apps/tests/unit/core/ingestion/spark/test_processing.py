"""Unit tests for core.ingestion.spark.processing module.

This file tests the Spark-specific processing classes that handle S3 object
processing, embedding generation, and vector database upserts.

# Test Coverage

The tests cover:
  - SparkProcessingConfig: Configuration with Spark-specific settings
  - SparkProcessingPipeline: Pipeline initialization and processing

# Test Structure

Tests use pytest class-based organization with mocking for external dependencies.
Spark RDD operations are not tested here (require integration tests).

# Running Tests

Run with: pytest tests/unit/core/ingestion/spark/test_processing.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import attrs.exceptions
import pytest
from config import EmbeddingConfig
from core.ingestion.interfaces import (
    ProcessingClientsFactory,
)
from core.ingestion.spark.processing import (
    SparkProcessingConfig,
    SparkProcessingPipeline,
)

# =============================================================================
# SparkProcessingConfig Tests
# =============================================================================


class TestSparkProcessingConfig:
    """Test suite for SparkProcessingConfig type class."""

    def test_creates_config_with_required_fields(self) -> None:
        """Test that SparkProcessingConfig is created with required fields.

        **Why this test is important:**
          - Configuration is foundation of pipeline
          - Validates inheritance from ProcessingConfig

        **What it tests:**
          - Config created with required fields
          - Default Spark-specific values set
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = SparkProcessingConfig(
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
        # Base defaults
        assert config.embed_batch_size == 8
        assert config.upsert_batch_size == 200
        # Spark-specific defaults
        assert config.ollama_max_concurrency == 10
        assert config.ollama_rps == 5
        assert config.min_embed_batch == 1
        assert config.max_embed_batch == 8

    def test_creates_config_with_custom_spark_settings(self) -> None:
        """Test that SparkProcessingConfig accepts custom Spark settings.

        **Why this test is important:**
          - Spark-specific tuning needed
          - Validates optional parameters

        **What it tests:**
          - Custom Spark values stored
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
            ollama_max_concurrency=20,
            ollama_rps=10,
            min_embed_batch=2,
            max_embed_batch=16,
        )

        assert config.ollama_max_concurrency == 20
        assert config.ollama_rps == 10
        assert config.min_embed_batch == 2
        assert config.max_embed_batch == 16

    def test_config_is_immutable(self) -> None:
        """Test that SparkProcessingConfig is immutable.

        **Why this test is important:**
          - Immutability ensures thread safety
          - Prevents accidental modification

        **What it tests:**
          - Modification raises FrozenInstanceError
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            config.ollama_rps = 100


# =============================================================================
# SparkProcessingPipeline Tests
# =============================================================================


class TestSparkProcessingPipeline:
    """Test suite for SparkProcessingPipeline."""

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

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = SparkProcessingPipeline(config)

        assert pipeline.config == config

    def test_config_property(self) -> None:
        """Test config property access.

        **Why this test is important:**
          - Config must be accessible
          - Validates property

        **What it tests:**
          - Config property returns config
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = SparkProcessingPipeline(config)

        assert pipeline.config.s3_bucket == "documents"
        assert pipeline.config.ollama_rps == 5

    @pytest.mark.asyncio
    async def test_process_keys_async_empty_list(self) -> None:
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

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = SparkProcessingPipeline(config)
        results = await pipeline.process_keys_async([])

        assert results == []

    @patch("core.ingestion.spark.processing.RateLimiter")
    @patch("core.ingestion.spark.processing.ProcessingClientsFactory")
    @pytest.mark.asyncio
    async def test_process_keys_async_creates_clients(
        self, mock_factory_class, mock_rate_limiter_class
    ) -> None:
        """Test that processing creates clients.

        **Why this test is important:**
          - Clients must be created for processing
          - Validates factory usage

        **What it tests:**
          - Factory create called
        """
        # Set up rate limiter mock with async acquire
        mock_rate_limiter_class.return_value.acquire = AsyncMock()

        mock_clients = MagicMock()
        mock_clients.s3 = MagicMock()
        mock_clients.s3.get_object.return_value = b"content"
        mock_clients.embedder = MagicMock()
        mock_clients.embedder.embed_async = AsyncMock(return_value=[0.1, 0.2])
        mock_clients.embedder.vector_size = 768
        mock_clients.qdrant_db = MagicMock()
        mock_clients.qdrant_db.batch_upsert_async = AsyncMock(return_value=None)
        mock_clients.weaviate_db = MagicMock()
        mock_clients.weaviate_db.batch_upsert_async = AsyncMock(return_value=None)
        mock_clients.close_async = AsyncMock()
        mock_factory_class.return_value.create.return_value = mock_clients

        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = SparkProcessingPipeline(config)
        await pipeline.process_keys_async(["doc.txt"])

        mock_factory_class.return_value.create.assert_called_once_with(config)

    @patch("core.ingestion.spark.processing.RateLimiter")
    @patch("core.ingestion.spark.processing.ProcessingClientsFactory")
    @pytest.mark.asyncio
    async def test_process_keys_async_closes_clients(
        self, mock_factory_class, mock_rate_limiter_class
    ) -> None:
        """Test that clients are closed after processing.

        **Why this test is important:**
          - Resource cleanup critical
          - Must close in finally block

        **What it tests:**
          - close_async called
        """
        # Set up rate limiter mock with async acquire
        mock_rate_limiter_class.return_value.acquire = AsyncMock()

        mock_clients = MagicMock()
        mock_clients.s3 = MagicMock()
        mock_clients.s3.get_object.return_value = b"content"
        mock_clients.embedder = MagicMock()
        mock_clients.embedder.embed_async = AsyncMock(return_value=[0.1, 0.2])
        mock_clients.embedder.vector_size = 768
        mock_clients.qdrant_db = MagicMock()
        mock_clients.qdrant_db.batch_upsert_async = AsyncMock(return_value=None)
        mock_clients.weaviate_db = MagicMock()
        mock_clients.weaviate_db.batch_upsert_async = AsyncMock(return_value=None)
        mock_clients.close_async = AsyncMock()
        mock_factory_class.return_value.create.return_value = mock_clients

        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = SparkProcessingPipeline(config)
        await pipeline.process_keys_async(["doc.txt"])

        mock_clients.close_async.assert_awaited_once()

    @patch("core.ingestion.spark.processing.ProcessingClientsFactory")
    @pytest.mark.asyncio
    async def test_process_keys_async_handles_fetch_failures(self, mock_factory_class) -> None:
        """Test that fetch failures are tracked.

        **Why this test is important:**
          - Partial failures happen
          - Must return failure results

        **What it tests:**
          - Failure result returned
        """
        from botocore.exceptions import ClientError

        mock_clients = MagicMock()
        mock_clients.s3 = MagicMock()
        mock_clients.s3.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        mock_clients.close_async = AsyncMock()
        mock_factory_class.return_value.create.return_value = mock_clients

        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        config = SparkProcessingConfig(
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
        )

        pipeline = SparkProcessingPipeline(config)
        results = await pipeline.process_keys_async(["doc.txt"])

        assert len(results) == 1
        assert results[0].success is False


# =============================================================================
# Partition Function Tests
# =============================================================================


class TestProcessPartitionAsync:
    """Test suite for process_partition_async function."""

    @patch("core.ingestion.spark.processing.ProcessingClientsFactory")
    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_keys(self, mock_factory_class) -> None:
        """Test that empty keys returns empty iterator.

        **Why this test is important:**
          - Edge case handling
          - Should not create clients

        **What it tests:**
          - Empty iterator returned
        """
        from core.ingestion.spark.processing import process_partition_async

        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        result = await process_partition_async(
            keys=[],
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
            ollama_max_concurrency=10,
            ollama_rps=5,
            min_embed_batch=1,
            max_embed_batch=8,
        )

        assert list(result) == []
        mock_factory_class.return_value.create.assert_not_called()

    @patch("core.ingestion.spark.processing.RateLimiter")
    @patch("core.ingestion.spark.processing.ProcessingClientsFactory")
    @pytest.mark.asyncio
    async def test_returns_tuples(self, mock_factory_class, mock_rate_limiter_class) -> None:
        """Test that results are returned as tuples.

        **Why this test is important:**
          - Spark requires tuple format
          - Validates backward compatibility

        **What it tests:**
          - Results are tuples
        """
        from core.ingestion.spark.processing import process_partition_async

        # Set up rate limiter mock with async acquire
        mock_rate_limiter_class.return_value.acquire = AsyncMock()

        mock_clients = MagicMock()
        mock_clients.s3 = MagicMock()
        mock_clients.s3.get_object.return_value = b"content"
        mock_clients.embedder = MagicMock()
        mock_clients.embedder.embed_async = AsyncMock(return_value=[0.1, 0.2])
        mock_clients.embedder.vector_size = 768
        mock_clients.qdrant_db = MagicMock()
        mock_clients.qdrant_db.batch_upsert_async = AsyncMock(return_value=None)
        mock_clients.weaviate_db = MagicMock()
        mock_clients.weaviate_db.batch_upsert_async = AsyncMock(return_value=None)
        mock_clients.close_async = AsyncMock()
        mock_factory_class.return_value.create.return_value = mock_clients

        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_model="nomic-embed-text",
            ollama_url="http://localhost:11434",
        )

        result = await process_partition_async(
            keys=["doc.txt"],
            s3_endpoint="http://localhost:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="documents",
            embedding_config=embed_config,
            collection="test-collection",
            ollama_max_concurrency=10,
            ollama_rps=5,
            min_embed_batch=1,
            max_embed_batch=8,
        )

        result_list = list(result)
        assert len(result_list) == 1
        assert isinstance(result_list[0], tuple)
        assert len(result_list[0]) == 3  # (key, success, error)
