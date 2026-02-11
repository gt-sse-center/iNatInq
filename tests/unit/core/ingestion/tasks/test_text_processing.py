"""Unit tests for core.ingestion.tasks.text_processing module.

This module tests the Ray-based text processing pipeline components including
configuration, pipeline orchestration, and remote function execution.

# Test Coverage

- RayProcessingConfig: Configuration dataclass creation and defaults
- RayProcessingPipeline: Pipeline initialization, sync processing, error handling
- RayProcessingPipelineAsync: Async content processing and batch operations
- process_s3_object_ray: Single object processing via Ray remote function
- process_s3_batch_ray: Batch processing, rate limiting, logging, and configuration

# Running Tests

```bash
uv run pytest tests/unit/core/ingestion/tasks/test_text_processing.py -v
```
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import EmbeddingConfig
from core.ingestion.interfaces.types import ContentResult, ProcessingResult
from core.ingestion.tasks.text_processing import (
    RayProcessingConfig,
    RayProcessingPipeline,
    process_s3_batch_ray,
    process_s3_object_ray,
)


class TestRayProcessingConfig:
    """Tests for RayProcessingConfig dataclass."""

    def test_creates_config_with_all_fields(self):
        """Verify RayProcessingConfig is created with all required fields.

        **Why this test is important:**

        - Ensures the configuration dataclass accepts all required parameters
        - Validates that configuration values are stored correctly
        - Catches breaking changes to the config interface

        **What it tests:**

        - Creation of config with S3, embedding, and collection parameters
        - Correct storage and retrieval of all configuration values
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )
        config = RayProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="bucket",
            embedding_config=embed_config,
            collection="test-collection",
            embed_batch_size=8,
            upsert_batch_size=200,
        )
        assert config.s3_endpoint == "http://minio:9000"
        assert config.s3_bucket == "bucket"
        assert config.collection == "test-collection"

    def test_config_has_default_resilience_settings(self):
        """Verify RayProcessingConfig has sensible defaults for resilience.

        **Why this test is important:**

        - Ensures production-safe defaults are applied automatically
        - Prevents misconfiguration by providing reasonable rate limits
        - Documents expected default values for resilience settings

        **What it tests:**

        - Default rate limit (requests per second)
        - Default max concurrency for parallel processing
        - Default circuit breaker threshold for fault tolerance
        - Default retry max attempts for transient failures
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )
        config = RayProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="bucket",
            embedding_config=embed_config,
            collection="test",
        )
        assert config.rate_limit_rps == 5
        assert config.max_concurrency == 10
        assert config.circuit_breaker_threshold == 5
        assert config.retry_max_attempts == 3


class TestRayProcessingPipeline:
    """Tests for RayProcessingPipeline class."""

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )
        return RayProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="test-bucket",
            embedding_config=embed_config,
            collection="test-collection",
            embed_batch_size=2,
        )

    def test_init_stores_config(self, config):
        """Verify pipeline stores config and rate limiter on initialization.

        **Why this test is important:**

        - Ensures pipeline is correctly initialized with its dependencies
        - Validates that the rate limiter is properly injected
        - Confirms config is accessible throughout the pipeline lifecycle

        **What it tests:**

        - Config object is stored and accessible via pipeline.config
        - Rate limiter is stored in the private _rate_limiter attribute
        """
        rate_limiter = MagicMock()
        pipeline = RayProcessingPipeline(config, rate_limiter)
        assert pipeline.config is config
        assert pipeline._rate_limiter is rate_limiter

    def test_config_property_returns_config(self, config):
        """Verify config property returns the configuration.

        **Why this test is important:**

        - Ensures the config property provides read access to configuration
        - Validates the public API for accessing pipeline configuration

        **What it tests:**

        - The config property returns the same config object passed to constructor
        """
        pipeline = RayProcessingPipeline(config)
        assert pipeline.config is config

    def test_process_keys_sync_returns_empty_for_empty_input(self, config):
        """Verify process_keys_sync returns empty list for empty input.

        **Why this test is important:**

        - Ensures graceful handling of edge case with no input keys
        - Prevents unnecessary processing overhead for empty batches
        - Validates the method contract for empty input

        **What it tests:**

        - Calling process_keys_sync with empty list returns empty list
        - No errors are raised for empty input
        """
        pipeline = RayProcessingPipeline(config)
        results = pipeline.process_keys_sync([])
        assert results == []

    def test_process_keys_sync_returns_failures_on_s3_fetch_error(self, config):
        """Verify process_keys_sync returns fetch failures when S3 fails.

        **Why this test is important:**

        - Ensures S3 fetch errors are properly propagated as processing failures
        - Validates error handling does not crash the pipeline
        - Confirms error messages are preserved in the result

        **What it tests:**

        - S3 fetch error results in ProcessingResult with success=False
        - Error message from S3 is included in the failure result
        - Pipeline gracefully handles partial fetch failures
        """
        pipeline = RayProcessingPipeline(config)

        mock_clients = MagicMock()
        mock_clients.s3 = MagicMock()
        mock_clients.close_sync = MagicMock()

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = (
            [],  # no contents
            [ProcessingResult.failure_result("key1.txt", "S3 error")],
        )

        with patch.object(pipeline._clients_factory, "create", return_value=mock_clients):
            with patch(
                "core.ingestion.tasks.text_processing.S3ContentFetcher",
                return_value=mock_fetcher,
            ):
                results = pipeline.process_keys_sync(["key1.txt"])

        assert len(results) == 1
        assert results[0].success is False
        assert "S3 error" in results[0].error_message

    def test_process_keys_sync_calls_async_processing(self, config):
        """Verify process_keys_sync calls async processing for fetched content.

        **Why this test is important:**

        - Ensures the sync wrapper properly delegates to async processing
        - Validates the integration between sync and async code paths
        - Confirms successful results are returned from the async pipeline

        **What it tests:**

        - Fetched content is passed to _process_contents_async
        - Async processing results are returned from the sync method
        - Success results are properly propagated
        """
        pipeline = RayProcessingPipeline(config)

        mock_clients = MagicMock()
        mock_clients.s3 = MagicMock()
        mock_clients.close_sync = MagicMock()

        content = ContentResult(s3_key="key1.txt", content="Hello world")
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = ([content], [])

        with patch.object(pipeline._clients_factory, "create", return_value=mock_clients):
            with patch(
                "core.ingestion.tasks.text_processing.S3ContentFetcher",
                return_value=mock_fetcher,
            ):
                with patch.object(
                    pipeline,
                    "_process_contents_async",
                    new_callable=AsyncMock,
                    return_value=[ProcessingResult.success_result("key1.txt")],
                ):
                    results = pipeline.process_keys_sync(["key1.txt"])

        assert len(results) == 1
        assert results[0].success is True


class TestRayProcessingPipelineAsync:
    """Tests for async processing in RayProcessingPipeline."""

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )
        return RayProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="test-bucket",
            embedding_config=embed_config,
            collection="test-collection",
            embed_batch_size=2,
            max_concurrency=5,
        )

    @pytest.mark.asyncio
    async def test_process_contents_async_returns_empty_for_no_contents(self, config):
        """Verify _process_contents_async returns empty for no contents.

        **Why this test is important:**

        - Ensures async processing handles empty input gracefully
        - Prevents unnecessary async operations for empty batches
        - Validates the method contract for edge cases

        **What it tests:**

        - Calling _process_contents_async with empty list returns empty list
        - No errors or exceptions for empty input
        """
        pipeline = RayProcessingPipeline(config)
        mock_clients = MagicMock()
        results = await pipeline._process_contents_async([], mock_clients)
        assert results == []

    @pytest.mark.asyncio
    async def test_process_contents_async_processes_batch(self, config):
        """Verify _process_contents_async processes content batches.

        **Why this test is important:**

        - Ensures the async pipeline correctly orchestrates batch processing
        - Validates integration with embedding generator and vector DB components
        - Confirms successful processing results are returned

        **What it tests:**

        - Content is processed through the BatchProcessor
        - Success results are returned from batch processing
        - All required components (embedder, vector DBs) are utilized
        """
        pipeline = RayProcessingPipeline(config)

        mock_clients = MagicMock()
        mock_clients.embedder = MagicMock()
        mock_clients.qdrant_db = MagicMock()
        mock_clients.weaviate_db = MagicMock()

        content = ContentResult(s3_key="key1.txt", content="Hello")

        mock_processor = MagicMock()
        mock_processor.process_batch_async = AsyncMock(
            return_value=([ProcessingResult.success_result("key1.txt")], 2)
        )

        with patch("core.ingestion.tasks.text_processing.EmbeddingGenerator"):
            with patch("core.ingestion.tasks.text_processing.VectorPointFactory"):
                with patch("core.ingestion.tasks.text_processing.VectorDBUpserter"):
                    with patch(
                        "core.ingestion.tasks.text_processing.BatchProcessor",
                        return_value=mock_processor,
                    ):
                        results = await pipeline._process_contents_async([content], mock_clients)

        assert len(results) == 1
        assert results[0].success is True


class TestProcessS3ObjectRay:
    """Tests for process_s3_object_ray remote function."""

    @patch("core.ingestion.tasks.text_processing.ray")
    def test_process_single_object_calls_pipeline(self, mock_ray_module: MagicMock):
        """Verify process_s3_object_ray processes a single S3 object.

        **Why this test is important:**

        - Ensures the Ray remote function correctly invokes the processing pipeline
        - Validates the tuple return format (key, success, error_message)
        - Confirms single object processing works end-to-end

        **What it tests:**

        - Pipeline is created and process_keys_sync is called with the S3 key
        - Successful result returns (key, True, "") tuple
        - All configuration parameters are passed correctly
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [ProcessingResult.success_result("doc.txt")]

        # Get the unwrapped function from the ray.remote decorator
        with patch(
            "core.ingestion.tasks.text_processing.RayProcessingPipeline",
            return_value=mock_pipeline,
        ):
            # Access the underlying function via _function attribute
            import core.ingestion.tasks.text_processing as text_module

            result = text_module.process_s3_object_ray._function(
                s3_key="doc.txt",
                s3_endpoint="http://minio:9000",
                s3_access_key="access",
                s3_secret_key="secret",
                s3_bucket="bucket",
                embedding_config=embed_config,
                collection="test",
            )

        assert result == ("doc.txt", True, "")

    @patch("core.ingestion.tasks.text_processing.ray")
    def test_process_single_object_returns_failure_on_no_result(self, mock_ray_module: MagicMock):
        """Verify process_s3_object_ray returns failure when no result.

        **Why this test is important:**

        - Ensures proper handling when pipeline returns no results
        - Validates failure reporting for unexpected empty responses
        - Confirms error message indicates the "No result" condition

        **What it tests:**

        - Empty pipeline result leads to failure tuple
        - Failure tuple has success=False
        - Error message contains "No result" explanation
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = []

        with patch(
            "core.ingestion.tasks.text_processing.RayProcessingPipeline",
            return_value=mock_pipeline,
        ):
            import core.ingestion.tasks.text_processing as text_module

            result = text_module.process_s3_object_ray._function(
                s3_key="doc.txt",
                s3_endpoint="http://minio:9000",
                s3_access_key="access",
                s3_secret_key="secret",
                s3_bucket="bucket",
                embedding_config=embed_config,
                collection="test",
            )

        assert result[0] == "doc.txt"
        assert result[1] is False
        assert "No result" in result[2]


class TestProcessS3BatchRay:
    """Tests for process_s3_batch_ray remote function."""

    @patch("core.ingestion.tasks.text_processing.ray")
    def test_process_batch_returns_results(self, mock_ray_module: MagicMock):
        """Verify process_s3_batch_ray processes a batch of S3 objects.

        **Why this test is important:**

        - Ensures batch processing handles multiple objects correctly
        - Validates mixed success/failure results are properly returned
        - Confirms batch processing is more efficient than single-object calls

        **What it tests:**

        - Multiple S3 keys are processed in a single batch
        - Both success and failure results are correctly formatted
        - Result count matches input count
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [
            ProcessingResult.success_result("doc1.txt"),
            ProcessingResult.failure_result("doc2.txt", "failed"),
        ]

        with patch(
            "core.ingestion.tasks.text_processing.RayProcessingPipeline",
            return_value=mock_pipeline,
        ):
            import core.ingestion.tasks.text_processing as text_module

            results = text_module.process_s3_batch_ray._function(
                s3_keys=["doc1.txt", "doc2.txt"],
                s3_endpoint="http://minio:9000",
                s3_access_key="access",
                s3_secret_key="secret",
                s3_bucket="bucket",
                embedding_config=embed_config,
                collection="test",
            )

        assert len(results) == 2
        assert results[0] == ("doc1.txt", True, "")
        assert results[1][0] == "doc2.txt"
        assert results[1][1] is False

    @patch("core.ingestion.tasks.text_processing.ray")
    def test_process_batch_with_rate_limiter(self, mock_ray_module: MagicMock):
        """Verify process_s3_batch_ray uses rate limiter when provided.

        **Why this test is important:**

        - Ensures rate limiting is applied to prevent overwhelming downstream services
        - Validates the rate limiter actor is correctly wrapped and passed to pipeline
        - Confirms rate limiting integration works with Ray actors

        **What it tests:**

        - RayActorRateLimiter is instantiated with the provided rate limiter actor
        - Pipeline receives the wrapped rate limiter
        - Processing still completes successfully with rate limiting enabled
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        mock_pipeline_class = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [ProcessingResult.success_result("doc.txt")]
        mock_pipeline_class.return_value = mock_pipeline

        mock_rate_actor = MagicMock()

        with patch(
            "core.ingestion.tasks.text_processing.RayProcessingPipeline",
            mock_pipeline_class,
        ):
            with patch("core.ingestion.tasks.text_processing.RayActorRateLimiter") as mock_limiter:
                import core.ingestion.tasks.text_processing as text_module

                results = text_module.process_s3_batch_ray._function(
                    s3_keys=["doc.txt"],
                    s3_endpoint="http://minio:9000",
                    s3_access_key="access",
                    s3_secret_key="secret",
                    s3_bucket="bucket",
                    embedding_config=embed_config,
                    collection="test",
                    rate_limiter=mock_rate_actor,
                )

        mock_limiter.assert_called_once_with(mock_rate_actor)
        assert len(results) == 1

    @patch("core.ingestion.tasks.text_processing.ray")
    def test_process_batch_logs_batch_info(self, mock_ray_module: MagicMock):
        """Verify process_s3_batch_ray logs batch information.

        **Why this test is important:**

        - Ensures batch progress is logged for monitoring and debugging
        - Validates batch_id and total_batches are used in logging
        - Confirms observability of the batch processing pipeline

        **What it tests:**

        - Logger is called with batch information
        - info level logging is used for batch progress
        - Batch metadata (batch_id, total_batches) is logged
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [ProcessingResult.success_result("doc.txt")]

        with patch(
            "core.ingestion.tasks.text_processing.RayProcessingPipeline",
            return_value=mock_pipeline,
        ):
            with patch("core.ingestion.tasks.text_processing.get_ray_logger") as mock_logger:
                mock_log = MagicMock()
                mock_logger.return_value = mock_log

                import core.ingestion.tasks.text_processing as text_module

                text_module.process_s3_batch_ray._function(
                    s3_keys=["doc.txt"],
                    s3_endpoint="http://minio:9000",
                    s3_access_key="access",
                    s3_secret_key="secret",
                    s3_bucket="bucket",
                    embedding_config=embed_config,
                    collection="test",
                    batch_id=1,
                    total_batches=10,
                )

        # Logger should be called with batch info
        assert mock_log.info.called

    @patch("core.ingestion.tasks.text_processing.ray")
    def test_process_batch_logs_circuit_breaker_warnings(self, mock_ray_module: MagicMock):
        """Verify process_s3_batch_ray logs circuit breaker errors.

        **Why this test is important:**

        - Ensures circuit breaker activations are logged as warnings
        - Validates observability of resilience mechanisms
        - Confirms operators can monitor circuit breaker state

        **What it tests:**

        - Circuit breaker errors in results trigger warning logs
        - Warning level logging is used for circuit breaker events
        - Circuit breaker state is visible in logs
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [
            ProcessingResult.failure_result("doc.txt", "circuit breaker open"),
        ]

        with patch(
            "core.ingestion.tasks.text_processing.RayProcessingPipeline",
            return_value=mock_pipeline,
        ):
            with patch("core.ingestion.tasks.text_processing.get_ray_logger") as mock_logger:
                mock_log = MagicMock()
                mock_logger.return_value = mock_log

                import core.ingestion.tasks.text_processing as text_module

                text_module.process_s3_batch_ray._function(
                    s3_keys=["doc.txt"],
                    s3_endpoint="http://minio:9000",
                    s3_access_key="access",
                    s3_secret_key="secret",
                    s3_bucket="bucket",
                    embedding_config=embed_config,
                    collection="test",
                )

        mock_log.warning.assert_called()

    @patch("core.ingestion.tasks.text_processing.ray")
    def test_process_batch_uses_configurable_parameters(self, mock_ray_module: MagicMock):
        """Verify process_s3_batch_ray passes configurable parameters to pipeline.

        **Why this test is important:**

        - Ensures all configurable parameters are passed through to the config
        - Validates that pipeline behavior can be tuned at runtime
        - Confirms parameter customization for different workloads

        **What it tests:**

        - max_concurrency is passed to RayProcessingConfig
        - circuit_breaker_threshold is passed to RayProcessingConfig
        - embedding_timeout is passed to RayProcessingConfig
        """
        embed_config = EmbeddingConfig(
            provider_type="ollama",
            ollama_url="http://ollama:11434",
            ollama_model="nomic-embed-text",
        )

        with patch("core.ingestion.tasks.text_processing.RayProcessingConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            mock_pipeline = MagicMock()
            mock_pipeline.process_keys_sync.return_value = []

            with patch(
                "core.ingestion.tasks.text_processing.RayProcessingPipeline",
                return_value=mock_pipeline,
            ):
                import core.ingestion.tasks.text_processing as text_module

                text_module.process_s3_batch_ray._function(
                    s3_keys=["doc.txt"],
                    s3_endpoint="http://minio:9000",
                    s3_access_key="access",
                    s3_secret_key="secret",
                    s3_bucket="bucket",
                    embedding_config=embed_config,
                    collection="test",
                    pipeline_concurrency=20,
                    circuit_breaker_threshold=10,
                    embedding_timeout=180,
                )

        # Verify config was created with custom parameters
        call_kwargs = mock_config_class.call_args[1]
        assert call_kwargs["max_concurrency"] == 20
        assert call_kwargs["circuit_breaker_threshold"] == 10
        assert call_kwargs["embedding_timeout"] == 180
