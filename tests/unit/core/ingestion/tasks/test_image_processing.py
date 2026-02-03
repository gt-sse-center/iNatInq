"""Unit tests for core.ingestion.tasks.image_processing module.

This module contains comprehensive tests for the image processing pipeline,
including configuration validation, synchronous and asynchronous processing,
and the Ray distributed processing function.

# Test Coverage

- RayImageProcessingConfig dataclass initialization and defaults
- ImageProcessingPipeline initialization and synchronous processing
- Async image processing with various failure scenarios
- process_image_batch_ray remote function behavior

# Running Tests

    pytest tests/unit/core/ingestion/tasks/test_image_processing.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import ImageEmbeddingConfig
from core.ingestion.interfaces.types import ImageContentResult, ProcessingResult
from core.ingestion.tasks.image_processing import (
    DEFAULT_IMAGE_PREPROCESS_MAX_SIZE,
    ImageProcessingPipeline,
    RayImageProcessingConfig,
    process_image_batch_ray,
)


class TestRayImageProcessingConfig:
    """Tests for RayImageProcessingConfig dataclass."""

    def test_creates_config_with_all_fields(self):
        """Verify that RayImageProcessingConfig is created with all required fields.

        **Why this test is important:**
        - Ensures the config dataclass accepts all required parameters
        - Validates that field values are stored correctly
        - Confirms the API contract for config creation

        **What it tests:**
        - Creating config with S3 credentials and endpoint
        - Setting image embedding configuration
        - Collection name assignment
        """
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )
        config = RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="bucket",
            image_embedding_config=embed_config,
            collection="test-collection",
        )
        assert config.s3_endpoint == "http://minio:9000"
        assert config.s3_bucket == "bucket"
        assert config.collection == "test-collection"

    def test_config_has_default_settings(self):
        """Verify that RayImageProcessingConfig has sensible defaults.

        **Why this test is important:**
        - Ensures users can create configs without specifying every parameter
        - Documents the expected default values for the codebase
        - Prevents accidental changes to default behavior

        **What it tests:**
        - Default image_batch_size is 20
        - Default image_embed_batch_size is 4
        - Default rate_limit_rps is 5
        - Default max_concurrency is 10
        - Default image_preprocess_max_size matches CLIP requirements
        """
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )
        config = RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="bucket",
            image_embedding_config=embed_config,
            collection="test",
        )
        assert config.image_batch_size == 20
        assert config.image_embed_batch_size == 4
        assert config.rate_limit_rps == 5
        assert config.max_concurrency == 10
        assert config.image_preprocess_max_size == DEFAULT_IMAGE_PREPROCESS_MAX_SIZE

    def test_default_image_preprocess_size_is_224(self):
        """Verify DEFAULT_IMAGE_PREPROCESS_MAX_SIZE is 224 for CLIP.

        **Why this test is important:**
        - CLIP models expect 224x224 images; wrong size causes errors or poor embeddings
        - Documents the expected constant value
        - Guards against accidental changes to this critical parameter

        **What it tests:**
        - The DEFAULT_IMAGE_PREPROCESS_MAX_SIZE constant equals 224
        """
        assert DEFAULT_IMAGE_PREPROCESS_MAX_SIZE == 224


class TestImageProcessingPipeline:
    """Tests for ImageProcessingPipeline class."""

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )
        return RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="test-bucket",
            image_embedding_config=embed_config,
            collection="test-collection",
            image_embed_batch_size=2,
        )

    def test_init_stores_config(self, config):
        """Verify pipeline stores config and rate limiter on initialization.

        **Why this test is important:**
        - Ensures dependency injection works correctly
        - Validates that the pipeline can access its configuration
        - Confirms rate limiter is properly stored for later use

        **What it tests:**
        - Config is accessible via pipeline.config
        - Rate limiter is stored in pipeline._rate_limiter
        """
        rate_limiter = MagicMock()
        pipeline = ImageProcessingPipeline(config, rate_limiter)
        assert pipeline.config is config
        assert pipeline._rate_limiter is rate_limiter

    def test_config_property_returns_config(self, config):
        """Verify config property returns the configuration.

        **Why this test is important:**
        - Ensures the config property provides access to pipeline settings
        - Validates encapsulation works correctly

        **What it tests:**
        - The config property returns the same config passed to constructor
        """
        pipeline = ImageProcessingPipeline(config)
        assert pipeline.config is config

    def test_process_keys_sync_returns_empty_for_empty_input(self, config):
        """Verify process_keys_sync returns empty list for empty input.

        **Why this test is important:**
        - Empty input should be handled gracefully without errors
        - Prevents unnecessary processing overhead for no-op calls
        - Documents the expected behavior for edge cases

        **What it tests:**
        - Calling process_keys_sync with empty list returns empty list
        """
        pipeline = ImageProcessingPipeline(config)
        results = pipeline.process_keys_sync([])
        assert results == []

    def test_process_keys_sync_handles_fetch_failures(self, config):
        """Verify process_keys_sync returns failures when S3 fetch fails.

        **Why this test is important:**
        - S3 connectivity issues should not crash the pipeline
        - Users need clear error messages about fetch failures
        - Partial failures should be reported, not silently ignored

        **What it tests:**
        - Pipeline continues when S3 fetch fails for an image
        - Failure result is returned with appropriate error message
        - Error message contains details about the S3 error
        """
        pipeline = ImageProcessingPipeline(config)

        mock_s3 = MagicMock()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = (
            [],  # no images
            [ProcessingResult.failure_result("image.jpg", "S3 error")],
        )

        mock_db_factory = MagicMock()
        mock_db_factory.create_both.return_value = (MagicMock(), MagicMock())

        with patch("core.ingestion.tasks.image_processing.S3ClientWrapper", return_value=mock_s3):
            with patch("core.ingestion.tasks.image_processing.create_retry_session"):
                with patch("core.ingestion.tasks.image_processing.CLIPClient"):
                    with patch(
                        "core.ingestion.tasks.image_processing.VectorDBConfigFactory",
                        return_value=mock_db_factory,
                    ):
                        with patch(
                            "core.ingestion.tasks.image_processing.create_vector_db_provider"
                        ) as mock_create:
                            mock_db = MagicMock()
                            mock_db.close = MagicMock()
                            mock_create.return_value = mock_db
                            with patch(
                                "core.ingestion.tasks.image_processing.ImageContentFetcher",
                                return_value=mock_fetcher,
                            ):
                                results = pipeline.process_keys_sync(["image.jpg"])

        assert len(results) == 1
        assert results[0].success is False
        assert "S3 error" in results[0].error_message

    def test_process_keys_sync_processes_images(self, config):
        """Verify process_keys_sync processes images through pipeline.

        **Why this test is important:**
        - Validates the end-to-end synchronous processing flow
        - Ensures all components are wired together correctly
        - Confirms successful processing returns success results

        **What it tests:**
        - Images are fetched from S3
        - Images are processed through the async pipeline
        - Success results are returned for processed images
        """
        pipeline = ImageProcessingPipeline(config)

        mock_s3 = MagicMock()
        mock_session = MagicMock()
        mock_clip = MagicMock()

        image = ImageContentResult(
            s3_key="image.jpg",
            image_bytes=b"fake_image_data",
            format="jpeg",
            size_bytes=15,
        )
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_all.return_value = ([image], [])

        mock_db_factory = MagicMock()
        mock_db_factory.create_both.return_value = (MagicMock(), MagicMock())

        with patch("core.ingestion.tasks.image_processing.S3ClientWrapper", return_value=mock_s3):
            with patch(
                "core.ingestion.tasks.image_processing.create_retry_session", return_value=mock_session
            ):
                with patch("core.ingestion.tasks.image_processing.CLIPClient") as mock_clip_class:
                    mock_clip_class.from_config.return_value = mock_clip
                    with patch(
                        "core.ingestion.tasks.image_processing.VectorDBConfigFactory",
                        return_value=mock_db_factory,
                    ):
                        with patch(
                            "core.ingestion.tasks.image_processing.create_vector_db_provider"
                        ) as mock_create:
                            mock_db = MagicMock()
                            mock_db.close = MagicMock()
                            mock_create.return_value = mock_db
                            with patch(
                                "core.ingestion.tasks.image_processing.ImageContentFetcher",
                                return_value=mock_fetcher,
                            ):
                                with patch.object(
                                    pipeline,
                                    "_process_images_async",
                                    new_callable=AsyncMock,
                                    return_value=[ProcessingResult.success_result("image.jpg")],
                                ):
                                    results = pipeline.process_keys_sync(["image.jpg"])

        assert len(results) == 1
        assert results[0].success is True


class TestImageProcessingPipelineAsync:
    """Tests for async processing in ImageProcessingPipeline."""

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )
        return RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="access",
            s3_secret_key="secret",
            s3_bucket="test-bucket",
            image_embedding_config=embed_config,
            collection="test-collection",
            image_embed_batch_size=2,
        )

    @pytest.mark.asyncio
    async def test_process_images_async_returns_empty_for_no_images(self, config):
        """Verify _process_images_async returns empty for no images.

        **Why this test is important:**
        - Empty input should be handled efficiently in async context
        - Avoids unnecessary async operations for no-op calls
        - Documents edge case behavior

        **What it tests:**
        - Calling _process_images_async with empty list returns empty list
        """
        pipeline = ImageProcessingPipeline(config)
        mock_clip = MagicMock()
        mock_qdrant = MagicMock()
        mock_weaviate = MagicMock()
        results = await pipeline._process_images_async([], mock_clip, mock_qdrant, mock_weaviate)
        assert results == []

    @pytest.mark.asyncio
    async def test_process_images_async_handles_preprocessing_failure(self, config):
        """Verify _process_images_async handles image preprocessing failures gracefully.

        **Why this test is important:**
        - Corrupt or unusual images should not crash the pipeline
        - Pipeline should attempt fallback behavior when resize fails
        - Processing should continue even with preprocessing errors

        **What it tests:**
        - resize_for_embedding failure is caught
        - Pipeline continues with fallback behavior
        - Results are still returned for the image
        """
        pipeline = ImageProcessingPipeline(config)

        mock_clip = MagicMock()
        mock_clip.embed_image_batch_async = AsyncMock(return_value=[[0.1] * 512])
        mock_clip.vector_size = 512

        mock_qdrant = MagicMock()
        mock_qdrant.batch_upsert_async = AsyncMock()
        mock_qdrant.ensure_image_collection_async = AsyncMock()

        mock_weaviate = MagicMock()
        mock_weaviate.batch_upsert_async = AsyncMock()
        mock_weaviate.ensure_image_collection_async = AsyncMock()

        image = ImageContentResult(
            s3_key="image.jpg",
            image_bytes=b"invalid_image_data",
            format="jpeg",
            size_bytes=18,
        )

        with patch(
            "core.ingestion.tasks.image_processing.resize_for_embedding",
            side_effect=Exception("Resize failed"),
        ):
            # Should not raise, but use original image bytes as fallback
            results = await pipeline._process_images_async([image], mock_clip, mock_qdrant, mock_weaviate)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_process_images_async_handles_embed_failure(self, config):
        """Verify _process_images_async handles embedding failures.

        **Why this test is important:**
        - CLIP service failures should not crash the pipeline
        - Users need clear error reporting for embedding failures
        - Failed images should be marked as failures, not silently dropped

        **What it tests:**
        - CLIP embed_image_batch_async exception is caught
        - Result is marked as failure
        - Pipeline completes without raising exception
        """
        pipeline = ImageProcessingPipeline(config)

        mock_clip = MagicMock()
        mock_clip.embed_image_batch_async = AsyncMock(side_effect=Exception("CLIP error"))
        mock_clip.vector_size = 512

        mock_qdrant = MagicMock()
        mock_qdrant.ensure_image_collection_async = AsyncMock()
        mock_weaviate = MagicMock()
        mock_weaviate.ensure_image_collection_async = AsyncMock()

        image = ImageContentResult(
            s3_key="image.jpg",
            image_bytes=b"image_data",
            format="jpeg",
            size_bytes=10,
        )

        with patch("core.ingestion.tasks.image_processing.resize_for_embedding", return_value=b"resized"):
            results = await pipeline._process_images_async([image], mock_clip, mock_qdrant, mock_weaviate)

        assert len(results) == 1
        assert results[0].success is False

    @pytest.mark.asyncio
    async def test_process_images_async_handles_qdrant_upsert_failure(self, config):
        """Verify _process_images_async handles Qdrant upsert failures.

        **Why this test is important:**
        - Database failures should not crash the pipeline
        - Users need clear error messages identifying Qdrant as the failure point
        - Pipeline should handle vector DB errors gracefully

        **What it tests:**
        - Qdrant batch_upsert_async exception is caught
        - Result is marked as failure with Qdrant mentioned in error
        - Pipeline completes without raising exception
        """
        pipeline = ImageProcessingPipeline(config)

        mock_clip = MagicMock()
        mock_clip.embed_image_batch_async = AsyncMock(return_value=[[0.1] * 512])
        mock_clip.vector_size = 512

        mock_qdrant = MagicMock()
        mock_qdrant.batch_upsert_async = AsyncMock(side_effect=Exception("Qdrant error"))
        mock_qdrant.ensure_image_collection_async = AsyncMock()

        mock_weaviate = MagicMock()
        mock_weaviate.ensure_image_collection_async = AsyncMock()

        image = ImageContentResult(
            s3_key="image.jpg",
            image_bytes=b"image_data",
            format="jpeg",
            size_bytes=10,
            width=100,
            height=100,
        )

        with patch("core.ingestion.tasks.image_processing.resize_for_embedding", return_value=b"resized"):
            results = await pipeline._process_images_async([image], mock_clip, mock_qdrant, mock_weaviate)

        assert len(results) == 1
        assert results[0].success is False
        assert "Qdrant" in results[0].error_message


class TestProcessImageBatchRay:
    """Tests for process_image_batch_ray remote function."""

    def test_process_batch_returns_results(self, mock_ray):
        """Verify process_image_batch_ray processes a batch of images.

        **Why this test is important:**
        - Validates the Ray remote function correctly invokes the pipeline
        - Ensures results are properly formatted as tuples
        - Confirms both success and failure results are returned correctly

        **What it tests:**
        - Pipeline is invoked with S3 keys
        - Results are returned as (s3_key, success, error_message) tuples
        - Both successful and failed images are represented
        """
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [
            ProcessingResult.success_result("img1.jpg"),
            ProcessingResult.failure_result("img2.jpg", "failed"),
        ]

        with patch(
            "core.ingestion.tasks.image_processing.ImageProcessingPipeline",
            return_value=mock_pipeline,
        ):
            results = process_image_batch_ray(
                s3_keys=["img1.jpg", "img2.jpg"],
                s3_endpoint="http://minio:9000",
                s3_access_key="access",
                s3_secret_key="secret",
                s3_bucket="bucket",
                image_embedding_config=embed_config,
                collection="test",
            )

        assert len(results) == 2
        assert results[0] == ("img1.jpg", True, "")
        assert results[1][0] == "img2.jpg"
        assert results[1][1] is False

    def test_process_batch_with_rate_limiter(self, mock_ray):
        """Verify process_image_batch_ray uses rate limiter when provided.

        **Why this test is important:**
        - Rate limiting prevents overwhelming external services
        - Validates rate limiter actor is properly integrated
        - Ensures optional rate limiting can be enabled

        **What it tests:**
        - RayActorRateLimiter is created when rate_limiter actor is provided
        - Rate limiter is passed to the pipeline
        - Processing completes successfully with rate limiting
        """
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )

        mock_pipeline_class = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [ProcessingResult.success_result("img.jpg")]
        mock_pipeline_class.return_value = mock_pipeline

        mock_rate_actor = MagicMock()

        with patch(
            "core.ingestion.tasks.image_processing.ImageProcessingPipeline",
            mock_pipeline_class,
        ):
            with patch("core.ingestion.tasks.image_processing.RayActorRateLimiter") as mock_limiter:
                results = process_image_batch_ray(
                    s3_keys=["img.jpg"],
                    s3_endpoint="http://minio:9000",
                    s3_access_key="access",
                    s3_secret_key="secret",
                    s3_bucket="bucket",
                    image_embedding_config=embed_config,
                    collection="test",
                    rate_limiter=mock_rate_actor,
                )

        mock_limiter.assert_called_once_with(mock_rate_actor)
        assert len(results) == 1

    def test_process_batch_logs_info(self, mock_ray):
        """Verify process_image_batch_ray logs batch information.

        **Why this test is important:**
        - Logging is essential for debugging distributed Ray tasks
        - Validates observability of batch processing
        - Ensures operators can monitor processing progress

        **What it tests:**
        - Ray logger is obtained and used
        - Info-level logging occurs during processing
        """
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )

        mock_pipeline = MagicMock()
        mock_pipeline.process_keys_sync.return_value = [ProcessingResult.success_result("img.jpg")]

        with patch(
            "core.ingestion.tasks.image_processing.ImageProcessingPipeline",
            return_value=mock_pipeline,
        ):
            with patch("core.ingestion.tasks.image_processing.get_ray_logger") as mock_logger:
                mock_log = MagicMock()
                mock_logger.return_value = mock_log

                process_image_batch_ray(
                    s3_keys=["img.jpg"],
                    s3_endpoint="http://minio:9000",
                    s3_access_key="access",
                    s3_secret_key="secret",
                    s3_bucket="bucket",
                    image_embedding_config=embed_config,
                    collection="test",
                )

        assert mock_log.info.called

    def test_process_batch_uses_configurable_parameters(self, mock_ray):
        """Verify process_image_batch_ray passes configurable parameters to pipeline.

        **Why this test is important:**
        - Users need to tune concurrency, timeouts, and thresholds
        - Ensures configuration is properly propagated to the pipeline
        - Validates the parameter passing contract

        **What it tests:**
        - pipeline_concurrency is passed as max_concurrency
        - circuit_breaker_threshold is passed correctly
        - embedding_timeout is passed correctly
        """
        embed_config = ImageEmbeddingConfig(
            clip_model_id="openai/clip-vit-base-patch32",
            clip_service_url="http://clip:8000",
        )

        with patch("core.ingestion.tasks.image_processing.RayImageProcessingConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            mock_pipeline = MagicMock()
            mock_pipeline.process_keys_sync.return_value = []

            with patch(
                "core.ingestion.tasks.image_processing.ImageProcessingPipeline",
                return_value=mock_pipeline,
            ):
                process_image_batch_ray(
                    s3_keys=["img.jpg"],
                    s3_endpoint="http://minio:9000",
                    s3_access_key="access",
                    s3_secret_key="secret",
                    s3_bucket="bucket",
                    image_embedding_config=embed_config,
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
