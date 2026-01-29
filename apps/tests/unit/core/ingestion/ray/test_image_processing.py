"""Unit tests for core.ingestion.ray.image_processing module.

Tests the image processing pipeline: S3 → Preprocessing → CLIP → Vector DBs.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config import ImageEmbeddingConfig

from core.ingestion.interfaces.types import ImageContentResult, ProcessingResult
from core.ingestion.ray.image_processing import (
    DEFAULT_IMAGE_PREPROCESS_MAX_SIZE,
    ImageProcessingPipeline,
    RayImageProcessingConfig,
    process_image_batch_ray,
)


# =============================================================================
# Config Tests
# =============================================================================


class TestRayImageProcessingConfig:
    """Test suite for RayImageProcessingConfig."""

    def test_creates_config_with_required_fields(self) -> None:
        """Config is created with required fields and image-specific defaults."""
        image_embed = ImageEmbeddingConfig(
            provider_type="clip",
            clip_url="http://localhost:8000",
            clip_model="ViT-B/32",
        )
        config = RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="minioadmin",
            s3_secret_key="minioadmin",
            s3_bucket="images",
            image_embedding_config=image_embed,
            collection="documents",
        )
        assert config.s3_bucket == "images"
        assert config.collection == "documents"
        assert config.image_batch_size == 20
        assert config.image_embed_batch_size == 4
        assert config.image_preprocess_max_size == DEFAULT_IMAGE_PREPROCESS_MAX_SIZE

    def test_config_with_custom_image_batch_sizes(self) -> None:
        """Config accepts custom image batch sizes (smaller than text)."""
        image_embed = ImageEmbeddingConfig(
            provider_type="clip",
            clip_url="http://localhost:8000",
            clip_model="ViT-B/32",
        )
        config = RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_bucket="b",
            image_embedding_config=image_embed,
            collection="photos",
            image_batch_size=10,
            image_embed_batch_size=2,
        )
        assert config.image_batch_size == 10
        assert config.image_embed_batch_size == 2


# =============================================================================
# ImageProcessingPipeline Tests
# =============================================================================


class TestImageProcessingPipeline:
    """Test suite for ImageProcessingPipeline."""

    def test_process_keys_sync_returns_empty_for_empty_keys(self) -> None:
        """process_keys_sync returns empty list when keys is empty."""
        image_embed = ImageEmbeddingConfig(
            provider_type="clip",
            clip_url="http://localhost:8000",
            clip_model="ViT-B/32",
        )
        config = RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_bucket="b",
            image_embedding_config=image_embed,
            collection="documents",
        )
        pipeline = ImageProcessingPipeline(config)
        result = pipeline.process_keys_sync([])
        assert result == []

    def test_process_keys_sync_returns_fetch_failures_when_no_images_fetched(
        self,
    ) -> None:
        """When fetch_all returns no images, only fetch failures are returned."""
        image_embed = ImageEmbeddingConfig(
            provider_type="clip",
            clip_url="http://localhost:8000",
            clip_model="ViT-B/32",
        )
        config = RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_bucket="b",
            image_embedding_config=image_embed,
            collection="documents",
        )
        pipeline = ImageProcessingPipeline(config)

        with (
            patch("core.ingestion.ray.image_processing.S3ClientWrapper"),
            patch("core.ingestion.ray.image_processing.create_retry_session"),
            patch("core.ingestion.ray.image_processing.CLIPClient"),
            patch(
                "core.ingestion.ray.image_processing.create_vector_db_provider",
                side_effect=[MagicMock(), MagicMock()],
            ),
            patch("core.ingestion.ray.image_processing.VectorDBConfigFactory") as mock_factory,
        ):
            mock_factory.return_value.create_both.return_value = (
                MagicMock(),
                MagicMock(),
            )
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_all.return_value = (
                [],  # no images
                [
                    ProcessingResult.failure_result("a.jpg", "Image fetch/validation failed"),
                ],
            )
            with patch(
                "core.ingestion.ray.image_processing.ImageContentFetcher",
                return_value=mock_fetcher,
            ):
                result = pipeline.process_keys_sync(["a.jpg"])
        assert len(result) == 1
        assert result[0].s3_key == "a.jpg"
        assert result[0].success is False
        assert "fetch" in result[0].error_message.lower() or "validation" in result[0].error_message.lower()

    def test_config_property_returns_config(self) -> None:
        """config property returns the pipeline config."""
        image_embed = ImageEmbeddingConfig(
            provider_type="clip",
            clip_url="http://localhost:8000",
            clip_model="ViT-B/32",
        )
        config = RayImageProcessingConfig(
            s3_endpoint="http://minio:9000",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_bucket="b",
            image_embedding_config=image_embed,
            collection="documents",
        )
        pipeline = ImageProcessingPipeline(config)
        assert pipeline.config is config


# =============================================================================
# process_image_batch_ray Tests
# =============================================================================


class TestProcessImageBatchRay:
    """Test suite for process_image_batch_ray remote function."""

    def test_process_image_batch_ray_is_defined_and_callable(self) -> None:
        """process_image_batch_ray is a Ray remote function (callable)."""
        assert callable(process_image_batch_ray)
        # Ray decorator adds .remote for task submission
        assert hasattr(process_image_batch_ray, "remote")
