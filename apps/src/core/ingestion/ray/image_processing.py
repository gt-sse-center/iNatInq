"""Ray image processing pipeline: S3 → Preprocessing → CLIP → Vector DBs.

This module provides a parallel image pipeline that mirrors the text pipeline
structure: fetch images from S3, preprocess (resize for embedding), embed via
CLIP, and upsert to Qdrant and Weaviate image collections. Reuses rate limiter
and circuit breaker infrastructure.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import attrs
import ray  # type: ignore[import-untyped]
from qdrant_client.models import PointStruct

from clients.clip import CLIPClient
from clients.interfaces.vector_db import VectorDBProvider  # noqa: TC001
from clients.s3 import S3ClientWrapper
from clients.weaviate import WeaviateClientWrapper, WeaviateDataObject
from config import ImageEmbeddingConfig  # noqa: TC001
from core.ingestion.image_utils import resize_for_embedding
from core.ingestion.interfaces.factories import VectorDBConfigFactory, create_vector_db_provider
from core.ingestion.interfaces.operations import ImageContentFetcher
from core.ingestion.interfaces.types import ImageContentResult, ProcessingResult
from foundation.http import create_retry_session
from foundation.rate_limiter import RateLimiter  # noqa: TC001

from .processing import _RayActorRateLimiter, get_ray_logger

logger = get_ray_logger("ray.image_pipeline")

# Default max dimension for CLIP (e.g. ViT-B/32)
DEFAULT_IMAGE_PREPROCESS_MAX_SIZE = 224


# =============================================================================
# Image Processing Config
# =============================================================================


@attrs.define(frozen=True, slots=True)
class RayImageProcessingConfig:
    """Configuration for Ray image processing pipeline.

    Mirrors RayProcessingConfig but with image-specific settings: image
    embedding config, smaller batch sizes, and image collection names.
    """

    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    image_embedding_config: ImageEmbeddingConfig
    collection: str
    image_batch_size: int = 20
    image_embed_batch_size: int = 4
    namespace: str = attrs.field(factory=lambda: os.getenv("K8S_NAMESPACE", "ml-system"))
    # Resilience (reuse same as text pipeline)
    rate_limit_rps: int = 5
    max_concurrency: int = 10
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30
    embedding_timeout: int = 120
    upsert_timeout: int = 60
    retry_max_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0
    image_preprocess_max_size: int = DEFAULT_IMAGE_PREPROCESS_MAX_SIZE


# =============================================================================
# Image Processing Pipeline
# =============================================================================


class ImageProcessingPipeline:
    """Ray implementation of the image processing pipeline.

    Processes images through S3 → Preprocessing → CLIP Embedding → Vector DBs.
    Mirrors RayProcessingPipeline structure and reuses rate limiter and
    circuit breaker infrastructure.
    """

    def __init__(
        self,
        config: RayImageProcessingConfig,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            config: Ray image processing configuration.
            rate_limiter: Optional rate limiter for CLIP API calls.
        """
        self._config = config
        self._rate_limiter = rate_limiter

    @property
    def config(self) -> RayImageProcessingConfig:
        """Get the processing configuration."""
        return self._config

    def process_keys_sync(self, keys: list[str]) -> list[ProcessingResult]:
        """Process image S3 keys synchronously (for Ray remote functions).

        Args:
            keys: List of S3 object keys (images).

        Returns:
            List of ProcessingResult objects.
        """
        if not keys:
            return []

        s3 = S3ClientWrapper(
            endpoint_url=self._config.s3_endpoint,
            access_key_id=self._config.s3_access_key,
            secret_access_key=self._config.s3_secret_key,
        )
        session = create_retry_session()
        clip_client = CLIPClient.from_config(self._config.image_embedding_config, session=session)
        db_factory = VectorDBConfigFactory(self._config.namespace)
        qdrant_cfg, weaviate_cfg = db_factory.create_both()
        qdrant_db = create_vector_db_provider(qdrant_cfg)
        weaviate_db = create_vector_db_provider(weaviate_cfg)

        try:
            fetcher = ImageContentFetcher(s3, self._config.s3_bucket)
            images, fetch_failures = fetcher.fetch_all(keys, with_dimensions=True)

            if not images:
                return fetch_failures

            process_results = asyncio.run(
                self._process_images_async(images, clip_client, qdrant_db, weaviate_db)
            )
            return fetch_failures + process_results
        finally:
            qdrant_db.close()
            weaviate_db.close()
            session.close()

    async def _process_images_async(
        self,
        images: list[ImageContentResult],
        clip_client: CLIPClient,
        qdrant_db: VectorDBProvider,
        weaviate_db: VectorDBProvider,
    ) -> list[ProcessingResult]:
        """Preprocess, embed, and upsert images to vector DBs."""
        if not images:
            return []

        max_size = self._config.image_preprocess_max_size
        batch_size = self._config.image_embed_batch_size
        collection = self._config.collection

        # Preprocess: resize for embedding
        processed: list[tuple[ImageContentResult, bytes]] = []
        for img in images:
            try:
                resized = resize_for_embedding(img.image_bytes, max_size=max_size)
                processed.append((img, resized))
            except Exception as e:
                logger.warning(
                    "Image preprocessing failed",
                    extra={"s3_key": img.s3_key, "error": str(e), "error_type": type(e).__name__},
                )
                # Treat as failure for this key
                processed.append((img, img.image_bytes))  # fallback to original for embedding attempt

        # Embed in batches (with rate limit)
        all_vectors: list[list[float]] = []
        for i in range(0, len(processed), batch_size):
            batch = processed[i : i + batch_size]
            image_bytes_batch = [p[1] for p in batch]
            acq = getattr(self._rate_limiter, "acquire", None) or getattr(
                self._rate_limiter, "acquire_permission", None
            )
            if acq:
                await acq()
            try:
                vectors = await clip_client.embed_image_batch_async(image_bytes_batch)
                if vectors is None:
                    all_vectors.extend([[] for _ in batch])  # placeholder for failure
                else:
                    all_vectors.extend(vectors)
            except Exception as e:
                logger.warning(
                    "CLIP batch embed failed",
                    extra={"error": str(e), "batch_size": len(batch)},
                )
                all_vectors.extend([[] for _ in batch])

        # Build points and upsert
        qdrant_image_collection = f"{collection}_images"
        weaviate_image_class = WeaviateClientWrapper._collection_to_image_class_name(collection)
        vector_size = clip_client.vector_size

        # Ensure image collections exist
        ensure_qdrant = getattr(qdrant_db, "ensure_image_collection_async", None)
        ensure_weaviate = getattr(weaviate_db, "ensure_image_collection_async", None)
        if callable(ensure_qdrant):
            await ensure_qdrant(collection=collection, vector_size=vector_size)
        if callable(ensure_weaviate):
            await ensure_weaviate(collection=collection, vector_size=vector_size)

        # Build Qdrant points and Weaviate objects
        qdrant_points: list[PointStruct] = []
        weaviate_objects: list[WeaviateDataObject] = []
        valid_indices: list[int] = []

        for idx, (img, _) in enumerate(processed):
            if idx >= len(all_vectors) or not all_vectors[idx]:
                continue
            vec = all_vectors[idx]
            point_id = str(uuid.uuid4())
            payload = {
                "s3_key": img.s3_key,
                "s3_uri": img.s3_uri,
                "format": img.format,
                "width": img.width if img.width is not None else 0,
                "height": img.height if img.height is not None else 0,
                "thumbnail_key": "",  # optional, leave empty if not set
            }
            qdrant_points.append(PointStruct(id=point_id, vector=vec, payload=payload))
            weaviate_objects.append(WeaviateDataObject(uuid=point_id, properties=payload, vector=vec))
            valid_indices.append(idx)

        if not qdrant_points:
            return [
                ProcessingResult.failure_result(images[i].s3_key, "Embedding or point build failed")
                for i in range(len(images))
            ]

        # Upsert to both DBs
        try:
            await qdrant_db.batch_upsert_async(
                collection=qdrant_image_collection,
                points=qdrant_points,
                vector_size=vector_size,
            )
        except Exception as e:
            logger.exception(
                "Qdrant image batch upsert failed",
                extra={"error": str(e), "collection": qdrant_image_collection},
            )
            return [
                ProcessingResult.failure_result(images[i].s3_key, f"Qdrant upsert: {e}")
                for i in range(len(images))
            ]

        try:
            await weaviate_db.batch_upsert_async(
                collection=weaviate_image_class,
                points=weaviate_objects,
                vector_size=vector_size,
            )
        except Exception as e:
            logger.exception(
                "Weaviate image batch upsert failed",
                extra={"error": str(e), "class": weaviate_image_class},
            )
            return [
                ProcessingResult.failure_result(images[i].s3_key, f"Weaviate upsert: {e}")
                for i in range(len(images))
            ]

        # Success for all processed images
        success_keys = {images[i].s3_key for i in valid_indices}
        results: list[ProcessingResult] = []
        for img in images:
            if img.s3_key in success_keys:
                results.append(ProcessingResult.success_result(img.s3_key))
            else:
                results.append(ProcessingResult.failure_result(img.s3_key, "Not in valid embed batch"))
        return results


# =============================================================================
# Ray Remote Function
# =============================================================================


@ray.remote(num_cpus=1, max_retries=3)
def process_image_batch_ray(
    s3_keys: list[str],
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket: str,
    image_embedding_config: ImageEmbeddingConfig,
    collection: str,
    image_batch_size: int = 20,
    image_embed_batch_size: int = 4,
    rate_limiter: Any | None = None,
    pipeline_concurrency: int = 10,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_timeout: int = 30,
    embedding_timeout: int = 120,
    upsert_timeout: int = 60,
    retry_max_attempts: int = 3,
    retry_min_wait: float = 1.0,
    retry_max_wait: float = 10.0,
) -> list[tuple[str, bool, str]]:
    """Process a batch of images through the embedding pipeline.

    Ray remote function: S3 → Preprocessing → CLIP → Vector DBs (image collections).

    Args:
        s3_keys: List of S3 image keys to process.
        s3_endpoint: S3 endpoint URL.
        s3_access_key: S3 access key.
        s3_secret_key: S3 secret key.
        s3_bucket: S3 bucket name.
        image_embedding_config: CLIP/image embedding configuration.
        collection: Base collection name (image collections: {collection}_images, {Collection}Images).
        image_batch_size: S3 keys per task (already batched by caller).
        image_embed_batch_size: Images per CLIP API call.
        rate_limiter: Optional Ray actor for rate limiting.
        pipeline_concurrency: Max concurrent async operations.
        circuit_breaker_threshold: Failures before circuit opens.
        circuit_breaker_timeout: Circuit recovery timeout (seconds).
        embedding_timeout: Timeout for embedding requests (seconds).
        upsert_timeout: Timeout for vector DB upserts (seconds).
        retry_max_attempts: Max retry attempts.
        retry_min_wait: Min wait between retries (seconds).
        retry_max_wait: Max wait between retries (seconds).

    Returns:
        List of (s3_key, success, error_message) tuples.
    """
    task_logger = get_ray_logger("ray.task")
    task_logger.info("Processing image batch of %d keys", len(s3_keys))

    namespace = os.getenv("K8S_NAMESPACE", "ml-system")

    config = RayImageProcessingConfig(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        image_embedding_config=image_embedding_config,
        collection=collection,
        image_batch_size=image_batch_size,
        image_embed_batch_size=image_embed_batch_size,
        namespace=namespace,
        max_concurrency=pipeline_concurrency,
        circuit_breaker_threshold=circuit_breaker_threshold,
        circuit_breaker_timeout=circuit_breaker_timeout,
        embedding_timeout=embedding_timeout,
        upsert_timeout=upsert_timeout,
        retry_max_attempts=retry_max_attempts,
        retry_min_wait=retry_min_wait,
        retry_max_wait=retry_max_wait,
    )

    local_rate_limiter = None
    if rate_limiter is not None:
        local_rate_limiter = _RayActorRateLimiter(rate_limiter)

    pipeline = ImageProcessingPipeline(config, local_rate_limiter)
    results = pipeline.process_keys_sync(s3_keys)

    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes
    task_logger.info("Image batch complete: %d succeeded, %d failed", successes, failures)

    return [r.to_tuple() for r in results]
