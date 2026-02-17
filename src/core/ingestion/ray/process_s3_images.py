"""Ray job: S3 images → CLIP embeddings → Qdrant/Weaviate image collections.

This script processes S3 image objects using Ray for parallel execution.
It lists objects under the given prefix, filters for likely image keys
(by extension or extensionless acceptance), batches keys, and runs the
image processing pipeline on Ray workers.

Supports checkpointing so failed runs can resume from the last saved state.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from logging.config import dictConfig
from typing import Any

import ray
from botocore.exceptions import ClientError

from clients.s3 import S3ClientWrapper
from config import ImageEmbeddingConfig, MinIOConfig, RayJobConfig, VectorDBConfig
from core.ingestion.databricks.batch_runner import run_ray_batch_processing
from core.ingestion.interfaces.operations import ImageContentFetcher
from core.ingestion.strategies import LocalRayStrategy
from core.ingestion.tasks import process_image_batch_ray
from foundation.checkpoint import CheckpointManager, is_s3_path
from foundation.logger import LOGGING_CONFIG

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.ray")


def _parse_optional_positive_int_env(name: str) -> int | None:
    """Parse an optional positive integer from an environment variable."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except ValueError:
        return None


def main() -> None:
    """Process S3 image objects and store embeddings in vector DB image collections.

    Reads S3_BUCKET, S3_PREFIX, VECTOR_DB_COLLECTION from environment (set by
    the job submission API). Lists S3 keys under the prefix, filters for image
    keys, and runs the image pipeline on Ray workers.

    Supports checkpointing: previously processed keys are skipped on restart.
    """
    job_logger = logging.getLogger("pipeline.ray.job")
    job_logger.info("Ray image job started", extra={"pid": os.getpid()})
    start = time.time()
    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
    s3_prefix = os.environ.get("S3_PREFIX", "")
    if not s3_prefix:
        s3_prefix = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[0].endswith("uvicorn") else ""
    collection = os.environ.get("VECTOR_DB_COLLECTION", "documents")

    ray_cfg = RayJobConfig.from_env(namespace)
    minio_cfg = MinIOConfig.from_env(namespace)
    embed_cfg = ImageEmbeddingConfig.from_env(namespace)
    vector_cfg = VectorDBConfig.from_env(namespace)
    bucket = minio_cfg.bucket
    ingestion_targets = vector_cfg.ingestion_targets
    image_max_items = _parse_optional_positive_int_env("IMAGE_MAX_ITEMS")

    job_logger.info(
        "Configuration loaded",
        extra={
            "namespace": namespace,
            "s3_bucket": bucket,
            "s3_prefix": s3_prefix,
            "collection": collection,
            "num_workers": ray_cfg.num_workers,
            "image_batch_size": ray_cfg.image_batch_size,
            "ingestion_targets": sorted(ingestion_targets),
            "image_max_items": image_max_items,
        },
    )

    if ray is None:
        job_logger.error("Ray is not installed. Install with: pip install ray[default]")
        sys.exit(1)

    # Use LocalRayStrategy for cluster initialization
    strategy = LocalRayStrategy.from_env(namespace)
    try:
        strategy.init()
        job_logger.info("Ray cluster initialized")
    except (ImportError, RuntimeError, ValueError) as e:
        job_logger.error("Failed to initialize Ray cluster", extra={"error": str(e)}, exc_info=True)
        sys.exit(1)

    try:
        s3 = S3ClientWrapper(
            endpoint_url=minio_cfg.endpoint_url,
            access_key_id=minio_cfg.access_key_id,
            secret_access_key=minio_cfg.secret_access_key,
        )

        try:
            all_keys = s3.list_objects(bucket=bucket, prefix=s3_prefix)
            job_logger.info("S3 objects listed", extra={"count": len(all_keys)})
        except ClientError as e:
            job_logger.exception("Failed to list S3 objects", extra={"error": str(e)})
            sys.exit(1)

        keys = ImageContentFetcher.filter_image_keys(all_keys)
        job_logger.info(
            "Filtered to image keys",
            extra={"total_listed": len(all_keys), "image_keys": len(keys)},
        )

        # Load checkpoint if enabled
        processed: set[str] = set()
        checkpoint_path: str | None = None
        checkpoint_manager = CheckpointManager(
            s3_client=s3 if is_s3_path(ray_cfg.checkpoint_dir) else None,
        )
        if ray_cfg.checkpoint_enabled:
            checkpoint_path = f"{ray_cfg.checkpoint_dir}/{collection}_images.json"
            processed = checkpoint_manager.load(checkpoint_path)
            keys = [k for k in keys if k not in processed]

        # Apply image_max_items limit after checkpoint filtering
        if image_max_items is not None and len(keys) > image_max_items:
            keys = keys[:image_max_items]

        if not keys:
            job_logger.info("No image objects to process")
            return

        image_batch_size = ray_cfg.image_batch_size
        image_embed_batch_size = ray_cfg.image_embed_batch_size
        key_batches = [keys[i : i + image_batch_size] for i in range(0, len(keys), image_batch_size)]
        max_inflight_batches = max(1, ray_cfg.num_workers)

        job_logger.info(
            "Starting image batch processing: %d images in %d batches",
            len(keys),
            len(key_batches),
            extra={
                "total_keys": len(keys),
                "num_batches": len(key_batches),
                "image_batch_size": image_batch_size,
                "image_embed_batch_size": image_embed_batch_size,
                "max_inflight_batches": max_inflight_batches,
            },
        )

        task_fn = process_image_batch_ray.options(
            num_cpus=ray_cfg.task_num_cpus,
            max_retries=ray_cfg.task_max_retries,
        )

        def _submit_batch(batch: list[Any]) -> Any:
            return task_fn.remote(
                s3_keys=batch,
                s3_endpoint=minio_cfg.endpoint_url,
                s3_access_key=minio_cfg.access_key_id,
                s3_secret_key=minio_cfg.secret_access_key,
                s3_bucket=bucket,
                image_embedding_config=embed_cfg,
                collection=collection,
                image_batch_size=image_batch_size,
                image_embed_batch_size=image_embed_batch_size,
                rate_limiter=None,
                pipeline_concurrency=ray_cfg.pipeline_concurrency,
                circuit_breaker_threshold=ray_cfg.circuit_breaker_threshold,
                circuit_breaker_timeout=ray_cfg.circuit_breaker_timeout,
                embedding_timeout=ray_cfg.embedding_timeout,
                upsert_timeout=ray_cfg.upsert_timeout,
                retry_max_attempts=ray_cfg.retry_max_attempts,
                retry_min_wait=ray_cfg.retry_min_wait,
                retry_max_wait=ray_cfg.retry_max_wait,
                ingestion_targets=ingestion_targets,
            )

        stats = run_ray_batch_processing(
            batches=key_batches,
            submit_batch=_submit_batch,
            wait_batch_size=ray_cfg.wait_batch_size,
            wait_timeout=ray_cfg.wait_timeout,
            max_inflight_batches=max_inflight_batches,
            job_logger=job_logger,
            progress_label="Image batch progress",
            total_expected_records=len(keys),
        )

        success = stats.successful
        failed = stats.failed

        # Save checkpoint if enabled
        if ray_cfg.checkpoint_enabled and checkpoint_path:
            processed.update(stats.successful_keys)
            checkpoint_manager.save(checkpoint_path, processed)

        elapsed = round(time.time() - start, 2)
        rate = round(stats.completed_records / elapsed, 2) if elapsed > 0 else 0
        job_logger.info(
            "Ray image job complete: %d successful, %d failed in %.2fs (%.2f images/s)",
            success,
            failed,
            elapsed,
            rate,
            extra={
                "successful": success,
                "failed": failed,
                "total": stats.completed_records,
                "submitted_records": stats.submitted_records,
                "num_batches": stats.num_batches,
                "elapsed_seconds": elapsed,
                "rate_per_sec": rate,
            },
        )

    except Exception as e:
        job_logger.error("Unexpected error in Ray image job: %s", e, extra={"error": str(e)}, exc_info=True)
        raise
    finally:
        strategy.shutdown()


if __name__ == "__main__":
    main()
