"""Databricks Ray job: S3 images → CLIP embeddings → vector DB image collections.

This entrypoint initializes Ray on a Databricks cluster (via ray.util.spark),
then runs a Databricks-specific image ingestion pipeline. Mirrors the structure
of ray/process_s3_images.py but uses Databricks cluster initialization.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import nullcontext
from logging.config import dictConfig
from typing import Any

from clients.qdrant import QdrantClientWrapper
from clients.s3 import S3ClientWrapper
from config import EmbeddingConfig, MinIOConfig, RayJobConfig, VectorDBConfig
from core.ingestion.databricks.batch_runner import run_ray_batch_processing
from core.ingestion.databricks.runtime import (
    apply_python_params as _apply_python_params,
)
from core.ingestion.shared.batching import clear_successful_dlq_entries, iter_dlq_entries, iter_image_batches
from core.ingestion.shared.qdrant_indexing import qdrant_indexing_disabled
from core.ingestion.strategies import DatabricksStrategy
from core.ingestion.tasks import process_image_batch_ray
from foundation.checkpoint import CheckpointManager, is_s3_path
from foundation.logger import LOGGING_CONFIG

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.ray.databricks")

PIPELINE_LABEL = "databricks ingestion pipeline"


def main() -> None:
    """Process S3 image objects using Ray on Databricks.

    Reads S3_BUCKET, S3_PREFIX, VECTOR_DB_COLLECTION from environment (set by
    job submission or Databricks python_params). Lists S3 keys, filters by image
    extensions, and runs the image pipeline on Databricks Ray workers.
    """
    job_logger = logging.getLogger("pipeline.ray.job")
    job_logger.info("Databricks Ray image job started", extra={"pid": os.getpid()})
    start = time.time()

    _apply_python_params(sys.argv[1:])

    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")

    ray_cfg = RayJobConfig.from_env(namespace)
    minio_cfg = MinIOConfig.from_env(namespace)
    embed_cfg = EmbeddingConfig.from_env(namespace)
    vector_cfg = VectorDBConfig.from_env(namespace)

    s3_prefix = ray_cfg.s3_prefix
    collection = vector_cfg.collection
    bucket = minio_cfg.bucket
    image_max_items = ray_cfg.image_max_items
    image_page_size = ray_cfg.image_page_size

    job_logger.info(
        "Configuration loaded",
        extra={
            "namespace": namespace,
            "s3_bucket": bucket,
            "s3_prefix": s3_prefix,
            "collection": collection,
            "num_workers": ray_cfg.num_workers,
            "image_batch_size": ray_cfg.image_batch_size,
            "image_max_items": image_max_items,
            "image_page_size": image_page_size,
        },
    )

    # Use DatabricksStrategy for cluster initialization
    strategy = DatabricksStrategy.from_env(namespace)
    strategy.init()

    try:
        s3 = S3ClientWrapper(
            endpoint_url=minio_cfg.endpoint_url,
            access_key_id=minio_cfg.access_key_id,
            secret_access_key=minio_cfg.secret_access_key,
        )

        # Load checkpoint if enabled
        processed: set[str] = set()
        checkpoint_path: str | None = None
        checkpoint_manager = CheckpointManager(
            s3_client=s3 if is_s3_path(ray_cfg.checkpoint_dir) else None,
        )
        if ray_cfg.checkpoint_enabled:
            checkpoint_path = f"{ray_cfg.checkpoint_dir}/{collection}.json"
            processed = checkpoint_manager.load(checkpoint_path)

        image_batch_size = ray_cfg.image_batch_size
        image_embed_batch_size = ray_cfg.image_embed_batch_size
        max_inflight_batches = max(1, ray_cfg.num_workers)

        processing_dlq = os.getenv("PULL_FROM_DLQ", "").lower() == "true"
        if processing_dlq:
            logger.info("Processing previously failed image ingestions from the dead letter queue")
            batch_gen = iter_dlq_entries(
                batch_size=image_batch_size,
                max_items=image_max_items,
            )
        else:
            batch_gen = iter_image_batches(
                s3=s3,
                bucket=bucket,
                prefix=s3_prefix,
                processed=processed,
                batch_size=image_batch_size,
                max_items=image_max_items,
                page_size=image_page_size,
            )

        job_logger.info(
            "Starting streaming image batch processing",
            extra={
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
                embedding_config=embed_cfg,
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
            )

        should_disable_indexing = ray_cfg.disable_indexing_during_ingest
        qdrant_wrapper = QdrantClientWrapper.from_config(vector_cfg) if should_disable_indexing else None

        with (
            qdrant_indexing_disabled(
                client=qdrant_wrapper,
                collection=collection,
                vector_size=embed_cfg.clip_vector_size,
            )
            if should_disable_indexing
            else nullcontext()
        ):
            stats = run_ray_batch_processing(
                batches=batch_gen,
                submit_batch=_submit_batch,
                wait_batch_size=ray_cfg.wait_batch_size,
                wait_timeout=ray_cfg.wait_timeout,
                max_inflight_batches=max_inflight_batches,
                job_logger=job_logger,
                progress_label="Image batch progress",
                pipeline=PIPELINE_LABEL,
                total_expected_records=None,
            )
        if processing_dlq:
            clear_successful_dlq_entries(stats.successful_keys)

        success = stats.successful
        failed = stats.failed

        # Save checkpoint if enabled
        if ray_cfg.checkpoint_enabled and checkpoint_path:
            processed.update(stats.successful_keys)
            checkpoint_manager.save(checkpoint_path, processed, PIPELINE_LABEL)

        elapsed = round(time.time() - start, 2)
        rate = round(stats.completed_records / elapsed, 2) if elapsed > 0 else 0
        job_logger.info(
            "Databricks Ray image job complete: %d successful, %d failed in %.2fs (%.2f images/s)",
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
        job_logger.error(
            "Unexpected error in Databricks Ray image job: %s",
            e,
            extra={"error": str(e)},
            exc_info=True,
        )
        raise
    finally:
        strategy.shutdown()


if __name__ == "__main__":
    main()
