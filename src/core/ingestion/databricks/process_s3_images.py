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
from logging.config import dictConfig

import ray
from botocore.exceptions import ClientError

from clients.s3 import S3ClientWrapper
from config import ImageEmbeddingConfig, MinIOConfig, RayJobConfig, resolve_vector_db_provider
from core.ingestion.interfaces.operations import ImageContentFetcher
from core.ingestion.tasks import process_image_batch_ray
from core.ingestion.strategies import DatabricksStrategy
from foundation.logger import LOGGING_CONFIG

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.ray.databricks")


def _apply_python_params(args: list[str]) -> None:
    """Apply KEY=VALUE args (Databricks python_params) to the environment."""
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        if not key or not key.isupper() or not key.replace("_", "").isalnum():
            continue
        os.environ[key] = value


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

    vector_db_provider = resolve_vector_db_provider()

    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
    s3_prefix = os.environ.get("S3_PREFIX") or (
        sys.argv[1] if len(sys.argv) > 1 and not sys.argv[0].endswith("uvicorn") else "images/"
    )
    collection = os.environ.get("VECTOR_DB_COLLECTION", "documents")

    ray_cfg = RayJobConfig.from_env(namespace)
    minio_cfg = MinIOConfig.from_env(namespace)
    embed_cfg = ImageEmbeddingConfig.from_env(namespace)
    bucket = minio_cfg.bucket

    job_logger.info(
        "Configuration loaded",
        extra={
            "namespace": namespace,
            "s3_bucket": bucket,
            "s3_prefix": s3_prefix,
            "collection": collection,
            "vector_db_provider": vector_db_provider,
            "num_workers": ray_cfg.num_workers,
            "image_batch_size": ray_cfg.image_batch_size,
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

        if not keys:
            job_logger.info("No image objects to process")
            return

        image_batch_size = ray_cfg.image_batch_size
        image_embed_batch_size = ray_cfg.image_embed_batch_size
        key_batches = [keys[i : i + image_batch_size] for i in range(0, len(keys), image_batch_size)]
        num_batches = len(key_batches)

        job_logger.info(
            "Starting image batch processing: %d images in %d batches",
            len(keys),
            num_batches,
            extra={
                "total_keys": len(keys),
                "num_batches": num_batches,
                "image_batch_size": image_batch_size,
                "image_embed_batch_size": image_embed_batch_size,
            },
        )

        task_fn = process_image_batch_ray.options(
            num_cpus=ray_cfg.task_num_cpus,
            max_retries=ray_cfg.task_max_retries,
        )

        futures = [
            task_fn.remote(
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
            )
            for batch in key_batches
        ]

        results: list[tuple[str, bool, str]] = []
        completed_keys = 0
        wait_batch = ray_cfg.wait_batch_size
        wait_timeout = ray_cfg.wait_timeout

        while futures:
            ready, not_ready = ray.wait(
                futures,
                num_returns=min(wait_batch, len(futures)),
                timeout=wait_timeout,
            )
            futures = not_ready
            batch_results = ray.get(ready)
            for batch_result in batch_results:
                results.extend(batch_result)
                completed_keys += len(batch_result)
            if batch_results:
                success_so_far = sum(1 for _, ok, _ in results if ok)
                failed_so_far = len(results) - success_so_far
                job_logger.info(
                    "Image batch progress: %d/%d completed (%d ok, %d failed)",
                    completed_keys,
                    len(keys),
                    success_so_far,
                    failed_so_far,
                    extra={
                        "completed_keys": completed_keys,
                        "total_keys": len(keys),
                        "successful": success_so_far,
                        "failed": failed_so_far,
                        "remaining_keys": len(keys) - completed_keys,
                    },
                )

        success = sum(1 for _, ok, _ in results if ok)
        failed = len(results) - success
        elapsed = round(time.time() - start, 2)
        rate = round(len(results) / elapsed, 2) if elapsed > 0 else 0
        job_logger.info(
            "Databricks Ray image job complete: %d successful, %d failed in %.2fs (%.2f images/s)",
            success,
            failed,
            elapsed,
            rate,
            extra={
                "successful": success,
                "failed": failed,
                "total": len(results),
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
