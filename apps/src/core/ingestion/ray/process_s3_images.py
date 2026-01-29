"""Ray job: S3 images → CLIP embeddings → Qdrant/Weaviate image collections.

This script processes S3 image objects using Ray for parallel execution.
It lists objects under the given prefix, filters by image extensions
(.jpg, .jpeg, .png, .webp, .gif), batches keys, and runs the image
processing pipeline on Ray workers.
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
from config import ImageEmbeddingConfig, MinIOConfig, RayJobConfig
from core.ingestion.interfaces.operations import ImageContentFetcher
from foundation.logger import LOGGING_CONFIG

from .image_processing import process_image_batch_ray
from .ray_cluster import init_ray_cluster, shutdown_ray_cluster

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.ray")


def main() -> None:
    """Process S3 image objects and store embeddings in vector DB image collections.

    Reads S3_BUCKET, S3_PREFIX, VECTOR_DB_COLLECTION from environment (set by
    the job submission API). Lists S3 keys under the prefix, filters by image
    extensions, and runs the image pipeline on Ray workers.
    """
    job_logger = logging.getLogger("pipeline.ray.job")
    job_logger.info("Ray image job started", extra={"pid": os.getpid()})
    start = time.time()
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
            "num_workers": ray_cfg.num_workers,
            "image_batch_size": ray_cfg.image_batch_size,
        },
    )

    if ray is None:
        job_logger.error("Ray is not installed. Install with: pip install ray[default]")
        sys.exit(1)

    try:
        init_ray_cluster(ray_cfg)
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

        success = sum(1 for _, ok, _ in results if ok)
        failed = len(results) - success
        elapsed = round(time.time() - start, 2)
        rate = round(len(results) / elapsed, 2) if elapsed > 0 else 0
        job_logger.info(
            "Ray image job complete: %d successful, %d failed in %.2fs (%.2f images/s)",
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
        job_logger.error("Unexpected error in Ray image job: %s", e, extra={"error": str(e)}, exc_info=True)
        raise
    finally:
        shutdown_ray_cluster()


if __name__ == "__main__":
    main()
