"""Ray job: S3 → Ollama embeddings → Qdrant.

This script processes S3 objects using Ray for parallel execution.
It mirrors the Spark implementation but uses Ray's distributed task execution.

Main responsibilities:
- Initialize Ray cluster
- List S3 objects
- Load checkpoint (if enabled)
- Distribute processing across Ray workers
- Collect results and save checkpoint
"""

import logging
import os
import sys
import time
from logging.config import dictConfig

import ray
from botocore.exceptions import ClientError

from clients.s3 import S3ClientWrapper
from config import EmbeddingConfig, MinIOConfig, RayJobConfig, VectorDBConfig
from core.ingestion.checkpoint import CheckpointManager, is_s3_path
from foundation.logger import LOGGING_CONFIG

from .processing import process_s3_batch_ray
from .rate_limiter import RateLimiterActor
from .ray_cluster import init_ray_cluster, shutdown_ray_cluster

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.ray")


def main() -> None:
    """Process S3 objects and store embeddings in Qdrant using Ray.

    This function initializes Ray, lists S3 objects, loads checkpoints,
    and processes objects using Ray remote functions.
    """
    job_logger = logging.getLogger("pipeline.ray.job")
    job_logger.info("Ray job started", extra={"pid": os.getpid()})
    start = time.time()
    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
    # Prefer environment variable over sys.argv to support in-process execution
    # When called in-process, sys.argv contains uvicorn args, not script args
    s3_prefix = os.environ.get("S3_PREFIX") or (
        sys.argv[1] if len(sys.argv) > 1 and not sys.argv[0].endswith("uvicorn") else "inputs/"
    )

    # Load configuration
    ray_cfg = RayJobConfig.from_env(namespace)
    minio_cfg = MinIOConfig.from_env(namespace)
    vector_cfg = VectorDBConfig.from_env(namespace)
    embed_cfg = EmbeddingConfig.from_env(namespace)

    job_logger.info(
        "Configuration loaded",
        extra={
            "namespace": namespace,
            "s3_prefix": s3_prefix,
            "num_workers": ray_cfg.num_workers,
        },
    )

    # Initialize Ray cluster
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
        # Initialize S3 client
        s3 = S3ClientWrapper(
            endpoint_url=minio_cfg.endpoint_url,
            access_key_id=minio_cfg.access_key_id,
            secret_access_key=minio_cfg.secret_access_key,
        )

        # List S3 objects
        try:
            keys = s3.list_objects(bucket=minio_cfg.bucket, prefix=s3_prefix)
            job_logger.info("S3 objects listed", extra={"count": len(keys)})
        except ClientError as e:
            job_logger.exception("Failed to list S3 objects", extra={"error": str(e)})
            sys.exit(1)

        if not keys:
            job_logger.info("No objects to process")
            return

        # Load checkpoint if enabled
        processed: set[str] = set()
        checkpoint_path: str | None = None
        checkpoint_manager = CheckpointManager(s3_client=s3 if is_s3_path(ray_cfg.checkpoint_dir) else None)
        if ray_cfg.checkpoint_enabled:
            # Use the same path format regardless of S3 or local
            checkpoint_path = f"{ray_cfg.checkpoint_dir}/{vector_cfg.collection}.json"
            processed = checkpoint_manager.load(checkpoint_path)
            keys = [k for k in keys if k not in processed]

        if not keys:
            job_logger.info("No new objects to process")
            return

        # Determine batch size based on configuration
        # For single-process mode (num_workers=0), use smaller batches
        # For distributed mode, use larger batches
        s3_batch_size = 100 if ray_cfg.num_workers > 0 else 50
        embed_batch_size = ray_cfg.embed_batch_max
        qdrant_batch_size = ray_cfg.batch_upsert_size

        job_logger.info(
            "Starting batch processing",
            extra={
                "total_keys": len(keys),
                "num_workers": ray_cfg.num_workers,
                "s3_batch_size": s3_batch_size,
                "embed_batch_size": embed_batch_size,
                "qdrant_batch_size": qdrant_batch_size,
            },
        )

        # Create rate limiter actor for Ollama requests
        ollama_rps = ray_cfg.ollama_requests_per_second
        rate_limiter = RateLimiterActor.remote(rate_per_sec=ollama_rps)

        # Split keys into batches for batch processing
        key_batches = [keys[i : i + s3_batch_size] for i in range(0, len(keys), s3_batch_size)]

        # Submit batch processing tasks
        futures = [
            process_s3_batch_ray.remote(
                s3_keys=batch,
                s3_endpoint=minio_cfg.endpoint_url,
                s3_access_key=minio_cfg.access_key_id,
                s3_secret_key=minio_cfg.secret_access_key,
                s3_bucket=minio_cfg.bucket,
                embedding_config=embed_cfg,
                collection=vector_cfg.collection,
                embed_batch_size=embed_batch_size,
                qdrant_batch_size=qdrant_batch_size,
                rate_limiter=rate_limiter,
            )
            for batch in key_batches
        ]

        # Collect results with progress tracking
        results: list[tuple[str, bool, str]] = []
        total_keys = len(keys)
        completed_keys = 0
        while futures:
            # Get results in batches
            ready, not_ready = ray.wait(futures, num_returns=min(10, len(futures)), timeout=1.0)
            futures = not_ready
            batch_results = ray.get(ready)

            # Flatten batch results (each batch returns list of tuples)
            for batch_result in batch_results:
                results.extend(batch_result)
                completed_keys += len(batch_result)

            # Log progress every 1000 keys or at completion
            if completed_keys % 1000 == 0 or completed_keys == total_keys:
                job_logger.info(
                    "Processing progress",
                    extra={
                        "completed": completed_keys,
                        "total": total_keys,
                        "percent": round(100 * completed_keys / total_keys, 1),
                    },
                )

        # Calculate statistics
        success = sum(1 for _, ok, _ in results if ok)
        failed = len(results) - success

        # Save checkpoint if enabled
        if ray_cfg.checkpoint_enabled and checkpoint_path:
            processed.update(k for k, ok, _ in results if ok)
            checkpoint_manager.save(checkpoint_path, processed)

        elapsed = round(time.time() - start, 2)
        job_logger.info(
            "Ray job complete",
            extra={
                "successful": success,
                "failed": failed,
                "total": len(results),
                "elapsed_seconds": elapsed,
                "rate_per_sec": round(len(results) / elapsed, 2) if elapsed > 0 else 0,
            },
        )

    except Exception as e:
        job_logger.error("Unexpected error in Ray job", extra={"error": str(e)}, exc_info=True)
        raise
    finally:
        shutdown_ray_cluster()


if __name__ == "__main__":
    main()
