"""Databricks Ray job: S3 → embeddings → vector databases.

This entrypoint initializes Ray on a Databricks cluster (via ray.util.spark),
then runs a Databricks-specific ingestion pipeline with batch-level progress
logging for improved visibility in Databricks run logs.
"""

from __future__ import annotations

import inspect
import logging
from logging.config import dictConfig

import os
from pathlib import Path
import sys
import time

import ray
from botocore.exceptions import ClientError

from clients.s3 import S3ClientWrapper
from config import EmbeddingConfig, MinIOConfig, RayJobConfig, VectorDBConfig, resolve_vector_db_provider
from foundation.checkpoint import CheckpointManager, is_s3_path
from core.ingestion.tasks import process_s3_batch_ray
from core.ingestion.shared import RateLimiterActor
from foundation.logger import LOGGING_CONFIG

from ray.util.spark import setup_ray_cluster
from ray.util.spark import MAX_NUM_WORKER_NODES


logger = logging.getLogger("pipeline.ray.databricks")
dictConfig(LOGGING_CONFIG)


def _setup_ray_cluster(config: RayJobConfig) -> object | None:
    """Start Ray on Databricks and return the cluster handle when possible."""
    if setup_ray_cluster is None:
        raise RuntimeError("ray.util.spark is required to start Ray on Databricks.")

    kwargs = {
        "num_worker_nodes": config.num_workers,
        "cpus_per_node": int(config.worker_cpus),
        "memory_per_node": config.worker_memory,
    }
    signature = inspect.signature(setup_ray_cluster)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters and value}

    logger.info("Initializing Ray on Databricks", extra={"params": filtered})
    return setup_ray_cluster(
        max_worker_nodes=MAX_NUM_WORKER_NODES,
    )


def _init_ray(config: RayJobConfig, ray_cluster: object | None) -> None:
    """Initialize Ray client connection to the Databricks cluster."""
    address = getattr(ray_cluster, "address", None) if ray_cluster is not None else None
    runtime_env: dict[str, object] = {}
    env_vars: dict[str, str] = {}
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath

    passthrough_keys = (
        "K8S_NAMESPACE",
        "S3_PREFIX",
        "S3_ENDPOINT",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_BUCKET",
        "S3_REGION",
        "S3_USE_SSL",
        "S3_PATH_STYLE",
        "S3_TIMEOUT",
        "S3_MAX_RETRIES",
        "S3_RETRY_MIN_WAIT",
        "S3_RETRY_MAX_WAIT",
        "S3_CIRCUIT_BREAKER_THRESHOLD",
        "S3_CIRCUIT_BREAKER_TIMEOUT",
        "VECTOR_DB_PROVIDER",
        "VECTOR_DB_COLLECTION",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_TIMEOUT",
        "QDRANT_CIRCUIT_BREAKER_THRESHOLD",
        "QDRANT_CIRCUIT_BREAKER_TIMEOUT",
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "WEAVIATE_GRPC_HOST",
        "WEAVIATE_GRPC_PORT",
        "WEAVIATE_TIMEOUT",
        "WEAVIATE_CIRCUIT_BREAKER_THRESHOLD",
        "WEAVIATE_CIRCUIT_BREAKER_TIMEOUT",
        "EMBEDDING_PROVIDER",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT",
        "OLLAMA_MAX_RETRIES",
        "OLLAMA_RETRY_MIN_WAIT",
        "OLLAMA_RETRY_MAX_WAIT",
        "OLLAMA_CIRCUIT_BREAKER_THRESHOLD",
        "OLLAMA_CIRCUIT_BREAKER_TIMEOUT",
        "VECTOR_DB_TARGETS",
        "INATINQ_SRC_DIR",
    )
    for key in passthrough_keys:
        value = os.environ.get(key)
        if value is not None and value != "":
            env_vars[key] = value
    env_vars["VECTOR_DB_PROVIDER"] = resolve_vector_db_provider()

    if env_vars:
        runtime_env["env_vars"] = env_vars
    workspace_src = os.environ.get("INATINQ_SRC_DIR")
    if not workspace_src:
        raise RuntimeError(
            "INATINQ_SRC_DIR is not set. Please export INATINQ_SRC_DIR to the repo src path "
            "(e.g. /Workspace/Users/<user>/iNatInq/src) before running this job."
        )
    if Path(workspace_src).is_dir():
        runtime_env["working_dir"] = workspace_src
    else:
        logger.warning("Ray working_dir not found", extra={"working_dir": workspace_src})
    if not runtime_env:
        runtime_env = None
    if not ray.is_initialized():
        ray.init(
            address=address or "auto",
            namespace=config.ray_namespace,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
            log_to_driver=False,
            runtime_env=runtime_env,
        )


def _shutdown_ray_cluster(ray_cluster: object | None) -> None:
    """Shutdown Ray and Databricks cluster resources."""
    if ray.is_initialized():
        ray.shutdown()
    if ray_cluster is None:
        return
    shutdown = getattr(ray_cluster, "shutdown", None)
    if callable(shutdown):
        shutdown()


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
    """Initialize Ray on Databricks and run the ingestion job."""
    job_logger = logging.getLogger("pipeline.ray.job")
    job_logger.info("Databricks Ray job started", extra={"pid": os.getpid()})
    start = time.time()
    _apply_python_params(sys.argv[1:])
    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
    s3_prefix = os.environ.get("S3_PREFIX") or (
        sys.argv[1] if len(sys.argv) > 1 and not sys.argv[0].endswith("uvicorn") else "inputs/"
    )

    ray_cfg = RayJobConfig.from_env(namespace)
    minio_cfg = MinIOConfig.from_env(namespace)
    vector_cfg = VectorDBConfig.from_env(namespace)
    embed_cfg = EmbeddingConfig.from_env(namespace)

    ingestion_targets = vector_cfg.ingestion_targets
    job_logger.info(
        "Configuration loaded",
        extra={
            "namespace": namespace,
            "s3_prefix": s3_prefix,
            "num_workers": ray_cfg.num_workers,
            "ingestion_targets": sorted(ingestion_targets),
        },
    )

    ray_cluster = _setup_ray_cluster(ray_cfg)
    _init_ray(ray_cfg, ray_cluster)

    try:
        s3 = S3ClientWrapper(
            endpoint_url=minio_cfg.endpoint_url,
            access_key_id=minio_cfg.access_key_id,
            secret_access_key=minio_cfg.secret_access_key,
        )

        try:
            keys = s3.list_objects(bucket=minio_cfg.bucket, prefix=s3_prefix)
            job_logger.info("S3 objects listed", extra={"count": len(keys)})
        except ClientError as e:
            job_logger.exception("Failed to list S3 objects", extra={"error": str(e)})
            sys.exit(1)

        if not keys:
            job_logger.info("No objects to process")
            return

        processed: set[str] = set()
        checkpoint_path: str | None = None
        checkpoint_manager = CheckpointManager(s3_client=s3 if is_s3_path(ray_cfg.checkpoint_dir) else None)
        if ray_cfg.checkpoint_enabled:
            checkpoint_path = f"{ray_cfg.checkpoint_dir}/{vector_cfg.collection}.json"
            processed = checkpoint_manager.load(checkpoint_path)
            keys = [k for k in keys if k not in processed]

        if not keys:
            job_logger.info("No new objects to process")
            return

        s3_batch_size = ray_cfg.s3_batch_size
        embed_batch_size = ray_cfg.embed_batch_max
        qdrant_batch_size = ray_cfg.batch_upsert_size

        total_keys = len(keys)
        key_batches = [keys[i : i + s3_batch_size] for i in range(0, len(keys), s3_batch_size)]
        total_batches = len(key_batches)
        job_logger.info(
            "Starting batch processing: %d documents in %d batches",
            total_keys,
            total_batches,
            extra={
                "total_keys": total_keys,
                "num_batches": total_batches,
                "num_workers": ray_cfg.num_workers,
                "s3_batch_size": s3_batch_size,
                "embed_batch_size": embed_batch_size,
                "qdrant_batch_size": qdrant_batch_size,
            },
        )

        ollama_rps = ray_cfg.ollama_requests_per_second
        rate_limiter = RateLimiterActor.remote(rate_per_sec=ollama_rps)

        task_fn = process_s3_batch_ray.options(
            num_cpus=ray_cfg.task_num_cpus,
            max_retries=ray_cfg.task_max_retries,
        )

        futures = [
            task_fn.remote(
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
                batch_id=batch_index + 1,
                total_batches=total_batches,
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
            for batch_index, batch in enumerate(key_batches)
        ]

        results: list[tuple[str, bool, str]] = []
        completed_keys = 0
        last_logged_count = 0
        last_log_time = time.time()
        wait_batch = ray_cfg.wait_batch_size
        wait_timeout = ray_cfg.wait_timeout
        log_interval = ray_cfg.progress_log_interval
        time_log_interval = 10.0

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

            now = time.time()
            progress_since_last_log = completed_keys - last_logged_count
            time_since_last_log = now - last_log_time
            should_log = (
                progress_since_last_log >= log_interval
                or completed_keys == total_keys
                or time_since_last_log >= time_log_interval
            )
            if should_log:
                pct = round(100 * completed_keys / total_keys, 1)
                elapsed = round(now - start, 1)
                rate = round(completed_keys / elapsed, 1) if elapsed > 0 else 0
                pending = len(futures)
                job_logger.info(
                    "Progress: %d/%d (%.1f%%) - %.1fs, %.1f docs/s, %d pending",
                    completed_keys,
                    total_keys,
                    pct,
                    elapsed,
                    rate,
                    pending,
                    extra={
                        "completed": completed_keys,
                        "total": total_keys,
                        "percent": pct,
                        "elapsed_seconds": elapsed,
                        "docs_per_second": rate,
                        "pending_tasks": pending,
                    },
                )
                last_logged_count = completed_keys
                last_log_time = now

        success = sum(1 for _, ok, _ in results if ok)
        failed = len(results) - success

        if ray_cfg.checkpoint_enabled and checkpoint_path:
            processed.update(k for k, ok, _ in results if ok)
            checkpoint_manager.save(checkpoint_path, processed)

        elapsed = round(time.time() - start, 2)
        job_logger.info(
            "Job complete: %d succeeded, %d failed in %.2fs",
            success,
            failed,
            elapsed,
            extra={"success": success, "failed": failed, "elapsed_seconds": elapsed},
        )
    finally:
        logger.info("Databricks Ray job completed; shutting down Ray")
        _shutdown_ray_cluster(ray_cluster)


if __name__ == "__main__":
    main()
