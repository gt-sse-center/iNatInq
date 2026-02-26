"""Unified ingestion pipeline for Ray-based document and image processing.

This module provides a unified orchestrator that works across different
Ray deployment environments (local, Databricks, Kubernetes) using the
strategy pattern for cluster management.

## Usage

```python
from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.strategies import LocalRayStrategy

# Create pipeline with local Ray strategy
strategy = LocalRayStrategy.from_env()
pipeline = IngestionPipeline.from_env(cluster_strategy=strategy)

# Run image ingestion
result = pipeline.run(s3_prefix="images/")
```

## Design

The pipeline separates concerns:
- **ClusterStrategy**: Handles Ray cluster lifecycle (init/shutdown)
- **IngestionPipeline**: Handles orchestration (list, batch, distribute, collect)
- **Task functions**: Handle per-batch processing (existing @ray.remote functions)

This allows adding new environments by implementing ClusterStrategy without
changing orchestration logic.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import attrs
import ray
from botocore.exceptions import ClientError

from clients.s3 import S3ClientWrapper
from config import EmbeddingConfig, MinIOConfig, RayJobConfig, VectorDBConfig
from foundation.checkpoint import CheckpointManager, is_s3_path
from core.ingestion.shared import RateLimiterActor
from core.ingestion.shared.qdrant_indexing import qdrant_indexing_disabled
from core.ingestion.tasks import process_image_batch_ray
from contextlib import nullcontext

if TYPE_CHECKING:
    from core.ingestion.strategies.base import ClusterStrategy

logger = logging.getLogger("pipeline.ingestion")


@attrs.define
class JobResult:
    """Result of an ingestion job.

    Attributes:
        successful: Number of successfully processed items.
        failed: Number of failed items.
        total: Total number of items processed.
        elapsed_seconds: Total job duration in seconds.
        rate_per_second: Processing rate (items/second).
    """

    successful: int
    failed: int
    total: int
    elapsed_seconds: float
    rate_per_second: float


@attrs.define
class IngestionPipeline:
    """Unified ingestion pipeline for image processing.

    This class orchestrates the ingestion workflow:
    1. Initialize Ray cluster via strategy
    2. List S3 objects
    3. Load/filter via checkpoint
    4. Batch and distribute to Ray workers
    5. Collect results with progress tracking
    6. Save checkpoint
    7. Shutdown cluster

    Attributes:
        cluster_strategy: Strategy for Ray cluster management.
        ray_config: Ray job configuration.
        minio_config: S3/MinIO configuration.
        vector_config: Vector database configuration.
        embed_config: Embedding service configuration.
    """

    cluster_strategy: ClusterStrategy
    ray_config: RayJobConfig
    minio_config: MinIOConfig
    vector_config: VectorDBConfig
    embed_config: EmbeddingConfig

    @classmethod
    def from_env(
        cls,
        cluster_strategy: ClusterStrategy,
        namespace: str = "ml-system",
    ) -> IngestionPipeline:
        """Create pipeline from environment variables.

        Args:
            cluster_strategy: Strategy for Ray cluster management.
            namespace: Kubernetes namespace for config resolution.

        Returns:
            Configured IngestionPipeline instance.
        """
        return cls(
            cluster_strategy=cluster_strategy,
            ray_config=RayJobConfig.from_env(namespace),
            minio_config=MinIOConfig.from_env(namespace),
            vector_config=VectorDBConfig.from_env(namespace),
            embed_config=EmbeddingConfig.from_env(namespace),
        )

    def run(
        self,
        s3_prefix: str,
    ) -> JobResult:
        """Execute the image ingestion pipeline.

        Args:
            s3_prefix: S3 prefix to process (e.g., "inputs/", "images/").

        Returns:
            JobResult with success/failure counts and timing.

        Raises:
            RuntimeError: If cluster initialization fails.
            ClientError: If S3 operations fail.
        """
        job_logger = logging.getLogger("pipeline.ingestion.job")
        job_logger.info(
            "Ingestion job started",
            extra={"s3_prefix": s3_prefix},
        )
        start = time.time()

        self.cluster_strategy.init()
        try:
            return self._execute(s3_prefix, job_logger, start)
        finally:
            self.cluster_strategy.shutdown()

    def _execute(
        self,
        s3_prefix: str,
        job_logger: logging.Logger,
        start: float,
    ) -> JobResult:
        """Execute pipeline after cluster initialization.

        This is the core orchestration logic extracted from both
        ray/process_s3_to_vector_dbs.py and databricks/process_s3_to_vector_dbs.py.
        """
        # Initialize S3 client
        s3 = S3ClientWrapper(
            endpoint_url=self.minio_config.endpoint_url,
            access_key_id=self.minio_config.access_key_id,
            secret_access_key=self.minio_config.secret_access_key,
        )

        # List S3 objects
        try:
            keys = s3.list_objects(bucket=self.minio_config.bucket, prefix=s3_prefix)
            job_logger.info("S3 objects listed", extra={"count": len(keys)})
        except ClientError as e:
            job_logger.exception("Failed to list S3 objects", extra={"error": str(e)})
            raise

        if not keys:
            job_logger.info("No objects to process")
            return JobResult(
                successful=0,
                failed=0,
                total=0,
                elapsed_seconds=time.time() - start,
                rate_per_second=0.0,
            )

        # Load checkpoint if enabled
        processed, checkpoint_path = self._load_checkpoint(s3, keys)
        keys = [k for k in keys if k not in processed]

        if not keys:
            job_logger.info("No new objects to process (all checkpointed)")
            return JobResult(
                successful=0,
                failed=0,
                total=0,
                elapsed_seconds=time.time() - start,
                rate_per_second=0.0,
            )

        # Log batch configuration
        total_keys = len(keys)
        key_batches = self._create_batches(keys)
        total_batches = len(key_batches)

        job_logger.info(
            "Starting batch processing",
            extra={
                "total_keys": total_keys,
                "num_batches": total_batches,
                "num_workers": self.ray_config.num_workers,
            },
        )

        # Create rate limiter and submit tasks
        rate_limiter = RateLimiterActor.remote(rate_per_sec=self.ray_config.ollama_requests_per_second)

        should_disable_indexing = (
            self.ray_config.disable_indexing_during_ingest
            and "qdrant" in self.vector_config.ingestion_targets
        )

        with (
            qdrant_indexing_disabled(
                client=self.vector_config.qdrant_client,
                collection=self.vector_config.collection,
            )
            if should_disable_indexing
            else nullcontext()
        ):
            futures = self._submit_tasks(
                key_batches=key_batches,
                total_batches=total_batches,
                rate_limiter=rate_limiter,
            )

            # Collect results with progress tracking
            results = self._collect_results(
                futures=futures,
                total_keys=total_keys,
                job_logger=job_logger,
                start=start,
            )

        # Calculate statistics
        success = sum(1 for _, ok, _ in results if ok)
        failed = len(results) - success

        # Save checkpoint
        if self.ray_config.checkpoint_enabled and checkpoint_path:
            checkpoint_manager = CheckpointManager(
                s3_client=s3 if is_s3_path(self.ray_config.checkpoint_dir) else None
            )
            processed.update(k for k, ok, _ in results if ok)
            checkpoint_manager.save(checkpoint_path, processed)

        elapsed = time.time() - start
        rate = len(results) / elapsed if elapsed > 0 else 0.0

        job_logger.info(
            "Ingestion job complete",
            extra={
                "successful": success,
                "failed": failed,
                "total": len(results),
                "elapsed_seconds": round(elapsed, 2),
                "rate_per_second": round(rate, 2),
            },
        )

        return JobResult(
            successful=success,
            failed=failed,
            total=len(results),
            elapsed_seconds=elapsed,
            rate_per_second=rate,
        )

    def _load_checkpoint(
        self,
        s3: S3ClientWrapper,
        _keys: list[str],  # will this be used in the future?
    ) -> tuple[set[str], str | None]:
        """Load checkpoint and return processed keys and checkpoint path."""
        processed: set[str] = set()
        checkpoint_path: str | None = None

        if self.ray_config.checkpoint_enabled:
            checkpoint_manager = CheckpointManager(
                s3_client=s3 if is_s3_path(self.ray_config.checkpoint_dir) else None
            )
            checkpoint_path = f"{self.ray_config.checkpoint_dir}/{self.vector_config.collection}.json"
            processed = checkpoint_manager.load(checkpoint_path)

        return processed, checkpoint_path

    def _create_batches(self, keys: list[str]) -> list[list[str]]:
        """Split keys into batches based on configuration."""
        batch_size = self.ray_config.s3_batch_size
        return [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]

    def _submit_tasks(
        self,
        key_batches: list[list[str]],
        total_batches: int,
        rate_limiter: ray.ObjectRef,
    ) -> list[ray.ObjectRef]:
        """Submit Ray tasks for all batches."""

        # Configure task with dynamic options
        task_fn = process_image_batch_ray.options(
            num_cpus=self.ray_config.task_num_cpus,
            max_retries=self.ray_config.task_max_retries,
        )

        futures = []
        for batch_index, batch in enumerate(key_batches):
            future = task_fn.remote(
                s3_keys=batch,
                s3_endpoint=self.minio_config.endpoint_url,
                s3_access_key=self.minio_config.access_key_id,
                s3_secret_key=self.minio_config.secret_access_key,
                s3_bucket=self.minio_config.bucket,
                embedding_config=self.embed_config,
                collection=self.vector_config.collection,
                embed_batch_size=self.ray_config.embed_batch_max,
                qdrant_batch_size=self.ray_config.batch_upsert_size,
                rate_limiter=rate_limiter,
                batch_id=batch_index + 1,
                total_batches=total_batches,
                pipeline_concurrency=self.ray_config.pipeline_concurrency,
                circuit_breaker_threshold=self.ray_config.circuit_breaker_threshold,
                circuit_breaker_timeout=self.ray_config.circuit_breaker_timeout,
                embedding_timeout=self.ray_config.embedding_timeout,
                upsert_timeout=self.ray_config.upsert_timeout,
                retry_max_attempts=self.ray_config.retry_max_attempts,
                retry_min_wait=self.ray_config.retry_min_wait,
                retry_max_wait=self.ray_config.retry_max_wait,
            )
            futures.append(future)

        return futures

    def _collect_results(
        self,
        futures: list[ray.ObjectRef],
        total_keys: int,
        job_logger: logging.Logger,
        start: float,
    ) -> list[tuple[str, bool, str]]:
        """Collect results from Ray tasks with progress tracking."""
        results: list[tuple[str, bool, str]] = []
        completed_keys = 0
        last_logged_count = 0
        last_log_time = time.time()

        wait_batch = self.ray_config.wait_batch_size
        wait_timeout = self.ray_config.wait_timeout
        log_interval = self.ray_config.progress_log_interval
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

            # Log progress based on count or time
            now = time.time()
            progress_since_last_log = completed_keys - last_logged_count
            time_since_last_log = now - last_log_time
            should_log = (
                progress_since_last_log >= log_interval
                or completed_keys == total_keys
                or time_since_last_log >= time_log_interval
            )

            if should_log:
                pct = round(100 * completed_keys / total_keys, 1) if total_keys > 0 else 0
                elapsed = round(now - start, 1)
                rate = round(completed_keys / elapsed, 1) if elapsed > 0 else 0
                pending = len(futures)

                job_logger.info(
                    "Progress: %d/%d (%.1f%%)",
                    completed_keys,
                    total_keys,
                    pct,
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

        return results
