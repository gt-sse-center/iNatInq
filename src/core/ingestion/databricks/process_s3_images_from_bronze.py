"""Databricks Ray job: Bronze Delta window -> CLIP embeddings -> vector DB.

This is the incremental CDC consumer for S3 image ingestion. It reads a bounded
ordered window from Bronze, processes keys with Ray, and updates a dedicated
progress table only through the last contiguous successful key.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from logging.config import dictConfig
from typing import TYPE_CHECKING, Any

from clients.qdrant import QdrantClientWrapper
from config import EmbeddingConfig, MinIOConfig, RayJobConfig, VectorDBConfig
from core.ingestion.databricks.batch_runner import run_ray_batch_processing
from core.ingestion.databricks.cdc import (
    CDCWindowConfig,
    assert_unique_window_keys,
    compute_commit_cursor,
    ensure_progress_table,
    load_next_window,
    load_progress_cursor,
    merge_progress_cursor,
)
from core.ingestion.databricks.runtime import apply_python_params as _apply_python_params
from core.ingestion.shared.qdrant_indexing import qdrant_indexing_disabled
from core.ingestion.strategies import DatabricksStrategy
from core.ingestion.tasks import process_image_batch_ray
from foundation.logger import LOGGING_CONFIG

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pyspark.sql import SparkSession

dictConfig(LOGGING_CONFIG)

logger = logging.getLogger("pipeline.ray.databricks")


@dataclass(frozen=True, slots=True)
class BronzeRayCDCConfig:
    """Runtime configuration for Bronze-backed incremental Ray ingestion."""

    bronze_table: str
    progress_table: str
    progress_id: str
    key_col: str
    watermark_col: str
    window_size: int

    def to_window_config(self) -> CDCWindowConfig:
        """Convert runtime env config to CDC window selection config."""
        return CDCWindowConfig(
            bronze_table=self.bronze_table,
            progress_table=self.progress_table,
            progress_id=self.progress_id,
            watermark_col=self.watermark_col,
            key_col=self.key_col,
            window_size=self.window_size,
        )

    @classmethod
    def from_env(cls) -> BronzeRayCDCConfig:
        """Load Bronze CDC read/progress settings from environment variables."""
        bronze_table = (os.getenv("AUTOLOADER_BRONZE_TABLE") or "").strip()
        if not bronze_table:
            raise ValueError("Missing required CDC config: AUTOLOADER_BRONZE_TABLE")

        progress_table = (os.getenv("CDC_PROGRESS_TABLE") or f"{bronze_table}_progress").strip()
        progress_id = (os.getenv("CDC_PROGRESS_ID") or "s3_bronze_image_ingestion").strip()
        key_col = (os.getenv("CDC_KEY_COL") or "s3_key").strip()
        watermark_col = (os.getenv("CDC_WATERMARK_COL") or "discovered_at").strip()
        window_size_raw = (os.getenv("CDC_WINDOW_SIZE") or "5000").strip()

        try:
            window_size = int(window_size_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("CDC_WINDOW_SIZE must be a positive integer") from exc
        if window_size <= 0:
            raise ValueError("CDC_WINDOW_SIZE must be a positive integer")

        return cls(
            bronze_table=bronze_table,
            progress_table=progress_table,
            progress_id=progress_id,
            key_col=key_col,
            watermark_col=watermark_col,
            window_size=window_size,
        )


def _iter_key_batches(keys: list[str], *, batch_size: int) -> Iterator[list[str]]:
    """Yield key batches from an in-memory list."""
    resolved_batch_size = max(1, batch_size)
    for idx in range(0, len(keys), resolved_batch_size):
        yield keys[idx : idx + resolved_batch_size]


def _assert_table_exists(spark: SparkSession, *, table_name: str) -> None:
    """Fail fast with a clear error when a required table is missing."""
    try:
        spark.table(table_name).limit(1).collect()
    except Exception as exc:  # pragma: no cover - depends on Spark runtime
        raise RuntimeError(f"Required Delta table is unavailable: {table_name}") from exc


def main() -> None:
    """Process the next incremental Bronze CDC window using Ray on Databricks."""
    job_logger = logging.getLogger("pipeline.ray.job")
    job_logger.info("Databricks Bronze CDC Ray image job started", extra={"pid": os.getpid()})
    start = time.time()

    _apply_python_params(sys.argv[1:])

    cdc_cfg = BronzeRayCDCConfig.from_env()
    ray_cfg = RayJobConfig.from_env()
    minio_cfg = MinIOConfig.from_env()
    embed_cfg = EmbeddingConfig.from_env()
    vector_cfg = VectorDBConfig.from_env()

    collection = vector_cfg.collection
    ingestion_targets = vector_cfg.ingestion_targets
    image_batch_size = max(1, ray_cfg.image_batch_size)
    image_embed_batch_size = max(1, ray_cfg.image_embed_batch_size)
    max_inflight_batches = max(1, ray_cfg.num_workers)
    s3_key_prefix = ray_cfg.s3_prefix

    job_logger.info(
        "Configuration loaded",
        extra={
            "bronze_table": cdc_cfg.bronze_table,
            "progress_table": cdc_cfg.progress_table,
            "progress_id": cdc_cfg.progress_id,
            "collection": collection,
            "window_size": cdc_cfg.window_size,
            "image_batch_size": image_batch_size,
            "image_embed_batch_size": image_embed_batch_size,
            "max_inflight_batches": max_inflight_batches,
            "s3_key_prefix": s3_key_prefix,
            "ingestion_targets": sorted(ingestion_targets),
        },
    )

    window_cfg = cdc_cfg.to_window_config()

    strategy = DatabricksStrategy.from_env()
    strategy.init()

    try:
        spark_session = _spark_session_class()
        spark = spark_session.getActiveSession() or spark_session.builder.getOrCreate()
        _assert_table_exists(spark, table_name=window_cfg.bronze_table)
        ensure_progress_table(spark, progress_table=window_cfg.progress_table)

        cursor = load_progress_cursor(spark, config=window_cfg, collection=collection)
        if cursor is not None:
            job_logger.info(
                "Loaded CDC cursor",
                extra={
                    "cursor_discovered_at": cursor.last_discovered_at.isoformat(),
                    "cursor_s3_key": cursor.last_s3_key,
                },
            )
        else:
            job_logger.info("No CDC cursor found; starting from beginning of Bronze table")

        window_records = load_next_window(spark, config=window_cfg, cursor=cursor)
        if not window_records:
            job_logger.info(
                "No new Bronze rows to process",
                extra={"bronze_table": cdc_cfg.bronze_table, "window_size": cdc_cfg.window_size},
            )
            return

        assert_unique_window_keys(window_records)
        window_keys = [record.s3_key for record in window_records]
        job_logger.info(
            "Loaded Bronze window",
            extra={
                "window_count": len(window_keys),
                "first_s3_key": window_keys[0],
                "last_s3_key": window_keys[-1],
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
                s3_bucket=minio_cfg.bucket,
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
                ingestion_targets=ingestion_targets,
                s3_key_prefix=s3_key_prefix,
            )

        should_disable_indexing = ray_cfg.disable_indexing_during_ingest and "qdrant" in ingestion_targets
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
                batches=_iter_key_batches(window_keys, batch_size=image_batch_size),
                submit_batch=_submit_batch,
                wait_batch_size=ray_cfg.wait_batch_size,
                wait_timeout=ray_cfg.wait_timeout,
                max_inflight_batches=max_inflight_batches,
                job_logger=job_logger,
                progress_label="Bronze CDC image batch progress",
                total_expected_records=len(window_keys),
            )

        commit_cursor = compute_commit_cursor(
            window_records=window_records,
            successful_keys=stats.successful_keys,
        )
        if commit_cursor is not None:
            merge_progress_cursor(
                spark,
                config=window_cfg,
                collection=collection,
                cursor=commit_cursor,
            )
            job_logger.info(
                "Committed CDC cursor",
                extra={
                    "cursor_discovered_at": commit_cursor.last_discovered_at.isoformat(),
                    "cursor_s3_key": commit_cursor.last_s3_key,
                    "successful_in_window": stats.successful,
                    "failed_in_window": stats.failed,
                },
            )
        else:
            job_logger.warning(
                "CDC cursor not advanced because no leading contiguous successes were found",
                extra={"failed_in_window": stats.failed, "window_count": len(window_keys)},
            )

        elapsed = round(time.time() - start, 2)
        rate = round(stats.completed_records / elapsed, 2) if elapsed > 0 else 0
        job_logger.info(
            "Databricks Bronze CDC Ray image job complete: %d successful, %d failed in %.2fs (%.2f images/s)",
            stats.successful,
            stats.failed,
            elapsed,
            rate,
            extra={
                "successful": stats.successful,
                "failed": stats.failed,
                "total": stats.completed_records,
                "submitted_records": stats.submitted_records,
                "num_batches": stats.num_batches,
                "elapsed_seconds": elapsed,
                "rate_per_sec": rate,
            },
        )
    except Exception as e:
        job_logger.error(
            "Unexpected error in Databricks Bronze CDC Ray image job: %s",
            e,
            extra={"error": str(e)},
            exc_info=True,
        )
        raise
    finally:
        strategy.shutdown()


def _spark_session_class() -> Any:
    """Load SparkSession lazily for environments without pyspark."""
    try:
        from pyspark.sql import SparkSession
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("pyspark is required for Bronze CDC Ray job") from exc
    return SparkSession


if __name__ == "__main__":
    main()
