"""Databricks Ray job: iNaturalist metadata → images → CLIP → vector DBs.

This entrypoint mirrors process_s3_images.py, but sources images from
iNaturalist open-data metadata using INaturalistOpenDataClient.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from logging.config import dictConfig
from typing import Any
from urllib.parse import urlparse

import ray

from clients.clip import CLIPClient
from clients.inaturalist_open_data import INaturalistOpenDataClient
from config import ImageEmbeddingConfig, RayJobConfig
from core.ingestion.interfaces.factories import VectorDBConfigFactory, create_vector_db_provider
from core.ingestion.interfaces.operations import detect_image_format
from core.ingestion.interfaces.types import ImageContentResult, ProcessingResult
from core.ingestion.strategies import DatabricksStrategy
from core.ingestion.tasks.image_processing import ImageProcessingPipeline, RayImageProcessingConfig
from foundation.http import create_retry_session
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


def _parse_required_positive_int_env(name: str) -> int:
    """Parse a required positive integer environment variable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(f"{name} is required and must be a positive integer")
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a positive integer") from None
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _record_to_task_payload(record: Any, image_size: str) -> dict[str, str]:
    """Convert INaturalistPhotoRecord-like object to serializable task payload."""
    photo_url = str(getattr(record, "photo_url", "")).strip()
    photo_id = str(getattr(record, "photo_id", "")).strip()
    extension = str(getattr(record, "extension", "")).strip().lstrip(".").lower()
    parsed_path = urlparse(photo_url).path.lstrip("/")
    fallback_key = f"photos/{photo_id}/{image_size}.{extension}" if photo_id and extension else photo_url
    return {
        "photo_id": photo_id,
        "extension": extension,
        "photo_url": photo_url,
        "s3_key": parsed_path or fallback_key,
    }


@ray.remote(num_cpus=1, max_retries=3)
def process_inat_photo_batch_ray(
    records: list[dict[str, str]],
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
    inat_photo_base_url: str = "https://inaturalist-open-data.s3.amazonaws.com/photos",
    inat_timeout_s: int = 120,
    inat_cb_failure_threshold: int = 5,
    inat_cb_recovery_timeout_s: int = 30,
) -> list[tuple[str, bool, str]]:
    """Process a batch of iNaturalist photo records through image pipeline."""
    task_logger = logging.getLogger("ray.task")
    task_logger.info("Processing iNaturalist image batch of %d records", len(records))

    namespace = os.getenv("K8S_NAMESPACE", "ml-system")
    config = RayImageProcessingConfig(
        s3_endpoint="",
        s3_access_key="",
        s3_secret_key="",
        s3_bucket="",
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
    pipeline = ImageProcessingPipeline(config, rate_limiter=rate_limiter)

    session = create_retry_session()
    clip_client = CLIPClient.from_config(image_embedding_config, session=session)
    db_factory = VectorDBConfigFactory(namespace)
    qdrant_cfg, weaviate_cfg = db_factory.create_both()
    qdrant_db = create_vector_db_provider(qdrant_cfg)
    weaviate_db = create_vector_db_provider(weaviate_cfg)
    inat_client = INaturalistOpenDataClient(
        photo_base_url=inat_photo_base_url,
        timeout_s=inat_timeout_s,
        circuit_breaker_failure_threshold=inat_cb_failure_threshold,
        circuit_breaker_recovery_timeout_s=inat_cb_recovery_timeout_s,
    )

    try:
        images: list[ImageContentResult] = []
        download_failures: list[ProcessingResult] = []

        for record in records:
            photo_url = record.get("photo_url", "").strip()
            key = record.get("s3_key", "").strip() or photo_url
            extension = record.get("extension", "").strip().lstrip(".").lower()

            try:
                image_bytes = inat_client.download_image(photo_url)
                detected_format = detect_image_format(image_bytes)
                resolved_format = detected_format or ("jpeg" if extension == "jpg" else extension or "jpeg")
                images.append(
                    ImageContentResult(
                        s3_key=key,
                        image_bytes=image_bytes,
                        format=resolved_format,
                        size_bytes=len(image_bytes),
                        source_uri=photo_url,
                    )
                )
            except Exception as e:
                download_failures.append(
                    ProcessingResult.failure_result(key, f"iNaturalist image download failed: {e}")
                )

        if not images:
            return [result.to_tuple() for result in download_failures]

        process_results = asyncio.run(pipeline._process_images_async(images, clip_client, qdrant_db, weaviate_db))
        return [result.to_tuple() for result in (download_failures + process_results)]
    finally:
        qdrant_db.close()
        weaviate_db.close()
        inat_client.close()
        session.close()


def main() -> None:
    """Process iNaturalist images using Ray on Databricks."""
    job_logger = logging.getLogger("pipeline.ray.job")
    job_logger.info("Databricks iNaturalist image job started", extra={"pid": os.getpid()})
    start = time.time()

    _apply_python_params(sys.argv[1:])

    vector_db_provider = (os.environ.get("VECTOR_DB_PROVIDER") or "").strip().lower()
    if not vector_db_provider:
        os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
        job_logger.warning(
            "VECTOR_DB_PROVIDER not set; defaulting to qdrant",
            extra={"vector_db_provider": "qdrant"},
        )

    namespace = os.environ.get("K8S_NAMESPACE", "ml-system")
    collection = os.environ.get("VECTOR_DB_COLLECTION", "documents")
    image_size = (os.environ.get("INAT_IMAGE_SIZE") or "medium").strip().lower()
    max_rows = _parse_required_positive_int_env("INAT_MAX_ROWS")
    metadata_url = (os.environ.get("INAT_METADATA_URL") or "").strip()

    inat_photo_base_url = (
        os.environ.get("INAT_PHOTO_BASE_URL") or "https://inaturalist-open-data.s3.amazonaws.com/photos"
    ).strip()
    inat_timeout_s = int((os.environ.get("INAT_TIMEOUT_S") or "120").strip())
    inat_cb_failure_threshold = int((os.environ.get("INAT_CB_FAILURE_THRESHOLD") or "5").strip())
    inat_cb_recovery_timeout_s = int((os.environ.get("INAT_CB_RECOVERY_TIMEOUT_S") or "30").strip())

    ray_cfg = RayJobConfig.from_env(namespace)
    embed_cfg = ImageEmbeddingConfig.from_env(namespace)

    job_logger.info(
        "Configuration loaded",
        extra={
            "namespace": namespace,
            "collection": collection,
            "num_workers": ray_cfg.num_workers,
            "image_batch_size": ray_cfg.image_batch_size,
            "metadata_url": metadata_url or "default:s3://inaturalist-open-data/photos.csv.gz",
            "image_size": image_size,
            "max_rows": max_rows,
        },
    )

    inat_client = INaturalistOpenDataClient(
        photo_base_url=inat_photo_base_url,
        timeout_s=inat_timeout_s,
        circuit_breaker_failure_threshold=inat_cb_failure_threshold,
        circuit_breaker_recovery_timeout_s=inat_cb_recovery_timeout_s,
    )
    if not metadata_url:
        metadata_url = inat_client.build_metadata_s3_uri(dataset="photos", compressed=True)
        job_logger.info(
            "INAT_METADATA_URL not set; defaulting to official photos metadata",
            extra={"metadata_url": metadata_url},
        )

    strategy = DatabricksStrategy.from_env(namespace)
    strategy.init()

    try:
        records = inat_client.read_photo_records(
            metadata_url=metadata_url,
            size=image_size,
            max_rows=max_rows,
        )
        task_records = [_record_to_task_payload(record, image_size) for record in records]
        job_logger.info(
            "Loaded iNaturalist records from metadata",
            extra={"records": len(task_records), "metadata_url": metadata_url},
        )

        if not task_records:
            job_logger.info("No iNaturalist image records to process")
            return

        image_batch_size = ray_cfg.image_batch_size
        image_embed_batch_size = ray_cfg.image_embed_batch_size
        record_batches = [task_records[i : i + image_batch_size] for i in range(0, len(task_records), image_batch_size)]
        num_batches = len(record_batches)

        job_logger.info(
            "Starting iNaturalist image batch processing: %d records in %d batches",
            len(task_records),
            num_batches,
            extra={
                "total_records": len(task_records),
                "num_batches": num_batches,
                "image_batch_size": image_batch_size,
                "image_embed_batch_size": image_embed_batch_size,
            },
        )

        task_fn = process_inat_photo_batch_ray.options(
            num_cpus=ray_cfg.task_num_cpus,
            max_retries=ray_cfg.task_max_retries,
        )

        futures = [
            task_fn.remote(
                records=batch,
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
                inat_photo_base_url=inat_photo_base_url,
                inat_timeout_s=inat_timeout_s,
                inat_cb_failure_threshold=inat_cb_failure_threshold,
                inat_cb_recovery_timeout_s=inat_cb_recovery_timeout_s,
            )
            for batch in record_batches
        ]

        results: list[tuple[str, bool, str]] = []
        completed_records = 0
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
                completed_records += len(batch_result)
            if batch_results:
                success_so_far = sum(1 for _, ok, _ in results if ok)
                failed_so_far = len(results) - success_so_far
                job_logger.info(
                    "iNaturalist image batch progress: %d/%d completed (%d ok, %d failed)",
                    completed_records,
                    len(task_records),
                    success_so_far,
                    failed_so_far,
                    extra={
                        "completed_records": completed_records,
                        "total_records": len(task_records),
                        "successful": success_so_far,
                        "failed": failed_so_far,
                        "remaining_records": len(task_records) - completed_records,
                    },
                )

        success = sum(1 for _, ok, _ in results if ok)
        failed = len(results) - success
        elapsed = round(time.time() - start, 2)
        rate = round(len(results) / elapsed, 2) if elapsed > 0 else 0
        job_logger.info(
            "Databricks iNaturalist image job complete: %d successful, %d failed in %.2fs (%.2f images/s)",
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
            "Unexpected error in Databricks iNaturalist image job: %s",
            e,
            extra={"error": str(e)},
            exc_info=True,
        )
        raise
    finally:
        inat_client.close()
        strategy.shutdown()


if __name__ == "__main__":
    main()
