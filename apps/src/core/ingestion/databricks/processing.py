"""Ray remote functions for processing S3 objects (Databricks runtime).

This module mirrors the Ray processing implementation but adds Databricks-
friendly progress logging at the batch level for better visibility in
Databricks run logs.
"""

import asyncio
import logging
import os
from logging.config import dictConfig
from typing import Any

import attrs
import ray  # type: ignore[import-untyped]

from config import EmbeddingConfig
from core.ingestion.interfaces import (
    BatchProcessor,
    ContentResult,
    EmbeddingGenerator,
    ProcessingClientsFactory,
    ProcessingConfig,
    ProcessingResult,
    S3ContentFetcher,
    VectorDBUpserter,
    VectorPointFactory,
)
from foundation.logger import LOGGING_CONFIG
from foundation.rate_limiter import RateLimiter

dictConfig(LOGGING_CONFIG)


def get_ray_logger(name: str = "ray.task") -> logging.Logger:
    """Get a logger configured for Ray workers."""
    return logging.getLogger(name)


logger = get_ray_logger("ray.pipeline")


@attrs.define(frozen=True, slots=True)
class RayProcessingConfig(ProcessingConfig):
    """Extended configuration for Ray processing with rate limiting."""

    rate_limit_rps: int = 5
    max_concurrency: int = 10
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30
    embedding_timeout: int = 120
    upsert_timeout: int = 60
    retry_max_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0


class RayProcessingPipeline:
    """Ray implementation of the processing pipeline for Databricks."""

    def __init__(
        self,
        config: RayProcessingConfig,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._config = config
        self._rate_limiter = rate_limiter
        self._clients_factory = ProcessingClientsFactory()

    @property
    def config(self) -> RayProcessingConfig:
        """Get the processing configuration."""
        return self._config

    def process_keys_sync(self, keys: list[str]) -> list[ProcessingResult]:
        """Process S3 keys synchronously (for Ray remote functions)."""
        if not keys:
            return []

        clients = self._clients_factory.create(self._config)

        try:
            fetcher = S3ContentFetcher(clients.s3, self._config.s3_bucket)
            contents, fetch_failures = fetcher.fetch_all(keys)

            if not contents:
                return fetch_failures

            process_results = asyncio.run(self._process_contents_async(contents, clients))

            return fetch_failures + process_results

        finally:
            clients.close_sync()

    async def _process_contents_async(
        self,
        contents: list[ContentResult],
        clients: Any,
    ) -> list[ProcessingResult]:
        """Process contents through embedding and upsert pipeline."""
        if not contents:
            return []

        generator = EmbeddingGenerator(clients.embedder, self._rate_limiter)
        point_factory = VectorPointFactory(self._config.s3_bucket)
        upserter = VectorDBUpserter(clients.qdrant_db, clients.weaviate_db)

        processor = BatchProcessor(
            embedding_generator=generator,
            point_factory=point_factory,
            upserter=upserter,
            collection=self._config.collection,
        )

        semaphore = asyncio.Semaphore(self._config.max_concurrency)
        results: list[ProcessingResult] = []
        batch_size = self._config.embed_batch_size

        buffer: list[ContentResult] = []
        for content in contents:
            buffer.append(content)

            if len(buffer) >= batch_size:
                batch = buffer[:batch_size]
                buffer = buffer[batch_size:]

                batch_results, batch_size = await processor.process_batch_async(
                    batch,
                    semaphore,
                    batch_size,
                    min_batch_size=1,
                    max_batch_size=self._config.embed_batch_size,
                )
                results.extend(batch_results)

        if buffer:
            batch_results, _ = await processor.process_batch_async(
                buffer,
                semaphore,
                batch_size,
                min_batch_size=1,
                max_batch_size=self._config.embed_batch_size,
            )
            results.extend(batch_results)

        return results


@ray.remote
def process_s3_batch_ray(
    s3_keys: list[str],
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket: str,
    embedding_config: EmbeddingConfig,
    collection: str,
    embed_batch_size: int = 8,
    qdrant_batch_size: int = 200,
    rate_limiter: Any | None = None,
    batch_id: int | None = None,
    total_batches: int | None = None,
    pipeline_concurrency: int = 10,
    circuit_breaker_threshold: int = 5,
    circuit_breaker_timeout: int = 30,
    embedding_timeout: int = 120,
    upsert_timeout: int = 60,
    retry_max_attempts: int = 3,
    retry_min_wait: float = 1.0,
    retry_max_wait: float = 10.0,
) -> list[tuple[str, bool, str]]:
    """Process a batch of S3 objects using Ray remote execution."""
    task_logger = get_ray_logger("ray.task")
    if batch_id is not None and total_batches is not None:
        task_logger.info(
            "Processing batch %d/%d (%d keys)",
            batch_id,
            total_batches,
            len(s3_keys),
            extra={"batch_id": batch_id, "total_batches": total_batches, "keys": len(s3_keys)},
        )
    else:
        task_logger.info("Processing batch of %d keys", len(s3_keys))

    namespace = os.getenv("K8S_NAMESPACE", "ml-system")

    config = RayProcessingConfig(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        embedding_config=embedding_config,
        collection=collection,
        embed_batch_size=embed_batch_size,
        upsert_batch_size=qdrant_batch_size,
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

    pipeline = RayProcessingPipeline(config, local_rate_limiter)
    results = pipeline.process_keys_sync(s3_keys)

    successes = sum(1 for r in results if r.success)
    failures = len(results) - successes
    if batch_id is not None and total_batches is not None:
        task_logger.info(
            "Batch complete %d/%d: %d succeeded, %d failed",
            batch_id,
            total_batches,
            successes,
            failures,
            extra={
                "batch_id": batch_id,
                "total_batches": total_batches,
                "succeeded": successes,
                "failed": failures,
            },
        )
    else:
        task_logger.info("Batch complete: %d succeeded, %d failed", successes, failures)

    for r in results:
        error_message = r.error_message
        if not r.success and "circuit breaker" in error_message.lower():
            task_logger.warning("CIRCUIT_BREAKER_OPEN: %s - %s", r.s3_key, error_message)
        elif not r.success and "upstream" in error_message.lower():
            task_logger.warning("UPSTREAM_ERROR: %s - %s", r.s3_key, error_message)

    return [r.to_tuple() for r in results]


class _RayActorRateLimiter:
    """Adapter that wraps a Ray rate limiter actor as a local RateLimiter."""

    def __init__(self, actor: Any) -> None:
        self._actor = actor

    async def acquire(self) -> None:
        await self._actor.acquire.remote()

    def get_rate(self) -> float:
        return 0.0
