"""Ray remote functions for processing S3 objects.

This module provides Ray remote functions that process S3 objects,
generate embeddings, and upsert to vector databases. Uses shared
interfaces from `core.ingestion.interfaces` for consistency
with Spark implementation.
"""

import asyncio
import logging
import os
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
from foundation.rate_limiter import RateLimiter

logger = logging.getLogger("pipeline.ray")


# =============================================================================
# Ray-Specific Configuration
# =============================================================================


@attrs.define(frozen=True, slots=True)
class RayProcessingConfig(ProcessingConfig):
    """Extended configuration for Ray processing with rate limiting.

    Inherits from ProcessingConfig and adds Ray-specific settings.

    Attributes:
        rate_limit_rps: Requests per second for rate limiting.
        max_concurrency: Maximum concurrent embedding requests.
    """

    rate_limit_rps: int = 5
    max_concurrency: int = 10


# =============================================================================
# Ray Processing Pipeline
# =============================================================================


class RayProcessingPipeline:
    """Ray implementation of the processing pipeline.

    Uses shared operation classes from interfaces package for
    S3 fetching, embedding generation, and vector DB upserts.

    Example:
        >>> config = RayProcessingConfig(...)
        >>> pipeline = RayProcessingPipeline(config, rate_limiter)
        >>> results = pipeline.process_keys_sync(["doc1.txt", "doc2.txt"])
    """

    def __init__(
        self,
        config: RayProcessingConfig,
        rate_limiter: RateLimiter | None = None,
    ):
        """Initialize the pipeline.

        Args:
            config: Ray processing configuration.
            rate_limiter: Optional rate limiter for embedding API calls.
        """
        self._config = config
        self._rate_limiter = rate_limiter
        self._clients_factory = ProcessingClientsFactory()

    @property
    def config(self) -> RayProcessingConfig:
        """Get the processing configuration."""
        return self._config

    def process_keys_sync(self, keys: list[str]) -> list[ProcessingResult]:
        """Process S3 keys synchronously (for Ray remote functions).

        Args:
            keys: List of S3 object keys to process.

        Returns:
            List of ProcessingResult objects.
        """
        if not keys:
            return []

        clients = self._clients_factory.create(self._config)

        try:
            # Phase 1: Fetch S3 content
            fetcher = S3ContentFetcher(clients.s3, self._config.s3_bucket)
            contents, fetch_failures = fetcher.fetch_all(keys)

            if not contents:
                return fetch_failures

            # Phase 2: Process content (embed + upsert)
            process_results = asyncio.run(self._process_contents_async(contents, clients))

            return fetch_failures + process_results

        finally:
            clients.close_sync()

    async def _process_contents_async(
        self,
        contents: list[ContentResult],
        clients: Any,
    ) -> list[ProcessingResult]:
        """Process contents through embedding and upsert pipeline.

        Args:
            contents: List of S3 content to process.
            clients: Processing clients bundle.

        Returns:
            List of processing results.
        """
        if not contents:
            return []

        # Create operation instances
        generator = EmbeddingGenerator(clients.embedder, self._rate_limiter)
        point_factory = VectorPointFactory(self._config.s3_bucket)
        upserter = VectorDBUpserter(clients.qdrant_db, clients.weaviate_db)

        processor = BatchProcessor(
            embedding_generator=generator,
            point_factory=point_factory,
            upserter=upserter,
            collection=self._config.collection,
        )

        # Process in batches
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

        # Process remaining buffer
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


# =============================================================================
# Ray Remote Functions
# =============================================================================


@ray.remote
def process_s3_object_ray(
    s3_key: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket: str,
    embedding_config: EmbeddingConfig,
    collection: str,
    embed_batch_size: int = 8,
    qdrant_batch_size: int = 200,
) -> tuple[str, bool, str]:
    """Process a single S3 object using Ray remote execution.

    This is a Ray remote function that processes one S3 object:
    1. Fetches content from S3
    2. Generates embedding
    3. Upserts to vector databases

    Args:
        s3_key: S3 object key to process.
        s3_endpoint: S3 endpoint URL.
        s3_access_key: S3 access key.
        s3_secret_key: S3 secret key.
        s3_bucket: S3 bucket name.
        embedding_config: Embedding provider configuration.
        collection: Vector database collection name.
        embed_batch_size: Batch size for embeddings.
        qdrant_batch_size: Batch size for Qdrant upserts.

    Returns:
        Tuple of (s3_key, success, error_message).
    """
    config = RayProcessingConfig(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        embedding_config=embedding_config,
        collection=collection,
        embed_batch_size=embed_batch_size,
        upsert_batch_size=qdrant_batch_size,
    )

    pipeline = RayProcessingPipeline(config)
    results = pipeline.process_keys_sync([s3_key])

    if results:
        return results[0].to_tuple()
    return (s3_key, False, "No result returned")


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
) -> list[tuple[str, bool, str]]:
    """Process a batch of S3 objects using Ray remote execution.

    This is a Ray remote function that processes multiple S3 objects
    for better throughput than individual calls.

    Args:
        s3_keys: List of S3 object keys to process.
        s3_endpoint: S3 endpoint URL.
        s3_access_key: S3 access key.
        s3_secret_key: S3 secret key.
        s3_bucket: S3 bucket name.
        embedding_config: Embedding provider configuration.
        collection: Vector database collection name.
        embed_batch_size: Batch size for embeddings.
        qdrant_batch_size: Batch size for Qdrant upserts.
        rate_limiter: Optional Ray actor for distributed rate limiting.

    Returns:
        List of tuples (s3_key, success, error_message).
    """
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
    )

    # Create rate limiter wrapper if actor provided
    local_rate_limiter = None
    if rate_limiter is not None:
        # Wrap the Ray actor in a local rate limiter interface
        local_rate_limiter = _RayActorRateLimiter(rate_limiter)

    pipeline = RayProcessingPipeline(config, local_rate_limiter)
    results = pipeline.process_keys_sync(s3_keys)

    return [r.to_tuple() for r in results]


class _RayActorRateLimiter:
    """Adapter that wraps a Ray rate limiter actor as a local RateLimiter.

    This allows the RayProcessingPipeline to use a distributed rate limiter
    actor transparently.
    """

    def __init__(self, actor: Any):
        """Initialize with a Ray rate limiter actor.

        Args:
            actor: Ray actor with an `acquire` remote method.
        """
        self._actor = actor

    async def acquire(self) -> None:
        """Acquire permission from the distributed rate limiter."""
        await self._actor.acquire.remote()

    def get_rate(self) -> float:
        """Get the rate limit (not used, but required by interface)."""
        return 0.0

