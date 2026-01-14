"""Executor-side async partition processor.

This module provides async processing functions for Spark partitions.
Uses shared interfaces from `core.ingestion.interfaces` for
consistency with Ray implementation.

Features:
- Async S3 reads
- Async Ollama embeddings
- Rate-limited Ollama calls
- Dynamic embedding batch sizing
- Batched Qdrant/Weaviate upserts
"""

import asyncio
import logging
from collections.abc import Iterable

import attrs

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

logger = logging.getLogger("pipeline.spark")


# =============================================================================
# Spark-Specific Configuration
# =============================================================================


@attrs.define(frozen=True, slots=True)
class SparkProcessingConfig(ProcessingConfig):
    """Extended configuration for Spark partition processing.

    Inherits from ProcessingConfig and adds Spark-specific settings
    for rate limiting and concurrency control.

    Attributes:
        ollama_max_concurrency: Maximum concurrent embedding requests.
        ollama_rps: Maximum embedding requests per second.
        min_embed_batch: Minimum batch size for dynamic sizing.
        max_embed_batch: Maximum batch size for dynamic sizing.
    """

    ollama_max_concurrency: int = 10
    ollama_rps: int = 5
    min_embed_batch: int = 1
    max_embed_batch: int = 8


# =============================================================================
# Spark Processing Pipeline
# =============================================================================


class SparkProcessingPipeline:
    """Spark implementation of the processing pipeline.

    Uses shared operation classes from interfaces package for
    S3 fetching, embedding generation, and vector DB upserts.

    Example:
        >>> config = SparkProcessingConfig(...)
        >>> pipeline = SparkProcessingPipeline(config)
        >>> results = await pipeline.process_keys_async(["doc1.txt", "doc2.txt"])
    """

    def __init__(self, config: SparkProcessingConfig) -> None:
        """Initialize the pipeline.

        Args:
            config: Spark processing configuration.
        """
        self._config = config
        self._clients_factory = ProcessingClientsFactory()

    @property
    def config(self) -> SparkProcessingConfig:
        """Get the processing configuration."""
        return self._config

    async def process_keys_async(self, keys: list[str]) -> list[ProcessingResult]:
        """Process S3 keys asynchronously.

        Args:
            keys: List of S3 object keys to process.

        Returns:
            List of ProcessingResult objects.
        """
        if not keys:
            return []

        clients = self._clients_factory.create(self._config)

        try:
            # Phase 1: Fetch S3 content (sync - S3 client is sync)
            fetcher = S3ContentFetcher(clients.s3, self._config.s3_bucket)
            contents, fetch_failures = fetcher.fetch_all(keys)

            if not contents:
                return fetch_failures

            # Phase 2: Process content (embed + upsert) - async
            process_results = await self._process_contents_async(contents, clients)

            return fetch_failures + process_results

        finally:
            await clients.close_async()

    async def _process_contents_async(
        self,
        contents: list[ContentResult],
        clients,
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

        # Create rate limiter
        rate_limiter = RateLimiter(self._config.ollama_rps)

        # Create operation instances
        generator = EmbeddingGenerator(clients.embedder, rate_limiter)
        point_factory = VectorPointFactory(self._config.s3_bucket)
        upserter = VectorDBUpserter(clients.qdrant_db, clients.weaviate_db)

        processor = BatchProcessor(
            embedding_generator=generator,
            point_factory=point_factory,
            upserter=upserter,
            collection=self._config.collection,
        )

        # Process in batches with dynamic sizing
        semaphore = asyncio.Semaphore(self._config.ollama_max_concurrency)
        results: list[ProcessingResult] = []
        batch_size = self._config.max_embed_batch

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
                    min_batch_size=self._config.min_embed_batch,
                    max_batch_size=self._config.max_embed_batch,
                )
                results.extend(batch_results)

        # Process remaining buffer
        if buffer:
            batch_results, _ = await processor.process_batch_async(
                buffer,
                semaphore,
                batch_size,
                min_batch_size=self._config.min_embed_batch,
                max_batch_size=self._config.max_embed_batch,
            )
            results.extend(batch_results)

        return results


# =============================================================================
# Spark Partition Functions
# =============================================================================


async def process_partition_async(
    keys: Iterable[str],
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    s3_bucket: str,
    embedding_config: EmbeddingConfig,
    collection: str,
    ollama_max_concurrency: int,
    ollama_rps: int,
    min_embed_batch: int,
    max_embed_batch: int,
) -> Iterable[tuple[str, bool, str]]:
    """Process a partition of S3 keys asynchronously.

    This is the main entry point for Spark partition processing.
    It maintains the original function signature for backward compatibility.

    Args:
        keys: Iterable of S3 object keys to process.
        s3_endpoint: S3 service endpoint URL.
        s3_access_key: S3 access key ID.
        s3_secret_key: S3 secret access key.
        s3_bucket: S3 bucket name containing the objects.
        embedding_config: Configuration for the embedding provider.
        collection: Vector database collection name.
        ollama_max_concurrency: Maximum concurrent embedding requests.
        ollama_rps: Maximum embedding requests per second (rate limit).
        min_embed_batch: Minimum batch size for embeddings.
        max_embed_batch: Maximum batch size for embeddings.

    Returns:
        Iterable of tuples (s3_key, success, error_message).
    """
    keys = list(keys)
    if not keys:
        return iter([])

    # Create config from parameters
    config = SparkProcessingConfig(
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
        embedding_config=embedding_config,
        collection=collection,
        ollama_max_concurrency=ollama_max_concurrency,
        ollama_rps=ollama_rps,
        min_embed_batch=min_embed_batch,
        max_embed_batch=max_embed_batch,
    )

    # Process using pipeline
    pipeline = SparkProcessingPipeline(config)
    results = await pipeline.process_keys_async(keys)

    # Convert to tuples for Spark compatibility
    return iter([r.to_tuple() for r in results])


def process_partition_async_wrapper(*args, **kwargs) -> Iterable[tuple[str, bool, str]]:
    """Spark-compatible wrapper that runs async function synchronously.

    Spark's RDD operations require synchronous functions, but our processing
    logic is async. This wrapper creates a new event loop, runs the async
    function, and returns the results synchronously.

    This function is called by Spark executors via `rdd.mapPartitions()`.
    Each partition gets its own event loop that is created and torn down
    for the partition's processing.

    Args:
        *args: Positional arguments passed to `process_partition_async`.
        **kwargs: Keyword arguments passed to `process_partition_async`.

    Returns:
        Iterable of tuples (s3_key, success, error_message).
    """
    return asyncio.run(process_partition_async(*args, **kwargs))
