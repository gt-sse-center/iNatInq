"""Operation classes for ingestion pipeline processing.

This module provides classes that encapsulate the core operations of the
ingestion pipeline: S3 fetching, embedding generation, and vector DB upserts.
These classes are shared between Ray and Spark implementations.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from clients.s3 import S3ClientWrapper
from core.exceptions import UpstreamError
from foundation.rate_limiter import RateLimiter

from .types import BatchEmbeddingResult, ContentResult, ProcessingResult

if TYPE_CHECKING:
    from .factories import VectorPointFactory

logger = logging.getLogger("pipeline.ingestion")


class S3ContentFetcher:
    """Fetches content from S3 objects.

    Provides both single-object and batch fetching with error handling
    and structured logging.

    Example:
        >>> fetcher = S3ContentFetcher(s3_client, bucket="pipeline")
        >>> content = fetcher.fetch_one("inputs/doc.txt")
        >>> if content:
        ...     print(content.content)
    """

    def __init__(self, s3_client: S3ClientWrapper, bucket: str) -> None:
        """Initialize the fetcher.

        Args:
            s3_client: S3 client wrapper.
            bucket: S3 bucket name.
        """
        self.s3 = s3_client
        self.bucket = bucket

    def fetch_one(self, key: str) -> ContentResult | None:
        """Fetch content from a single S3 object.

        Args:
            key: S3 object key.

        Returns:
            ContentResult if successful, None if failed.
        """
        try:
            content = self.s3.get_object(bucket=self.bucket, key=key).decode("utf-8")
            return ContentResult(s3_key=key, content=content)
        except (UpstreamError, ClientError, OSError, ValueError) as e:
            logger.error(
                "Failed to fetch S3 object",
                extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            return None
        except (RuntimeError, MemoryError, AttributeError, TypeError) as e:
            logger.error(
                "Unexpected error fetching S3 object",
                extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            return None

    def fetch_all(
        self,
        keys: list[str],
    ) -> tuple[list[ContentResult], list[ProcessingResult]]:
        """Fetch content from all S3 objects.

        Args:
            keys: List of S3 object keys.

        Returns:
            Tuple of (successful content results, failed processing results).
        """
        contents: list[ContentResult] = []
        failures: list[ProcessingResult] = []

        for key in keys:
            result = self.fetch_one(key)
            if result is not None:
                contents.append(result)
            else:
                failures.append(ProcessingResult.failure_result(key, "S3 fetch failed in fetch_all"))

        return contents, failures


class EmbeddingGenerator:
    """Generates embeddings for text content.

    Provides rate-limited, concurrent embedding generation with
    error handling and batch support.

    Example:
        >>> generator = EmbeddingGenerator(embedder, rate_limiter)
        >>> vectors = await generator.generate_batch_async(
        ...     [ContentResult("key", "text")],
        ...     semaphore,
        ... )
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            embedder: Embedding provider.
            rate_limiter: Optional rate limiter for API calls.
        """
        self.embedder = embedder
        self.rate_limiter = rate_limiter

    @property
    def vector_size(self) -> int:
        """Get the vector dimension from the embedder."""
        return self.embedder.vector_size

    async def generate_one_async(self, text: str) -> list[float] | None:
        """Generate embedding for a single text.

        Args:
            text: Input text.

        Returns:
            Embedding vector, or None if failed.
        """
        try:
            return await self.embedder.embed_async(text)
        except (UpstreamError, ClientError, OSError, ValueError) as e:
            logger.warning(
                "Embedding failed",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return None

    async def generate_batch_async(
        self,
        batch: list[ContentResult],
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[float]] | None:
        """Generate embeddings for a batch of content.

        Applies rate limiting if configured, and uses semaphore for
        concurrency control.

        Args:
            batch: List of content results to embed.
            semaphore: Optional concurrency semaphore.

        Returns:
            List of embedding vectors, or None if failed.
        """
        if not batch:
            return []

        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        async def _generate() -> list[list[float]] | None:
            try:
                texts = [c.content for c in batch]
                vectors = await asyncio.gather(*[self.embedder.embed_async(text) for text in texts])
                return list(vectors)
            except (UpstreamError, ClientError, OSError, ValueError) as e:
                logger.warning(
                    "Embedding batch failed",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "batch_size": len(batch),
                    },
                    exc_info=True,
                )
                return None
            except (RuntimeError, MemoryError, AttributeError, TypeError) as e:
                logger.error(
                    "Unexpected error in embedding batch",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "batch_size": len(batch),
                    },
                    exc_info=True,
                )
                return None

        if semaphore:
            async with semaphore:
                return await _generate()
        else:
            return await _generate()


class VectorDBUpserter:
    """Upserts vectors to Qdrant and Weaviate databases.

    Handles parallel upserts to both databases with error handling
    and partial failure support.

    Example:
        >>> upserter = VectorDBUpserter(qdrant_db, weaviate_db)
        >>> success = await upserter.upsert_batch_async(batch, "documents", 768)
    """

    def __init__(
        self,
        qdrant_db: VectorDBProvider,
        weaviate_db: VectorDBProvider,
    ) -> None:
        """Initialize the upserter.

        Args:
            qdrant_db: Qdrant vector database provider.
            weaviate_db: Weaviate vector database provider.
        """
        self.qdrant_db = qdrant_db
        self.weaviate_db = weaviate_db

    async def upsert_batch_async(
        self,
        embedding_result: BatchEmbeddingResult,
        collection: str,
        vector_size: int,
    ) -> bool:
        """Upsert vectors to both Qdrant and Weaviate in parallel.

        Args:
            embedding_result: Batch of points to upsert.
            collection: Collection name.
            vector_size: Vector dimension.

        Returns:
            True if at least one database succeeded, False if both failed.
        """
        if embedding_result.is_empty():
            return True

        # Convert VectorPoint to Qdrant PointStruct
        qdrant_point_structs = [point.to_qdrant() for point in embedding_result.qdrant_points]

        # Upsert to both databases in parallel
        upsert_results = await asyncio.gather(
            self.qdrant_db.batch_upsert_async(
                collection=collection,
                points=qdrant_point_structs,  # type: ignore[arg-type]
                vector_size=vector_size,
            ),
            self.weaviate_db.batch_upsert_async(
                collection=collection,
                points=embedding_result.weaviate_objects,  # type: ignore[arg-type]
                vector_size=vector_size,
            ),
            return_exceptions=True,
        )

        # Check for exceptions from either database
        any_success = False
        for i, result in enumerate(upsert_results):
            if isinstance(result, Exception):
                db_name = "Qdrant" if i == 0 else "Weaviate"
                logger.error(
                    "%s batch upsert failed",
                    db_name,
                    extra={
                        "error": str(result),
                        "error_type": type(result).__name__,
                        "batch_size": len(embedding_result),
                    },
                    exc_info=result,
                )
            else:
                any_success = True

        return any_success


class BatchProcessor:
    """Orchestrates batch processing of content through embedding and upsert.

    Combines EmbeddingGenerator, VectorPointFactory, and VectorDBUpserter
    into a cohesive processing pipeline.

    Example:
        >>> processor = BatchProcessor(
        ...     embedding_generator=generator,
        ...     point_factory=factory,
        ...     upserter=upserter,
        ...     collection="documents",
        ... )
        >>> results, new_size = await processor.process_batch_async(
        ...     batch, semaphore, current_batch_size, min_size, max_size
        ... )
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        point_factory: "VectorPointFactory",  # Forward ref to avoid circular import
        upserter: VectorDBUpserter,
        collection: str,
    ) -> None:
        """Initialize the processor.

        Args:
            embedding_generator: Generator for embeddings.
            point_factory: Factory for creating vector points.
            upserter: Upserter for vector databases.
            collection: Collection name for upserting.
        """
        self.embedding_generator = embedding_generator
        self.point_factory = point_factory
        self.upserter = upserter
        self.collection = collection

    async def process_batch_async(
        self,
        batch: list[ContentResult],
        semaphore: asyncio.Semaphore | None,
        current_batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
    ) -> tuple[list[ProcessingResult], int]:
        """Process a batch of content: embed and upsert.

        Implements dynamic batch sizing: grows on success, shrinks on failure.

        Args:
            batch: List of content to process.
            semaphore: Concurrency semaphore.
            current_batch_size: Current dynamic batch size.
            min_batch_size: Minimum batch size.
            max_batch_size: Maximum batch size.

        Returns:
            Tuple of (processing results, new batch size).
        """
        if not batch:
            return [], current_batch_size

        # Generate embeddings
        vectors = await self.embedding_generator.generate_batch_async(batch, semaphore)

        if vectors is None:
            # Embedding failed - shrink batch size
            new_size = max(current_batch_size // 2, min_batch_size)
            return [ProcessingResult.failure_result(c.s3_key, "Embedding failed") for c in batch], new_size

        # Create vector points
        embedding_result = self.point_factory.create_batch(batch, vectors)
        vector_size = len(vectors[0]) if vectors else self.embedding_generator.vector_size

        # Upsert to databases
        success = await self.upserter.upsert_batch_async(embedding_result, self.collection, vector_size)

        if success:
            # Success - grow batch size cautiously
            new_size = min(current_batch_size + 1, max_batch_size)
            return [ProcessingResult.success_result(c.s3_key) for c in batch], new_size
        # Both DBs failed - shrink batch size
        new_size = max(current_batch_size // 2, min_batch_size)
        return [ProcessingResult.failure_result(c.s3_key, "Upsert failed") for c in batch], new_size
