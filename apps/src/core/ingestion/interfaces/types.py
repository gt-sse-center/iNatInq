"""Shared type classes for ingestion pipelines.

This module provides attrs-based type classes that are shared between
Ray and Spark processing implementations. All types are immutable
(frozen) for thread safety and predictable behavior.
"""

import logging
import os

import attrs
import requests

from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from clients.s3 import S3ClientWrapper
from clients.weaviate import WeaviateDataObject
from config import EmbeddingConfig
from core.models import VectorPoint

logger = logging.getLogger("pipeline.ingestion")


# =============================================================================
# Result Types
# =============================================================================


@attrs.define(frozen=True, slots=True)
class ProcessingResult:
    """Result of processing a single S3 object.

    Attributes:
        s3_key: The S3 object key that was processed.
        success: True if processing succeeded, False otherwise.
        error_message: Empty string if successful, error message if failed.

    Example:
        >>> result = ProcessingResult(s3_key="inputs/doc.txt", success=True)
        >>> result.to_tuple()
        ('inputs/doc.txt', True, '')
    """

    s3_key: str
    success: bool
    error_message: str = ""

    def to_tuple(self) -> tuple[str, bool, str]:
        """Convert to legacy tuple format for Ray/Spark compatibility."""
        return (self.s3_key, self.success, self.error_message)

    @classmethod
    def success_result(cls, s3_key: str) -> "ProcessingResult":
        """Create a successful result."""
        return cls(s3_key=s3_key, success=True)

    @classmethod
    def failure_result(cls, s3_key: str, error: str) -> "ProcessingResult":
        """Create a failure result."""
        return cls(s3_key=s3_key, success=False, error_message=error)


@attrs.define(frozen=True, slots=True)
class ContentResult:
    """Downloaded S3 content.

    Attributes:
        s3_key: The S3 object key.
        content: The decoded text content.

    Example:
        >>> content = ContentResult(s3_key="inputs/doc.txt", content="Hello world")
    """

    s3_key: str
    content: str


@attrs.define(frozen=True, slots=True)
class BatchEmbeddingResult:
    """Result of batch embedding generation.

    Contains vector points ready for upserting to both Qdrant and Weaviate.

    Attributes:
        qdrant_points: List of VectorPoint objects for Qdrant.
        weaviate_objects: List of WeaviateDataObject objects for Weaviate.
    """

    qdrant_points: list[VectorPoint]
    weaviate_objects: list[WeaviateDataObject]

    def __len__(self) -> int:
        """Return the number of points in the batch."""
        return len(self.qdrant_points)

    def is_empty(self) -> bool:
        """Check if the batch is empty."""
        return len(self.qdrant_points) == 0


@attrs.define(frozen=True, slots=True)
class UpsertResult:
    """Result of upserting to vector databases.

    Tracks success/failure for each database independently to ensure
    accurate reporting and enable targeted retries.

    Attributes:
        qdrant_success: True if Qdrant upsert succeeded.
        weaviate_success: True if Weaviate upsert succeeded.
        qdrant_error: Error message if Qdrant failed, empty string otherwise.
        weaviate_error: Error message if Weaviate failed, empty string otherwise.
        batch_size: Number of points in the batch.

    Example:
        >>> result = UpsertResult(qdrant_success=True, weaviate_success=False,
        ...                       weaviate_error="Timeout", batch_size=50)
        >>> result.any_success  # True
        >>> result.all_success  # False
    """

    qdrant_success: bool
    weaviate_success: bool
    qdrant_error: str = ""
    weaviate_error: str = ""
    batch_size: int = 0

    @property
    def any_success(self) -> bool:
        """Return True if at least one database succeeded."""
        return self.qdrant_success or self.weaviate_success

    @property
    def all_success(self) -> bool:
        """Return True if both databases succeeded."""
        return self.qdrant_success and self.weaviate_success

    @classmethod
    def both_success(cls, batch_size: int) -> "UpsertResult":
        """Create result for successful upsert to both databases."""
        return cls(qdrant_success=True, weaviate_success=True, batch_size=batch_size)

    @classmethod
    def empty(cls) -> "UpsertResult":
        """Create result for empty batch (no-op, counts as success)."""
        return cls(qdrant_success=True, weaviate_success=True, batch_size=0)


# =============================================================================
# Configuration Types
# =============================================================================


@attrs.define(frozen=True, slots=True)
class ProcessingConfig:
    """Configuration for S3-to-vector-DB processing pipeline.

    This is the base configuration shared between Ray and Spark implementations.
    Framework-specific configs can extend this with additional fields.

    Attributes:
        s3_endpoint: S3 service endpoint URL.
        s3_access_key: S3 access key ID.
        s3_secret_key: S3 secret access key.
        s3_bucket: S3 bucket name.
        embedding_config: Configuration for the embedding provider.
        collection: Vector database collection name.
        embed_batch_size: Number of texts to embed per API call.
        upsert_batch_size: Number of vectors to upsert per batch.
        namespace: Kubernetes namespace for service discovery.

    Example:
        >>> config = ProcessingConfig(
        ...     s3_endpoint="http://minio:9000",
        ...     s3_access_key="minioadmin",
        ...     s3_secret_key="minioadmin",
        ...     s3_bucket="pipeline",
        ...     embedding_config=EmbeddingConfig.from_env(),
        ...     collection="documents",
        ... )
    """

    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    embedding_config: EmbeddingConfig
    collection: str
    embed_batch_size: int = 8
    upsert_batch_size: int = 200
    namespace: str = attrs.field(factory=lambda: os.getenv("K8S_NAMESPACE", "ml-system"))


@attrs.define(frozen=True, slots=True)
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_second: Maximum requests per second.
        max_concurrency: Maximum concurrent requests.
    """

    requests_per_second: int = 5
    max_concurrency: int = 10


# =============================================================================
# Client Bundle
# =============================================================================


@attrs.define(frozen=False, slots=True)
class ProcessingClients:
    """Bundle of clients needed for processing.

    Groups all external service clients used during processing. Provides
    methods for graceful cleanup of resources.

    Attributes:
        s3: S3 client wrapper for object storage.
        embedder: Embedding provider for vector generation.
        qdrant_db: Qdrant vector database provider.
        weaviate_db: Weaviate vector database provider.
        session: HTTP session for connection pooling.

    Example:
        >>> clients = ProcessingClients(s3=s3, embedder=ollama, ...)
        >>> try:
        ...     # Use clients
        ... finally:
        ...     clients.close_sync()  # or await clients.close_async()
    """

    s3: S3ClientWrapper
    embedder: EmbeddingProvider
    qdrant_db: VectorDBProvider
    weaviate_db: VectorDBProvider
    session: requests.Session

    def close_sync(self) -> None:
        """Close all clients gracefully (synchronous).

        Use this in Ray remote functions or synchronous code.
        Logs warnings for any errors during cleanup.
        """
        try:
            self.qdrant_db.close()
        except Exception as e:
            logger.warning("Error closing Qdrant client", extra={"error": str(e)})

        try:
            self.weaviate_db.close()
        except Exception as e:
            logger.warning("Error closing Weaviate client", extra={"error": str(e)})

        try:
            self.session.close()
        except Exception as e:
            logger.warning("Error closing session", extra={"error": str(e)})

    async def close_async(self) -> None:
        """Close all clients gracefully (asynchronous).

        Use this in Spark async partition processing or other async code.
        Handles async client cleanup properly.
        """
        # Close Qdrant async client
        if self.qdrant_db._client is not None:
            try:
                await self.qdrant_db._client.close()
            except Exception as e:
                logger.warning("Error closing Qdrant async client", extra={"error": str(e)})

        # Close Weaviate async client
        if self.weaviate_db._client is not None:
            try:
                await self.weaviate_db._client.close()
            except Exception as e:
                logger.warning("Error closing Weaviate async client", extra={"error": str(e)})

        # Close HTTP session (aiohttp-style close)
        try:
            await self.session.close()
        except Exception as e:
            logger.warning("Error closing HTTP session", extra={"error": str(e)})
