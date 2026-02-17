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

from .types import (
    BatchEmbeddingResult,
    ContentResult,
    ImageContentResult,
    ProcessingResult,
    UpsertResult,
)

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

    async def _fetch_one_async(self, key: str) -> ContentResult | None:
        """Fetch content from a single S3 object asynchronously.

        Args:
            key: S3 object key.

        Returns:
            ContentResult if successful, None if failed.
        """
        try:
            data = await self.s3.get_object_async(bucket=self.bucket, key=key)
            content = data.decode("utf-8")
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

    async def fetch_all_async(
        self,
        keys: list[str],
        max_concurrent: int = 20,
    ) -> tuple[list[ContentResult], list[ProcessingResult]]:
        """Fetch content from all S3 objects concurrently.

        Uses asyncio.gather with a semaphore for concurrency control.

        Args:
            keys: List of S3 object keys.
            max_concurrent: Maximum concurrent S3 fetch operations.

        Returns:
            Tuple of (successful content results, failed processing results).
        """
        if not keys:
            return [], []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_fetch(key: str) -> tuple[str, ContentResult | None]:
            async with semaphore:
                result = await self._fetch_one_async(key)
                return key, result

        fetch_results = await asyncio.gather(*[_bounded_fetch(key) for key in keys])

        contents: list[ContentResult] = []
        failures: list[ProcessingResult] = []
        for key, result in fetch_results:
            if result is not None:
                contents.append(result)
            else:
                failures.append(ProcessingResult.failure_result(key, "S3 fetch failed in fetch_all_async"))

        return contents, failures


# =============================================================================
# Image Format Detection Constants
# =============================================================================

# Magic bytes for supported image formats
# Reference: https://en.wikipedia.org/wiki/List_of_file_signatures
_IMAGE_MAGIC_BYTES = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "gif": (b"GIF87a", b"GIF89a"),
    "webp": None,  # Special handling: RIFF....WEBP
}

# Supported image extensions (for filtering S3 keys)
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

# Known non-image extensions to reject during filtering
_NON_IMAGE_EXTENSIONS = frozenset(
    {
        ".txt",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".pdf",
        ".md",
        ".yaml",
        ".yml",
        ".log",
        ".sql",
        ".parquet",
        ".avro",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
    }
)


def detect_image_format(data: bytes) -> str | None:
    r"""Detect image format from magic bytes.

    Args:
        data: Raw image bytes (at least first 12 bytes needed).

    Returns:
        Format string ("jpeg", "png", "gif", "webp") or None if unrecognized.

    Example:
        >>> detect_image_format(b"\xff\xd8\xff...")
        'jpeg'
        >>> detect_image_format(b"\x89PNG\r\n\x1a\n...")
        'png'
    """
    if len(data) < 12:
        return None

    # Check JPEG
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"

    # Check PNG
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # Check GIF (87a or 89a)
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"

    # Check WebP: starts with RIFF, has WEBP at offset 8
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"

    return None


class ImageContentFetcher:
    """Fetches binary image content from S3 objects.

    Similar to S3ContentFetcher but for images. Returns raw bytes with
    metadata including detected format from magic bytes and basic
    validation (size limits, format support).

    Does NOT perform any image preprocessing (resizing, normalization).
    That is handled by the image preprocessing utilities.

    Example:
        >>> fetcher = ImageContentFetcher(s3_client, bucket="pipeline")
        >>> result = fetcher.fetch_one("images/photo.jpg")
        >>> if result:
        ...     print(f"Format: {result.format}, Size: {result.size_bytes}")
    """

    # Default size limits (in bytes)
    DEFAULT_MIN_SIZE = 100  # 100 bytes - reject tiny/corrupt images
    DEFAULT_MAX_SIZE = 50 * 1024 * 1024  # 50 MB

    def __init__(
        self,
        s3_client: S3ClientWrapper,
        bucket: str,
        min_size_bytes: int = DEFAULT_MIN_SIZE,
        max_size_bytes: int = DEFAULT_MAX_SIZE,
    ) -> None:
        """Initialize the image fetcher.

        Args:
            s3_client: S3 client wrapper.
            bucket: S3 bucket name.
            min_size_bytes: Minimum valid image size in bytes.
            max_size_bytes: Maximum valid image size in bytes.
        """
        self.s3 = s3_client
        self.bucket = bucket
        self.min_size_bytes = min_size_bytes
        self.max_size_bytes = max_size_bytes

    def _validate_image(
        self,
        key: str,
        data: bytes,
        detected_format: str | None,
    ) -> tuple[bool, str]:
        """Validate image data.

        Args:
            key: S3 object key (for error messages).
            data: Raw image bytes.
            detected_format: Detected format from magic bytes.

        Returns:
            Tuple of (is_valid, error_message).
        """
        size = len(data)

        # Check size limits
        if size < self.min_size_bytes:
            return False, f"Image too small: {size} bytes (min: {self.min_size_bytes})"

        if size > self.max_size_bytes:
            return False, f"Image too large: {size} bytes (max: {self.max_size_bytes})"

        # Check format
        if detected_format is None:
            return False, "Unsupported or unrecognized image format"

        return True, ""

    def fetch_one(self, key: str) -> ImageContentResult | None:
        """Fetch image from a single S3 object.

        Downloads the image, detects format from magic bytes, and performs
        basic validation. Does NOT extract dimensions (expensive operation).

        Args:
            key: S3 object key.

        Returns:
            ImageContentResult if successful and valid, None if failed.
        """
        try:
            data = self.s3.get_object(bucket=self.bucket, key=key)

            # Detect format from magic bytes
            detected_format = detect_image_format(data)

            # Validate image
            is_valid, error_msg = self._validate_image(key, data, detected_format)
            if not is_valid:
                logger.warning(
                    "Image validation failed",
                    extra={"s3_key": key, "error": error_msg},
                )
                return None

            # detected_format is guaranteed non-None here due to validation
            return ImageContentResult(
                s3_key=key,
                image_bytes=data,
                format=detected_format,  # type: ignore[arg-type]
                size_bytes=len(data),
            )

        except (UpstreamError, ClientError, OSError, ValueError) as e:
            logger.error(
                "Failed to fetch S3 image object",
                extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            return None
        except (RuntimeError, MemoryError, AttributeError, TypeError) as e:
            logger.error(
                "Unexpected error fetching S3 image object",
                extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            return None

    def fetch_one_with_dimensions(self, key: str) -> ImageContentResult | None:
        """Fetch image and extract dimensions using PIL.

        More expensive than fetch_one() as it requires parsing the image.
        Use when dimensions are needed for metadata.

        Args:
            key: S3 object key.

        Returns:
            ImageContentResult with width/height populated, or None if failed.
        """
        result = self.fetch_one(key)
        if result is None:
            return None

        try:
            # Import PIL only when needed (optional dependency for basic fetch)
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(result.image_bytes)) as img:
                width, height = img.size

            # Create new result with dimensions (attrs is frozen)
            return ImageContentResult(
                s3_key=result.s3_key,
                image_bytes=result.image_bytes,
                format=result.format,
                size_bytes=result.size_bytes,
                width=width,
                height=height,
            )
        except ImportError:
            logger.warning(
                "PIL not available for dimension extraction",
                extra={"s3_key": key},
            )
            return result
        except Exception as e:
            logger.warning(
                "Failed to extract image dimensions",
                extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
            )
            return result

    def fetch_all(
        self,
        keys: list[str],
        *,
        with_dimensions: bool = False,
    ) -> tuple[list[ImageContentResult], list[ProcessingResult]]:
        """Fetch images from all S3 objects.

        Args:
            keys: List of S3 object keys.
            with_dimensions: If True, extract dimensions using PIL.

        Returns:
            Tuple of (successful image results, failed processing results).
        """
        images: list[ImageContentResult] = []
        failures: list[ProcessingResult] = []

        fetch_fn = self.fetch_one_with_dimensions if with_dimensions else self.fetch_one

        for key in keys:
            result = fetch_fn(key)
            if result is not None:
                images.append(result)
            else:
                failures.append(ProcessingResult.failure_result(key, "Image fetch/validation failed"))

        return images, failures

    async def _fetch_one_async(self, key: str, *, with_dimensions: bool = False) -> ImageContentResult | None:
        """Fetch image from a single S3 object asynchronously.

        Args:
            key: S3 object key.
            with_dimensions: If True, extract dimensions using PIL.

        Returns:
            ImageContentResult if successful, None if failed.
        """
        try:
            data = await self.s3.get_object_async(bucket=self.bucket, key=key)

            detected_format = detect_image_format(data)
            is_valid, error_msg = self._validate_image(key, data, detected_format)
            if not is_valid:
                logger.warning(
                    "Image validation failed",
                    extra={"s3_key": key, "error": error_msg},
                )
                return None

            result = ImageContentResult(
                s3_key=key,
                image_bytes=data,
                format=detected_format,  # type: ignore[arg-type]
                size_bytes=len(data),
            )

            if with_dimensions:
                try:
                    from io import BytesIO

                    from PIL import Image

                    with Image.open(BytesIO(result.image_bytes)) as img:
                        width, height = img.size
                    result = ImageContentResult(
                        s3_key=result.s3_key,
                        image_bytes=result.image_bytes,
                        format=result.format,
                        size_bytes=result.size_bytes,
                        width=width,
                        height=height,
                    )
                except ImportError:
                    logger.warning("PIL not available for dimension extraction", extra={"s3_key": key})
                except Exception as e:
                    logger.warning(
                        "Failed to extract image dimensions",
                        extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
                    )

            return result

        except (UpstreamError, ClientError, OSError, ValueError) as e:
            logger.error(
                "Failed to fetch S3 image object",
                extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            return None
        except (RuntimeError, MemoryError, AttributeError, TypeError) as e:
            logger.error(
                "Unexpected error fetching S3 image object",
                extra={"s3_key": key, "error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )
            return None

    async def fetch_all_async(
        self,
        keys: list[str],
        *,
        with_dimensions: bool = False,
        max_concurrent: int = 20,
    ) -> tuple[list[ImageContentResult], list[ProcessingResult]]:
        """Fetch images from all S3 objects concurrently.

        Uses asyncio.gather with a semaphore for concurrency control.

        Args:
            keys: List of S3 object keys.
            with_dimensions: If True, extract dimensions using PIL.
            max_concurrent: Maximum concurrent S3 fetch operations.

        Returns:
            Tuple of (successful image results, failed processing results).
        """
        if not keys:
            return [], []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_fetch(key: str) -> tuple[str, ImageContentResult | None]:
            async with semaphore:
                result = await self._fetch_one_async(key, with_dimensions=with_dimensions)
                return key, result

        fetch_results = await asyncio.gather(*[_bounded_fetch(key) for key in keys])

        images: list[ImageContentResult] = []
        failures: list[ProcessingResult] = []
        for key, result in fetch_results:
            if result is not None:
                images.append(result)
            else:
                failures.append(
                    ProcessingResult.failure_result(key, "Image fetch/validation failed in fetch_all_async")
                )

        return images, failures

    @staticmethod
    def filter_image_keys(
        keys: list[str],
        mime_type_resolver: object = None,  # deprecated, unused
    ) -> list[str]:
        """Filter S3 keys to likely image files without HEAD requests.

        Strategy:
          1. Key has recognized image extension → accept
          2. Key has NO extension (no '.' in basename) → accept optimistically
          3. Key has a known non-image extension → reject
          4. Key has unknown extension → accept optimistically

        Extensionless keys are accepted because the fetch step validates
        images via ``detect_image_format()`` magic bytes, making listing-time
        validation redundant.

        Args:
            keys: List of S3 object keys.
            mime_type_resolver: Deprecated, unused. Kept for backward compatibility.

        Returns:
            Filtered list containing likely image keys.
        """
        result: list[str] = []
        by_ext = 0
        optimistic = 0
        rejected = 0
        for key in keys:
            basename = key.rsplit("/", 1)[-1] if "/" in key else key
            dot_idx = basename.rfind(".")
            if dot_idx == -1:
                # No extension — accept optimistically
                result.append(key)
                optimistic += 1
                continue
            ext = basename[dot_idx:].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                result.append(key)
                by_ext += 1
            elif ext in _NON_IMAGE_EXTENSIONS:
                rejected += 1
            else:
                # Unknown extension — accept optimistically
                result.append(key)
                optimistic += 1
        logger.info(
            "filter_image_keys: %d accepted (%d by extension, %d optimistic), %d rejected",
            len(result),
            by_ext,
            optimistic,
            rejected,
        )
        return result


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
                vectors = await self.embedder.embed_batch_async(texts)
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

    Handles parallel upserts to targeted databases with error handling
    and partial failure support. Non-targeted databases (None) are treated
    as successful no-ops.

    Example:
        >>> upserter = VectorDBUpserter(qdrant_db, weaviate_db)
        >>> success = await upserter.upsert_batch_async(batch, "documents", 768)
    """

    def __init__(
        self,
        qdrant_db: VectorDBProvider | None,
        weaviate_db: VectorDBProvider | None,
    ) -> None:
        """Initialize the upserter.

        Args:
            qdrant_db: Qdrant vector database provider (None if not targeted).
            weaviate_db: Weaviate vector database provider (None if not targeted).
        """
        self.qdrant_db = qdrant_db
        self.weaviate_db = weaviate_db

    async def upsert_batch_async(
        self,
        embedding_result: BatchEmbeddingResult,
        collection: str,
        vector_size: int,
    ) -> UpsertResult:
        """Upsert vectors to targeted databases in parallel.

        Non-targeted databases (provider is None) default to success in the
        result without performing any I/O.

        Args:
            embedding_result: Batch of points to upsert.
            collection: Collection name.
            vector_size: Vector dimension.

        Returns:
            UpsertResult with per-database success/failure status.
        """
        if embedding_result.is_empty():
            return UpsertResult.noop()

        batch_size = len(embedding_result)

        qdrant_success = True
        weaviate_success = True
        qdrant_error = ""
        weaviate_error = ""

        # Build tasks only for targeted databases
        tasks: list[asyncio.Task] = []
        task_labels: list[str] = []

        if self.qdrant_db is not None and embedding_result.qdrant_points:
            qdrant_point_structs = [point.to_qdrant() for point in embedding_result.qdrant_points]
            tasks.append(
                asyncio.ensure_future(
                    self.qdrant_db.batch_upsert_async(
                        collection=collection,
                        points=qdrant_point_structs,  # type: ignore[arg-type]
                        vector_size=vector_size,
                    )
                )
            )
            task_labels.append("qdrant")

        if self.weaviate_db is not None and embedding_result.weaviate_objects:
            tasks.append(
                asyncio.ensure_future(
                    self.weaviate_db.batch_upsert_async(
                        collection=collection,
                        points=embedding_result.weaviate_objects,  # type: ignore[arg-type]
                        vector_size=vector_size,
                    )
                )
            )
            task_labels.append("weaviate")

        if tasks:
            upsert_results = await asyncio.gather(*tasks, return_exceptions=True)

            for label, result in zip(task_labels, upsert_results, strict=True):
                if isinstance(result, Exception):
                    if label == "qdrant":
                        qdrant_success = False
                        qdrant_error = f"{type(result).__name__}: {result}"
                        logger.error(
                            "Qdrant batch upsert failed",
                            extra={
                                "error": str(result),
                                "error_type": type(result).__name__,
                                "batch_size": batch_size,
                            },
                            exc_info=result,
                        )
                    elif label == "weaviate":
                        weaviate_success = False
                        weaviate_error = f"{type(result).__name__}: {result}"
                        logger.error(
                            "Weaviate batch upsert failed",
                            extra={
                                "error": str(result),
                                "error_type": type(result).__name__,
                                "batch_size": batch_size,
                            },
                            exc_info=result,
                        )

        return UpsertResult(
            qdrant_success=qdrant_success,
            weaviate_success=weaviate_success,
            qdrant_error=qdrant_error,
            weaviate_error=weaviate_error,
            batch_size=batch_size,
            qdrant_enabled=self.qdrant_db is not None,
            weaviate_enabled=self.weaviate_db is not None,
        )


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
        upsert_result = await self.upserter.upsert_batch_async(embedding_result, self.collection, vector_size)

        # Log per-DB status for observability
        if not upsert_result.all_success:
            failed_dbs = []
            if upsert_result.qdrant_enabled and not upsert_result.qdrant_success:
                failed_dbs.append(f"Qdrant: {upsert_result.qdrant_error}")
            if upsert_result.weaviate_enabled and not upsert_result.weaviate_success:
                failed_dbs.append(f"Weaviate: {upsert_result.weaviate_error}")
            logger.warning(
                "Partial upsert failure: %s",
                "; ".join(failed_dbs),
                extra={
                    "batch_size": upsert_result.batch_size,
                    "qdrant_success": upsert_result.qdrant_success,
                    "weaviate_success": upsert_result.weaviate_success,
                    "qdrant_enabled": upsert_result.qdrant_enabled,
                    "weaviate_enabled": upsert_result.weaviate_enabled,
                },
            )

        if upsert_result.any_success:
            # At least one DB succeeded - grow batch size cautiously
            new_size = min(current_batch_size + 1, max_batch_size)
            return [ProcessingResult.success_result(c.s3_key) for c in batch], new_size
        # Both DBs failed - shrink batch size
        new_size = max(current_batch_size // 2, min_batch_size)
        return [ProcessingResult.failure_result(c.s3_key, "Upsert failed") for c in batch], new_size
