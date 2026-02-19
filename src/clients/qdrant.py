"""Qdrant client wrapper class for vector database operations.

This module provides a Qdrant client wrapper class that encapsulates
configuration and provides methods for collection management and vector
operations. This replaces the functional API with an object-oriented approach
using attrs.

## Usage

```python
from clients.qdrant import QdrantClientWrapper

client = QdrantClientWrapper(url="http://qdrant.ml-system:6333")
client.ensure_collection_async(collection="documents", vector_size=768)
results = client.search_async(collection="documents", query_vector=[...], limit=10)

# For image embeddings (use a distinct base name; creates "photos_images" in Qdrant)
await client.ensure_image_collection_async(collection="photos")
# Creates "photos_images" collection with 512-dimensional vectors (CLIP default)
```

## Design

The client wrapper:
- Encapsulates Qdrant connection configuration
- Provides a clean interface for collection and search operations
- Handles errors consistently via `UpstreamError`
- Uses attrs for concise, correct class definition
"""

import logging
from typing import Any, Literal

import aiobreaker
import attrs
import httpx
import pybreaker
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import PointStruct  # Qdrant's native point type

from config import VectorDBConfig
from core.exceptions import UpstreamError
from core.models import SearchResultItem, SearchResults
from foundation.async_utils import close_async_resource, run_coroutine_sync
from foundation.circuit_breaker import create_async_circuit_breaker, with_circuit_breaker_async
from foundation.retry import HTTPErrorClassifier, async_retry_call, create_retry_logger

from .base import VectorDBClientBase
from .interfaces.vector_db import VectorDBProvider

_retry_logger = logging.getLogger("clients.qdrant.retry")


# =============================================================================
# Qdrant Error Classifier
# =============================================================================


class QdrantErrorClassifier(HTTPErrorClassifier):
    """Qdrant-specific error classification for retry logic."""

    def is_retriable(self, exc: BaseException) -> bool:
        """Classify whether the exception is retriable."""
        # Qdrant SDK response handling errors (serialization, transport)
        if isinstance(exc, ResponseHandlingException):
            return True
        # Qdrant HTTP errors: classify by status code
        if isinstance(exc, UnexpectedResponse):
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                return self.is_retriable_http_status(status_code)
            return True  # No status code implies transport error
        # httpx transport errors (qdrant-client uses httpx internally)
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
            return True
        if isinstance(exc, UpstreamError):
            return False
        return False

    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        """Extract structured error details for logging."""
        if isinstance(exc, UnexpectedResponse):
            return {"http_status": getattr(exc, "status_code", None)}
        return {}


_qdrant_classifier = QdrantErrorClassifier()
_qdrant_log_retry = create_retry_logger(
    _retry_logger,
    _qdrant_classifier.get_error_details,
    "Qdrant async operation failed, retrying",
)

# Type alias for supported distance metrics
DistanceMetric = Literal["cosine", "euclidean", "dot"]

# Mapping from string distance metric to Qdrant Distance enum
_DISTANCE_METRIC_MAP: dict[str, qmodels.Distance] = {
    "cosine": qmodels.Distance.COSINE,
    "euclidean": qmodels.Distance.EUCLID,
    "dot": qmodels.Distance.DOT,
}


@attrs.define(frozen=False, slots=True)
class QdrantClientWrapper(VectorDBClientBase, VectorDBProvider):
    """Wrapper for Qdrant async client with convenience methods.

    This client is async-only. All operations use AsyncQdrantClient internally.

    Attributes:
        url: Qdrant service URL (e.g., `http://qdrant.ml-system:6333`).
        api_key: API key for authenticated instances. Default: None.
        timeout_s: Request timeout in seconds. Default: 300.
        circuit_breaker_threshold: Failures before circuit opens. Default: 3.
        circuit_breaker_timeout: Circuit recovery timeout in seconds. Default: 60.
    """

    url: str
    api_key: str | None = None
    timeout_s: int = attrs.field(default=300)
    circuit_breaker_threshold: int = attrs.field(default=3)
    circuit_breaker_timeout: int = attrs.field(default=60)
    max_retries: int = attrs.field(default=3)
    retry_min_wait: float = attrs.field(default=1.0)
    retry_max_wait: float = attrs.field(default=10.0)
    _client: AsyncQdrantClient = attrs.field(init=False, default=None)
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)
    _async_breaker: aiobreaker.CircuitBreaker = attrs.field(init=False)

    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for Qdrant.

        Qdrant is on critical path (blocks user search requests).

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return ("qdrant", self.circuit_breaker_threshold, self.circuit_breaker_timeout)

    def __attrs_post_init__(self) -> None:
        """Initialize the Qdrant async client and circuit breaker."""
        self._client = AsyncQdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout_s)

        # Initialize circuit breaker from base class
        self._init_circuit_breaker()

        # Initialize async circuit breaker using aiobreaker
        name, fail_max, timeout = self._circuit_breaker_config()
        object.__setattr__(
            self,
            "_async_breaker",
            create_async_circuit_breaker(name, fail_max, timeout),
        )

    @property
    def client(self) -> AsyncQdrantClient:
        """Get the underlying AsyncQdrantClient instance."""
        return self._client

    @classmethod
    def from_config(cls, config: VectorDBConfig) -> "QdrantClientWrapper":
        """Create QdrantClientWrapper from VectorDBConfig.

        Args:
            config: Vector database configuration with Qdrant settings.

        Returns:
            Configured QdrantClientWrapper instance.

        Raises:
            ValueError: If Qdrant config is missing or invalid.

        Example:
            ```python
            from config import VectorDBConfig
            from clients.qdrant import QdrantClientWrapper

            config = VectorDBConfig.from_env()
            client = QdrantClientWrapper.from_config(config)
            ```
        """
        cls._validate_config(config, "qdrant", ["qdrant_url"])
        # Type narrowing: _validate_config ensures qdrant_url is not None
        assert config.qdrant_url is not None
        return cls(
            url=config.qdrant_url,
            api_key=getattr(config, "qdrant_api_key", None),
            timeout_s=getattr(config, "qdrant_timeout", 300),
            circuit_breaker_threshold=getattr(config, "qdrant_circuit_breaker_threshold", 3),
            circuit_breaker_timeout=getattr(config, "qdrant_circuit_breaker_timeout", 60),
        )

    @with_circuit_breaker_async("qdrant")
    async def ensure_collection_async(
        self,
        *,
        collection: str,
        vector_size: int,
    ) -> None:
        """Create a Qdrant collection if it does not already exist.

        This is a dev convenience function that checks for collection existence
        and creates it with sensible defaults if missing. In production, you
        might want to manage collections via infrastructure-as-code or separate
        tooling.

        Args:
            collection: Collection name to ensure exists.
            vector_size: Dimension of vectors that will be stored in this
            collection.  Must match the embedding dimension from your model
            (e.g., 768 for `nomic-embed-text`).

        Note:
            The collection is created with:
            - **Distance metric**: Cosine similarity (standard for embeddings)
            - **Vector size**: As specified by `vector_size`

            If the collection already exists, this function does nothing
            (no-op).
        """

        async def _do_ensure() -> None:
            existing_collections = await self._client.get_collections()
            existing = {c.name for c in existing_collections.collections}
            if collection in existing:
                return
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

        await async_retry_call(
            _do_ensure,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_qdrant_classifier.is_retriable,
            before_sleep=_qdrant_log_retry,
            operation="Qdrant ensure_collection",
        )

    @with_circuit_breaker_async("qdrant")
    async def ensure_image_collection_async(
        self,
        *,
        collection: str,
        vector_size: int = 512,
        distance_metric: DistanceMetric = "cosine",
    ) -> None:
        """Create a Qdrant collection for image embeddings if it does not already exist.

        This method creates a collection specifically for image embeddings with the
        naming pattern `{collection}_images`. For example, if `collection="documents"`,
        the created collection will be named `documents_images`.

        Image collections use a different schema than text collections, storing
        image-specific metadata in the payload:
        - `s3_key`: S3 object key for the original image
        - `s3_uri`: Full S3 URI (s3://bucket/key)
        - `format`: Image format (jpeg, png, webp, gif)
        - `width`: Image width in pixels (optional)
        - `height`: Image height in pixels (optional)
        - `thumbnail_key`: S3 key for thumbnail version (optional)

        Args:
            collection: Base collection name. The actual collection will be named
                `{collection}_images` (e.g., `documents_images`).
            vector_size: Dimension of vectors that will be stored in this collection.
                Default: 512 (common for CLIP models like ViT-B/32).
            distance_metric: Distance metric for vector similarity. One of "cosine",
                "euclidean", or "dot". Default "cosine" (standard for embeddings).

        Note:
            The collection is created with:
            - **Distance metric**: As specified by `distance_metric` (default: cosine)
            - **Vector size**: As specified by `vector_size` (default: 512 for CLIP)
            - **Collection name**: `{collection}_images`

            If the collection already exists, this function does nothing (no-op).

        Example:
            ```python
            # Create image collection (base name; Qdrant collection will be "{base}_images")
            await client.ensure_image_collection_async(collection="photos")
            # Creates collection named "photos_images" with 512-dimensional vectors

            # Custom vector size and distance metric
            await client.ensure_image_collection_async(
                collection="observations",
                vector_size=768,
                distance_metric="dot"
            )
            # Creates collection named "observations_images" with 768-dimensional vectors
            ```
        """
        image_collection = f"{collection}_images"

        async def _do_ensure_image() -> None:
            existing_collections = await self._client.get_collections()
            existing = {c.name for c in existing_collections.collections}
            if image_collection in existing:
                return
            try:
                await self._client.create_collection(
                    collection_name=image_collection,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size, distance=_DISTANCE_METRIC_MAP[distance_metric]
                    ),
                )
            except UnexpectedResponse as e:
                status_code = getattr(e, "status_code", None)
                if status_code == 409:
                    self._logger.warning(  # type: ignore[attr-defined]
                        "Qdrant image collection already exists; skipping create",
                        extra={"collection": image_collection},
                    )
                    return
                raise

        await async_retry_call(
            _do_ensure_image,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_qdrant_classifier.is_retriable,
            before_sleep=_qdrant_log_retry,
            operation="Qdrant ensure_image_collection",
        )

    @with_circuit_breaker_async("qdrant")
    async def search_async(
        self, *, collection: str, query_vector: list[float], limit: int = 10
    ) -> SearchResults:
        """Search for similar vectors in a Qdrant collection.

        Args:
            collection: Collection name to search.
            query_vector: Query embedding vector (must match collection
            dimension).
            limit: Maximum number of results to return.

        Returns:
            A `SearchResults` instance containing:
            - `items`: List of search result items, ordered by similarity
            (highest first)
            - `total`: Total number of results found

        Raises:
            UpstreamError: If collection doesn't exist or search fails. Also
            raised when circuit breaker is open.

        Example:
            ```python
            results = await client.search_async(
                collection="documents",
                query_vector=[0.1, 0.2, ...],  # 768-dimensional vector
                limit=10
            )
            # Access: results.items[0].point_id, results.items[0].score, etc.
            ```
        """

        async def _do_search() -> SearchResults:
            query_response = await self._client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

            items = [
                SearchResultItem(
                    point_id=str(point.id),
                    score=float(point.score) if point.score is not None else 0.0,
                    payload=point.payload or {},
                )
                for point in query_response.points
            ]

            return SearchResults(items=items, total=len(items))

        return await async_retry_call(
            _do_search,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_qdrant_classifier.is_retriable,
            before_sleep=_qdrant_log_retry,
            operation="Qdrant search",
        )

    @with_circuit_breaker_async("qdrant")
    async def disable_indexing(self, *, collection: str) -> None:
        """Disable indexing for a collection during bulk operations.

        This method optimizes performance during bulk data ingestion by
        disabling the HNSW index building. The index will be built after
        re-enabling indexing.

        Args:
            collection: Collection name to disable indexing for.

        Raises:
            UpstreamError: If Qdrant operations fail.

        Note:
            This should be called before bulk operations and followed by
            `enable_indexing()` after the bulk load is complete.
        """

        async def _do_disable() -> None:
            await self._client.update_collection(
                collection_name=collection,
                optimizer_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=0,
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=0),
            )
            self._logger.info(  # type: ignore[attr-defined]
                "Disabled indexing for collection",
                extra={"collection": collection},
            )

        await async_retry_call(
            _do_disable,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_qdrant_classifier.is_retriable,
            before_sleep=_qdrant_log_retry,
            operation="Qdrant disable_indexing",
        )

    @with_circuit_breaker_async("qdrant")
    async def enable_indexing(
        self,
        *,
        collection: str,
        indexing_threshold: int = 20000,
        hnsw_m: int = 16,
    ) -> None:
        """Re-enable indexing for a collection after bulk operations.

        This method re-enables indexing with default or custom parameters.
        Qdrant will build the index in the background.

        Args:
            collection: Collection name to enable indexing for.
            indexing_threshold: Number of points before indexing starts
            (default: 20000).  hnsw_m: HNSW parameter m (default: 16).

        Raises:
            UpstreamError: If Qdrant operations fail.

        Note:
            This should be called after bulk operations are complete.
        """

        async def _do_enable() -> None:
            await self._client.update_collection(
                collection_name=collection,
                optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
                hnsw_config=qmodels.HnswConfigDiff(m=hnsw_m),
            )
            self._logger.info(  # type: ignore[attr-defined]
                "Re-enabled indexing for collection",
                extra={
                    "collection": collection,
                    "indexing_threshold": indexing_threshold,
                    "hnsw_m": hnsw_m,
                },
            )

        await async_retry_call(
            _do_enable,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_qdrant_classifier.is_retriable,
            before_sleep=_qdrant_log_retry,
            operation="Qdrant enable_indexing",
        )

    @with_circuit_breaker_async("qdrant")
    async def _do_batch_upsert(self, *, collection: str, points: list[PointStruct]) -> None:
        """Qdrant-specific batch upsert implementation with retry.

        Args:
            collection: Collection name to upsert into.
            points: List of Qdrant PointStruct instances to upsert.

        Raises:
            Exception: Any Qdrant-specific exceptions (wrapped by base class).
        """

        async def _do_upsert() -> None:
            await self._client.upsert(collection_name=collection, points=points)

        await async_retry_call(
            _do_upsert,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_qdrant_classifier.is_retriable,
            before_sleep=_qdrant_log_retry,
            operation="Qdrant batch_upsert",
        )

    async def batch_upsert_async(
        self,
        *,
        collection: str,
        points: list[PointStruct],  # type: ignore[override]
        vector_size: int,
    ) -> None:
        """Batch upsert points into a Qdrant collection (async version).

        This method uses the base class template which ensures the collection
        exists before upserting and performs batch upserts for better
        performance.

        Args:
            collection: Collection name to upsert into.
            points: List of points to upsert. Must not be empty.
            vector_size: Vector dimension (e.g., 768 for nomic-embed-text).
                Used to ensure collection exists with correct dimensions.

        Raises:
            UpstreamError: If Qdrant operations fail. Also raised when circuit
                breaker is open.

        Example:
            ```python
            points = [
                PointStruct(
                    id="1", vector=[0.1, 0.2, ...], payload={"text": "hello"},
                ),
                PointStruct(
                    id="2", vector=[0.3, 0.4, ...], payload={"text": "world"},
                ),
            ]
            await client.batch_upsert_async(
                collection="documents",
                points=points,
                vector_size=768
            )
            ```

        Note:
            The collection is automatically created if it doesn't exist.
            Empty point lists are ignored (no-op).
        """
        # Use base class template method for common logic
        await VectorDBClientBase.batch_upsert_async(
            self, collection=collection, points=points, vector_size=vector_size
        )

    def close(self) -> None:
        """Close the Qdrant async client and release resources.

        The async client close operation uses the `close_async_resource`
        utility which handles three scenarios automatically:

        1. **Running event loop** (e.g., called from async context):
           - Schedules close as a background task
           - Errors are logged but don't propagate (since we can't await)

        2. **Stopped event loop** (e.g., called during cleanup):
           - Runs close synchronously using `run_until_complete()`
           - Errors are caught and logged

        3. **No event loop** (e.g., called from synchronous context):
           - Creates a new event loop using `asyncio.run()`
           - Errors are caught and logged

        **Error Handling:**
        This method never raises exceptions. All errors are caught, logged with
        full context, and suppressed to ensure cleanup completes.

        Note:
            This method is idempotent - calling it multiple times is safe.
            After the first call, subsequent calls are no-ops since the client
            reference is set to None.

        Example:
            ```python
            client = QdrantClientWrapper(url="http://qdrant:6333")
            try:
                # Use client...
                pass
            finally:
                client.close()  # Always closes, even if errors occur
            ```
        """
        if self._client is not None:
            client_to_close = self._client
            self._client = None
            run_coroutine_sync(
                close_async_resource(client_to_close, "qdrant_async_client"),
            )
