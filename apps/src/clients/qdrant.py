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
```

## Design

The client wrapper:
- Encapsulates Qdrant connection configuration
- Provides a clean interface for collection and search operations
- Handles errors consistently via `UpstreamError`
- Uses attrs for concise, correct class definition
"""

import asyncio

import attrs
import pybreaker
from config import VectorDBConfig
from core.exceptions import UpstreamError
from core.models import SearchResultItem, SearchResults
from foundation.async_utils import close_async_resource
from foundation.circuit_breaker import handle_circuit_breaker_error
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.models import PointStruct  # Qdrant's native point type

from .base import VectorDBClientBase
from .interfaces.vector_db import VectorDBProvider


@attrs.define(frozen=False, slots=True)
class QdrantClientWrapper(VectorDBClientBase, VectorDBProvider):
    """Wrapper for Qdrant async client with convenience methods.

    Attributes:
        url: Qdrant service URL (e.g., `http://qdrant.ml-system:6333`).

    Note:
        This class uses AsyncQdrantClient internally but provides a sync
        interface to match the VectorDBProvider ABC. Async operations are
        wrapped with asyncio.run().
    """

    url: str
    _client: AsyncQdrantClient = attrs.field(init=False, default=None)
    _sync_client: QdrantClient = attrs.field(init=False, default=None)
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)

    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for Qdrant.

        Qdrant is on critical path (blocks user search requests).

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return ("qdrant", 3, 60)

    def __attrs_post_init__(self) -> None:
        """Initialize the Qdrant async and sync clients and circuit breaker."""
        self._client = AsyncQdrantClient(url=self.url, timeout=300)
        self._sync_client = QdrantClient(url=self.url, timeout=300)

        # Initialize circuit breaker from base class
        self._init_circuit_breaker()

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
        """
        cls._validate_config(config, "qdrant", ["qdrant_url"])
        # Type narrowing: _validate_config ensures qdrant_url is not None
        assert config.qdrant_url is not None
        return cls(url=config.qdrant_url)

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
        existing_collections = await self._client.get_collections()
        existing = {c.name for c in existing_collections.collections}
        if collection in existing:
            return
        await self._client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=qmodels.Distance.COSINE
            ),
        )

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
        # Check circuit breaker state - if open, fail fast
        if self._breaker.current_state == pybreaker.STATE_OPEN:
            handle_circuit_breaker_error("qdrant")

        try:
            qdrant_results = await self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

            items = [
                SearchResultItem(
                    point_id=str(point.id),
                    score=float(point.score),
                    payload=point.payload or {},
                )
                for point in qdrant_results
            ]

            return SearchResults(items=items, total=len(items))
        except Exception as e:
            msg = f"Qdrant search failed: {e}"
            raise UpstreamError(msg) from e

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
        try:
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
        except Exception as e:
            msg = f"Failed to disable indexing: {e}"
            raise UpstreamError(msg) from e

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
        try:
            await self._client.update_collection(
                collection_name=collection,
                optimizer_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=indexing_threshold
                ),
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
        except Exception as e:
            msg = f"Failed to enable indexing: {e}"
            raise UpstreamError(msg) from e

    async def _do_batch_upsert(
        self, *, collection: str, points: list[PointStruct]
    ) -> None:
        """Qdrant-specific batch upsert implementation.

        Args:
            collection: Collection name to upsert into.
            points: List of Qdrant PointStruct instances to upsert.

        Raises:
            Exception: Any Qdrant-specific exceptions (wrapped by base class).
        """
        await self._client.upsert(collection_name=collection, points=points)

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

    def batch_upsert_sync(
        self,
        *,
        collection: str,
        points: list[PointStruct],
        vector_size: int,
    ) -> None:
        """Batch upsert points into a Qdrant collection (synchronous version).

        This method ensures the collection exists before upserting and performs
        batch upserts for better performance. It's designed for high-throughput
        scenarios like Spark job processing where async is not desired.

        Args:
            collection: Collection name to upsert into.
            points: List of points to upsert. Must not be empty.
            vector_size: Vector dimension (e.g., 768 for nomic-embed-text).
                Used to ensure collection exists with correct dimensions.

        Raises:
            UpstreamError: If Qdrant operations fail.

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
            client.batch_upsert_sync(
                collection="documents",
                points=points,
                vector_size=768
            )
            ```

        Note:
            The collection is automatically created if it doesn't exist.
            Empty point lists are ignored (no-op).
        """
        if not points:
            return

        # Ensure collection exists before upserting (sync version)
        try:
            collections = self._sync_client.get_collections().collections
            collection_names = {c.name for c in collections}
            if collection not in collection_names:
                self._sync_client.create_collection(
                    collection_name=collection,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size, distance=qmodels.Distance.COSINE
                    ),
                )
                self._logger.info(  # type: ignore[attr-defined]
                    "Created Qdrant collection",
                    extra={
                        "collection": collection,
                        "vector_size": vector_size,
                    },
                )
        except Exception as e:
            msg = f"Failed to ensure collection exists: {e}"
            raise UpstreamError(msg) from e

        # Batch upsert for better performance (sync version)
        try:
            self._logger.info(  # type: ignore[attr-defined]
                "Calling Qdrant sync upsert",
                extra={"collection": collection, "points_count": len(points)},
            )
            self._sync_client.upsert(collection_name=collection, points=points)
            self._logger.info(  # type: ignore[attr-defined]
                "Qdrant sync upsert completed",
                extra={"collection": collection, "points_count": len(points)},
            )
        except Exception as e:
            self._logger.exception(  # type: ignore[attr-defined]
                "Qdrant sync upsert failed",
                extra={
                    "collection": collection,
                    "points_count": len(points),
                    "error": str(e),
                },
            )
            msg = f"Qdrant batch upsert failed: {e}"
            raise UpstreamError(msg) from e

    def close(self) -> None:
        """Close the Qdrant clients and release resources.

        This method closes both the async and sync Qdrant clients to properly
        release HTTP connections and prevent resource leaks. The clients are
        closed independently - if one fails, the other will still be closed.

        **Event Loop Handling for Async Client:**
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
        full context, and suppressed to ensure both clients are closed even if
        one fails. This is important for cleanup operations where partial
        cleanup is better than no cleanup.

        Note:
            This method is idempotent - calling it multiple times is safe.
            After the first call, subsequent calls are no-ops since the client
            references are set to None.

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
        # Close async client using utility
        if self._client is not None:
            client_to_close = self._client
            self._client = None
            asyncio.run(
                close_async_resource(client_to_close, "qdrant_async_client"),
            )

        # Close sync client
        if self._sync_client is not None:
            try:
                self._sync_client.close()
            except Exception as e:
                self._logger.exception(  # type: ignore[attr-defined]
                    "Qdrant sync client close failed",
                    extra={"error": str(e)},
                )
            finally:
                self._sync_client = None
