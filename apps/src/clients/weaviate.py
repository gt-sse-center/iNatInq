"""Weaviate client wrapper class for vector database operations.

This module provides a Weaviate client wrapper class that encapsulates configuration
and provides methods for collection management and vector operations. This implements
the VectorDBProvider ABC for provider-agnostic usage.

Note: Type stubs for weaviate-client v4 are incomplete, so some type checker
warnings may appear but the code works correctly at runtime.

## Usage

```python
from clients.weaviate import WeaviateClientWrapper

client = WeaviateClientWrapper(url="http://weaviate.ml-system:8080")
client.ensure_collection_async(collection="documents", vector_size=768)
results = client.search_async(collection="documents", query_vector=[...], limit=10)
```

## Design

The client wrapper:
- Encapsulates Weaviate connection configuration
- Provides a clean interface for collection and search operations
- Handles errors consistently via `UpstreamError`
- Uses attrs for concise, correct class definition
- Implements VectorDBProvider ABC for provider-agnostic usage
"""

from typing import Any
from urllib.parse import urlparse

import attrs
import pybreaker
from weaviate import WeaviateAsyncClient
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.connect import ConnectionParams

from config import VectorDBConfig
from core.exceptions import UpstreamError
from core.models import SearchResultItem, SearchResults
from foundation.circuit_breaker import handle_circuit_breaker_error

from .base import VectorDBClientBase
from .interfaces.vector_db import VectorDBProvider


@attrs.define(frozen=True, slots=True)
class WeaviateDataObject:
    """Weaviate data object representation.

    This class represents a Weaviate data object with properties and vector.
    In Weaviate, objects have properties (metadata) and an optional vector.

    Attributes:
        uuid: Unique identifier for the object (UUID string).
        properties: Metadata dictionary (key-value pairs) associated with the object.
        vector: Embedding vector (list of floats).

    Example:
        ```python
        from clients.weaviate import WeaviateDataObject

        obj = WeaviateDataObject(
            uuid="123e4567-e89b-12d3-a456-426614174000",
            properties={"text": "hello world", "source": "file.txt"},
            vector=[0.1, 0.2, 0.3, ...]
        )
        ```
    """

    uuid: str
    properties: dict[str, Any]
    vector: list[float]


@attrs.define(frozen=False, slots=True)
class WeaviateClientWrapper(VectorDBClientBase, VectorDBProvider):
    """Wrapper for Weaviate async client with convenience methods.

    Attributes:
        url: Weaviate service URL (e.g., `http://weaviate.ml-system:8080`).
        api_key: Optional API key for authenticated Weaviate instances.
        grpc_host: Optional gRPC host for Weaviate Cloud (e.g.,
            `grpc-xxx.region.weaviate.cloud`). If not provided, defaults
            to the HTTP host with port 50051.

    Note:
        This class uses WeaviateAsyncClient internally but provides a sync
        interface to match the VectorDBProvider ABC. Async operations are
        wrapped with asyncio.run().
    """

    url: str
    api_key: str | None = None
    grpc_host: str | None = None
    _client: WeaviateAsyncClient = attrs.field(init=False, default=None)
    _breaker: pybreaker.CircuitBreaker = attrs.field(init=False)

    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for Weaviate.

        Weaviate is on critical path (blocks user search requests).

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return ("weaviate", 3, 60)

    def __attrs_post_init__(self) -> None:
        """Initialize the Weaviate async client and circuit breaker after attrs construction."""
        auth_config = None
        if self.api_key:
            auth_config = AuthApiKey(api_key=self.api_key)

        # Parse URL to extract host and port
        parsed = urlparse(self.url)
        http_host = parsed.hostname or "localhost"
        http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        http_secure = parsed.scheme == "https"

        # Weaviate v4 requires ConnectionParams with explicit HTTP and gRPC
        # parameters. For Weaviate Cloud, gRPC uses a separate host on port 443.
        # For local Docker, gRPC uses the same host on port 50051.
        if self.grpc_host:
            # Weaviate Cloud: separate gRPC host, secure on port 443
            grpc_host = self.grpc_host
            grpc_port = 443
            grpc_secure = True
        else:
            # Local Docker: same host, port 50051, no TLS
            grpc_host = http_host
            grpc_port = 50051
            grpc_secure = False

        connection_params = ConnectionParams.from_params(
            http_host=http_host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure,
        )

        _client_instance = WeaviateAsyncClient(
            connection_params=connection_params,
            auth_client_secret=auth_config,
        )

        self._client = _client_instance

        # Initialize circuit breaker from base class
        self._init_circuit_breaker()

    @property
    def client(self) -> WeaviateAsyncClient:
        """Get the underlying WeaviateAsyncClient instance."""
        return self._client

    @classmethod
    def from_config(cls, config: VectorDBConfig) -> "WeaviateClientWrapper":
        """Create WeaviateClientWrapper from VectorDBConfig.

        Args:
            config: Vector database configuration with Weaviate settings.

        Returns:
            Configured WeaviateClientWrapper instance.

        Raises:
            ValueError: If Weaviate config is missing or invalid.
        """
        cls._validate_config(config, "weaviate", ["weaviate_url"])
        # Type narrowing: _validate_config ensures weaviate_url is not None
        assert config.weaviate_url is not None
        return cls(
            url=config.weaviate_url,
            api_key=config.weaviate_api_key,
            grpc_host=config.weaviate_grpc_host,
        )

    async def ensure_collection_async(self, *, collection: str, vector_size: int) -> None:
        """Create a Weaviate collection (class) if it does not already exist.

        This is a dev convenience function that checks for collection existence and
        creates it with sensible defaults if missing. In production, you might want
        to manage collections via infrastructure-as-code or separate tooling.

        Args:
            collection: Collection name (class name in Weaviate) to ensure exists.
            vector_size: Dimension of vectors that will be stored in this collection.
                Must match the embedding dimension from your model (e.g., 768 for
                `nomic-embed-text`).

        Note:
            The collection is created with:
            - **Distance metric**: Cosine similarity (standard for embeddings)
            - **Vector size**: As specified by `vector_size`
            - **Vectorizer**: None (we provide vectors directly)

            If the collection already exists, this function does nothing (no-op).
        """
        try:
            # Weaviate v4 client is the async context manager
            async with self._client:
                # Check if collection exists
                exists = await self._client.collections.exists(collection)
                if exists:
                    return

                # Create collection with vector configuration
                await self._client.collections.create(
                    name=collection,
                    vectorizer_config=None,  # We provide vectors directly
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="s3_key", data_type=DataType.TEXT),
                        Property(name="s3_bucket", data_type=DataType.TEXT),
                        Property(name="s3_uri", data_type=DataType.TEXT),
                    ],
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE,
                    ),
                )
        except Exception as e:
            # If collection already exists, that's fine
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                return
            msg = f"Weaviate collection creation failed: {e}"
            raise UpstreamError(msg) from e

    async def search_async(
        self, *, collection: str, query_vector: list[float], limit: int = 10
    ) -> SearchResults:
        """Search for similar vectors in a Weaviate collection.

        Args:
            collection: Collection name (class name) to search.
            query_vector: Query embedding vector (must match collection dimension).
            limit: Maximum number of results to return.

        Returns:
            A `SearchResults` instance containing:
            - `items`: List of search result items, ordered by similarity (highest first)
            - `total`: Total number of results found

        Raises:
            UpstreamError: If collection doesn't exist or search fails. Also raised when
                circuit breaker is open.

        Example:
            ```python
            results = client.search_async(
                collection="documents",
                query_vector=[0.1, 0.2, ...],  # 768-dimensional vector
                limit=10
            )
            # Access: results.items[0].point_id, results.items[0].score, etc.
            ```
        """
        # Check circuit breaker state - if open, fail fast
        if self._breaker.current_state == pybreaker.STATE_OPEN:
            handle_circuit_breaker_error("weaviate")

        try:
            # Weaviate v4 client is the async context manager
            async with self._client:
                collection_obj = self._client.collections.get(collection)

                # Perform vector search (await the coroutine)
                response = await collection_obj.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=["distance", "certainty"],
                )

                items = []
                for obj in response.objects:
                    # Weaviate returns distance (lower is better) or certainty (higher is better)
                    # We'll use certainty if available, otherwise convert distance to similarity
                    distance = obj.metadata.distance if obj.metadata else None
                    certainty = obj.metadata.certainty if obj.metadata else None

                    # Convert to similarity score (0.0 to 1.0, higher is more similar)
                    if certainty is not None:
                        score = float(certainty)
                    elif distance is not None:
                        # Convert cosine distance to similarity (1 - distance for cosine)
                        score = max(0.0, min(1.0, 1.0 - float(distance)))
                    else:
                        score = 0.0

                    # Extract payload (properties)
                    payload: dict[str, Any] = {}
                    if obj.properties:
                        payload = dict(obj.properties)

                    items.append(
                        SearchResultItem(
                            point_id=str(obj.uuid),
                            score=score,
                            payload=payload,
                        )
                    )

                return SearchResults(items=items, total=len(items))
        except Exception as e:
            msg = f"Weaviate search failed: {e}"
            raise UpstreamError(msg) from e

    async def _do_batch_upsert(self, *, collection: str, points: list[WeaviateDataObject]) -> None:
        """Weaviate-specific batch upsert implementation.

        Args:
            collection: Collection name to upsert into.
            points: List of WeaviateDataObject instances to upsert.

        Raises:
            Exception: Any Weaviate-specific exceptions (wrapped by base class).
        """
        # Weaviate v4 client is the async context manager
        async with self._client:
            # Get collection for batch operations
            collection_obj = self._client.collections.get(collection)

            objects_to_insert = [
                DataObject(
                    properties=obj.properties,
                    vector=obj.vector if obj.vector else None,
                    uuid=obj.uuid if obj.uuid else None,
                )
                for obj in points
            ]

            # Batch insert using collection's insert_many
            await collection_obj.data.insert_many(objects_to_insert)

    async def batch_upsert_async(
        self,
        *,
        collection: str,
        points: list[WeaviateDataObject],  # type: ignore[override]
        vector_size: int,
    ) -> None:
        """Batch upsert data objects into a Weaviate collection.

        This method uses the base class template which ensures the collection exists
        before upserting and performs batch upserts for better performance.

        Args:
            collection: Collection name (class name) to upsert into.
            points: List of WeaviateDataObject instances to upsert. Must not be empty.
            vector_size: Vector dimension (e.g., 768 for nomic-embed-text).
                Used to ensure collection exists with correct dimensions.

        Raises:
            UpstreamError: If Weaviate operations fail. Also raised when circuit
                breaker is open.

        Example:
            ```python
            from clients.weaviate import WeaviateDataObject

            objects = [
                WeaviateDataObject(
                    uuid="1",
                    properties={"text": "hello"},
                    vector=[0.1, 0.2, ...]
                ),
                WeaviateDataObject(
                    uuid="2",
                    properties={"text": "world"},
                    vector=[0.3, 0.4, ...]
                ),
            ]
            client.batch_upsert_async(
                collection="documents",
                points=objects,
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
        """Close the Weaviate client and release resources.

        This method cleans up the underlying async Weaviate client to properly
        release connections and prevent resource leaks.

        Note:
            WeaviateAsyncClient uses context managers for connection lifecycle
            management. Since we use `async with self._client:` in our methods,
            connections are automatically cleaned up. This method primarily
            clears the client reference to allow garbage collection.
        """
        if self._client is not None:
            # WeaviateAsyncClient is designed to be used as a context manager
            # and doesn't maintain persistent connections. Connections are
            # created and cleaned up within each async context.
            # We clear the reference to allow garbage collection.
            self._client = None
