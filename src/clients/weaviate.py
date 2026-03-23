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

# For image embeddings (CLIP)
await client.ensure_image_collection_async(collection="documents")
# Creates class "DocumentsImages" with s3_key, s3_uri, format, width, height, thumbnail_key
```

## Design

The client wrapper:
- Encapsulates Weaviate connection configuration
- Provides a clean interface for collection and search operations
- Handles errors consistently via `UpstreamError`
- Uses attrs for concise, correct class definition
- Implements VectorDBProvider ABC for provider-agnostic usage
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import aiobreaker
import attrs
from typing_extensions import override
from weaviate import WeaviateAsyncClient
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

if TYPE_CHECKING:
    from weaviate.collections.classes.config_base import _QuantizerConfigCreate  # pyright: ignore[reportPrivateUsage]
from weaviate.classes.data import DataObject
from weaviate.classes.init import Timeout
from weaviate.config import AdditionalConfig
from weaviate.connect import ConnectionParams
from weaviate.exceptions import (
    UnexpectedStatusCodeError,
    WeaviateConnectionError,
    WeaviateGRPCUnavailableError,
    WeaviateTimeoutError,
)

from config import VectorDBConfig
from core.models import CollectionInfo, SearchResultItem, SearchResults
from foundation.circuit_breaker import (
    create_async_circuit_breaker,
    with_circuit_breaker_async,
)
from foundation.exceptions import UpstreamError
from foundation.metrics.decorators import with_client_metrics_async
from foundation.retry import HTTPErrorClassifier, async_retry_call, create_retry_logger

from .base import VectorDBClientBase
from .interfaces.vector_db import VectorDBProvider

logger = logging.getLogger("clients.weaviate")
_retry_logger = logging.getLogger("clients.weaviate.retry")


# =============================================================================
# Weaviate Error Classifier
# =============================================================================


class WeaviateErrorClassifier(HTTPErrorClassifier):
    """Weaviate-specific error classification for retry logic."""

    @override
    def is_retriable(self, exc: BaseException) -> bool:
        """Classify whether the exception is retriable."""
        # Connection/transport errors are always retriable
        if isinstance(exc, (WeaviateConnectionError, WeaviateGRPCUnavailableError)):
            return True
        if isinstance(exc, WeaviateTimeoutError):
            return True
        # HTTP status errors: 5xx retriable, 4xx not
        if isinstance(exc, UnexpectedStatusCodeError):
            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                return self.is_retriable_http_status(status_code)
            return True  # No status code implies transport error
        if isinstance(exc, UpstreamError):
            return False
        return False

    @override
    def get_error_details(self, exc: BaseException) -> dict[str, Any]:
        """Extract structured error details for logging."""
        if isinstance(exc, UnexpectedStatusCodeError):
            return {"http_status": getattr(exc, "status_code", None)}
        return {}


_weaviate_classifier = WeaviateErrorClassifier()
_weaviate_log_retry = create_retry_logger(
    _retry_logger,
    _weaviate_classifier.get_error_details,
    "Weaviate async operation failed, retrying",
)

# Type alias for supported distance metrics
DistanceMetric = Literal["cosine", "euclidean", "dot"]

# Mapping from string distance metric to Weaviate VectorDistances enum
_DISTANCE_METRIC_MAP: dict[str, VectorDistances] = {
    "cosine": VectorDistances.COSINE,
    "euclidean": VectorDistances.L2_SQUARED,
    "dot": VectorDistances.DOT,
}


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
        timeout_s: Request timeout in seconds. Default: 300.
        circuit_breaker_threshold: Number of failures before circuit opens.
            Default: 3.
        circuit_breaker_timeout: Seconds before circuit breaker recovery.
            Default: 60.
        skip_init_checks: Whether to skip startup gRPC health checks.
            Default: True (avoids Docker-to-cloud connectivity issues).

    Note:
        This class uses WeaviateAsyncClient internally but provides a sync
        interface to match the VectorDBProvider ABC. Async operations are
        wrapped with asyncio.run().
    """

    url: str
    api_key: str | None = None
    grpc_host: str | None = None
    grpc_port: int | None = None  # Custom gRPC port for testcontainers
    timeout_s: int = attrs.field(default=300)
    circuit_breaker_threshold: int = attrs.field(default=3)
    circuit_breaker_timeout: int = attrs.field(default=60)
    max_retries: int = attrs.field(default=3)
    retry_min_wait: float = attrs.field(default=1.0)
    retry_max_wait: float = attrs.field(default=10.0)
    skip_init_checks: bool = True  # Default True to avoid Docker-to-cloud gRPC issues
    _client: WeaviateAsyncClient = attrs.field(init=False, default=None)
    _async_breaker: aiobreaker.CircuitBreaker = attrs.field(init=False)

    @override
    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for Weaviate.

        Weaviate is on critical path (blocks user search requests).

        Returns:
            Tuple of (name, failure_threshold, recovery_timeout).
        """
        return (
            "weaviate",
            self.circuit_breaker_threshold,
            self.circuit_breaker_timeout,
        )

    def __attrs_post_init__(self) -> None:
        """Initialize the Weaviate async client and circuit breaker after attrs construction."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — ensure one exists for WeaviateAsyncClient
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

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
            grpc_port = self.grpc_port or 443
            grpc_secure = True
        else:
            # Local Docker: same host, port 50051 (or custom port for testcontainers)
            grpc_host = http_host
            grpc_port = self.grpc_port or 50051
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
            # Skip gRPC health checks to avoid connectivity issues from Docker
            # to external Weaviate Cloud. The health check can fail due to
            # firewall rules or network latency from containerized environments.
            skip_init_checks=self.skip_init_checks,
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    init=self.timeout_s,
                    query=self.timeout_s,
                    insert=self.timeout_s,
                ),
            ),
        )

        self._client = _client_instance

        # Initialize sync circuit breaker from base class
        self._init_circuit_breaker()

        # Initialize async circuit breaker using aiobreaker
        name, fail_max, timeout = self._circuit_breaker_config()
        object.__setattr__(
            self,
            "_async_breaker",
            create_async_circuit_breaker(name, fail_max, timeout),
        )

    @property
    def client(self) -> WeaviateAsyncClient:
        """Get the underlying WeaviateAsyncClient instance."""
        return self._client

    @classmethod
    @override
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
            grpc_port=config.weaviate_grpc_port,
            timeout_s=config.weaviate_timeout,
            circuit_breaker_threshold=config.weaviate_circuit_breaker_threshold,
            circuit_breaker_timeout=config.weaviate_circuit_breaker_timeout,
        )

    @with_client_metrics_async("weaviate", "ensure_collection_async")
    @with_circuit_breaker_async("weaviate")
    async def ensure_collection_async(
        self,
        *,
        collection: str,
        vector_size: int,
        quantizer: "_QuantizerConfigCreate | None" = None,
    ) -> None:
        """Create a Weaviate collection (class) if it does not already exist.

        This is a dev convenience function that checks for collection existence and
        creates it with sensible defaults if missing. In production, you might want
        to manage collections via infrastructure-as-code or separate tooling.

        Args:
            collection: Collection name (class name in Weaviate) to ensure exists.
            vector_size: Dimension of vectors that will be stored in this collection.
                Must match the embedding dimension from your model (e.g., 768 for
                `nomic-embed-text`).
            quantizer: Optional Weaviate quantizer configuration. Use
                ``Configure.VectorIndex.Quantizer.sq()`` for scalar or
                ``Configure.VectorIndex.Quantizer.bq()`` for binary.

        Note:
            The collection is created with:
            - **Distance metric**: Cosine similarity (standard for embeddings)
            - **Vector size**: As specified by `vector_size`
            - **Vectorizer**: None (we provide vectors directly)

            If the collection already exists, this function does nothing (no-op).
        """

        async def _do_ensure() -> None:
            try:
                async with self._client:
                    exists = await self._client.collections.exists(collection)
                    if exists:
                        return

                    await self._client.collections.create(
                        name=collection,
                        vectorizer_config=None,
                        properties=[
                            Property(name="text", data_type=DataType.TEXT),
                            Property(name="s3_key", data_type=DataType.TEXT),
                            Property(name="s3_bucket", data_type=DataType.TEXT),
                            Property(name="s3_uri", data_type=DataType.TEXT),
                        ],
                        vector_config=Configure.Vectors.self_provided(
                            vector_index_config=Configure.VectorIndex.hnsw(
                                distance_metric=VectorDistances.COSINE,
                                quantizer=quantizer,
                            ),
                        ),
                    )
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    return
                raise  # Let async_retry_call classify the native exception

        await async_retry_call(
            _do_ensure,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_weaviate_classifier.is_retriable,
            before_sleep=_weaviate_log_retry,
            operation="Weaviate ensure_collection",
            client="weaviate",
        )

    @staticmethod
    def _collection_to_image_class_name(collection: str) -> str:
        """Derive Weaviate class name for image collection from base collection name.

        Pattern: {Collection}Images (PascalCase base + 'Images'). E.g. 'documents' -> 'DocumentsImages'.

        Args:
            collection: Base collection name (e.g., 'documents', 'my-photos').

        Returns:
            Weaviate class name (e.g., 'DocumentsImages', 'MyPhotosImages').
        """
        parts = collection.replace("-", "_").split("_")
        pascal = "".join(part.capitalize() for part in parts)
        return f"{pascal}Images"

    def _to_class_name(self, collection: str) -> str:
        """Convert any collection name to Weaviate class name (PascalCase).

        Weaviate requires class names to be PascalCase. This method converts
        snake_case or kebab-case names to PascalCase. If the name is already
        a single PascalCase token (e.g. from _collection_to_image_class_name),
        it is returned as-is so that casing is preserved.

        Args:
            collection: Collection name (e.g., 'documents_images', 'my-photos',
                or already 'DocumentsImages').

        Returns:
            Weaviate class name in PascalCase (e.g., 'DocumentsImages', 'MyPhotos').
        """
        parts = collection.replace("-", "_").split("_")
        return "".join(part[:1].upper() + part[1:] for part in parts if part)

    @with_client_metrics_async("weaviate", "ensure_image_collection_async")
    @with_circuit_breaker_async("weaviate")
    async def ensure_image_collection_async(
        self,
        *,
        collection: str,
        vector_size: int = 512,
        distance_metric: DistanceMetric = "cosine",
        quantizer: "_QuantizerConfigCreate | None" = None,
    ) -> None:
        """Create a Weaviate class for image embeddings if it does not already exist.

        Mirrors the Qdrant image collection schema: class name pattern {Collection}Images
        (e.g., DocumentsImages), with image-specific properties and CLIP vector dimensions.

        Image collections use:
        - **Class name**: `{Collection}Images` (PascalCase, e.g., DocumentsImages)
        - **Properties**: s3_key, s3_uri, format, width, height, thumbnail_key
        - **Vector config**: Configurable distance metric; dimension from vector_size.

        Args:
            collection: Base collection name. The Weaviate class will be named
                {Collection}Images (e.g., collection="documents" -> "DocumentsImages").
            vector_size: Dimension of vectors. Default 512 (CLIP models like ViT-B/32).
            distance_metric: Distance metric for vector similarity. One of "cosine",
                "euclidean", or "dot". Default "cosine" (standard for embeddings).
            quantizer: Optional Weaviate quantizer configuration. Use
                ``Configure.VectorIndex.Quantizer.sq()`` for scalar or
                ``Configure.VectorIndex.Quantizer.bq()`` for binary.

        Note:
            If the class already exists, this function does nothing (no-op).

        Example:
            >>> await client.ensure_image_collection_async(collection="documents")
            >>> # Creates class "DocumentsImages" with 512-dimensional vectors
        """
        image_class = self._collection_to_image_class_name(collection)

        async def _do_ensure_image() -> None:
            try:
                async with self._client:
                    exists = await self._client.collections.exists(image_class)
                    if exists:
                        return

                    await self._client.collections.create(
                        name=image_class,
                        vectorizer_config=None,
                        properties=[
                            Property(name="s3_key", data_type=DataType.TEXT),
                            Property(name="s3_uri", data_type=DataType.TEXT),
                            Property(name="format", data_type=DataType.TEXT),
                            Property(name="width", data_type=DataType.INT),
                            Property(name="height", data_type=DataType.INT),
                            Property(name="thumbnail_key", data_type=DataType.TEXT),
                        ],
                        vector_config=Configure.Vectors.self_provided(
                            vector_index_config=Configure.VectorIndex.hnsw(
                                distance_metric=_DISTANCE_METRIC_MAP[distance_metric],
                                quantizer=quantizer,
                            ),
                        ),
                    )
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    return
                raise  # Let async_retry_call classify the native exception

        await async_retry_call(
            _do_ensure_image,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_weaviate_classifier.is_retriable,
            before_sleep=_weaviate_log_retry,
            operation="Weaviate ensure_image_collection",
            client="weaviate",
        )

    @with_client_metrics_async("weaviate", "get_collection_info_async")
    @with_circuit_breaker_async("weaviate")
    async def get_collection_info_async(self, *, collection: str) -> CollectionInfo:
        """Retrieve collection metadata from Weaviate.

        Weaviate may not expose vector dimensions depending on vectorizer
        configuration. When the dimension is unavailable, ``vector_size=0``
        is returned so callers can skip validation.

        Args:
            collection: Collection name to query.

        Returns:
            CollectionInfo with the collection name and vector dimension.
        """

        async def _do_get_info() -> CollectionInfo:
            class_name = self._to_class_name(collection)
            async with self._client:
                col = self._client.collections.get(class_name)
                cfg = await col.config.get()
                # Attempt to extract vector dimension from the config
                vector_size = 0
                if hasattr(cfg, "vectorizer_config") and cfg.vectorizer_config:
                    vec_cfg = cfg.vectorizer_config
                    if hasattr(vec_cfg, "vector_dimension"):
                        vector_size = int(vec_cfg.vector_dimension or 0)  # type: ignore[attr-defined]
                return CollectionInfo(name=collection, vector_size=vector_size)

        return await async_retry_call(
            _do_get_info,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_weaviate_classifier.is_retriable,
            before_sleep=_weaviate_log_retry,
            client="weaviate",
            operation="get_collection_info_async",
        )

    @with_client_metrics_async("weaviate", "search_async")
    @with_circuit_breaker_async("weaviate")
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
                circuit breaker is open or retries are exhausted.

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
            class_name = self._to_class_name(collection)
            async with self._client:
                collection_obj = self._client.collections.get(class_name)

                response = await collection_obj.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    return_metadata=["distance", "certainty"],
                    target_vector="default",
                )

                items = []
                for obj in response.objects:
                    distance = obj.metadata.distance if obj.metadata else None
                    certainty = obj.metadata.certainty if obj.metadata else None

                    if certainty is not None:
                        score = float(certainty)
                    elif distance is not None:
                        score = max(0.0, min(1.0, 1.0 - float(distance)))
                    else:
                        score = 0.0

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

        try:
            return await async_retry_call(
                _do_search,
                max_retries=self.max_retries,
                min_wait=self.retry_min_wait,
                max_wait=self.retry_max_wait,
                is_retriable=_weaviate_classifier.is_retriable,
                before_sleep=_weaviate_log_retry,
                client="weaviate",
                operation="Weaviate search",
            )
        except UpstreamError:
            raise
        except Exception as e:
            msg = f"Weaviate search failed: {e}"
            raise UpstreamError(msg) from e

    @with_client_metrics_async("weaviate", "_do_batch_upsert")
    @with_circuit_breaker_async("weaviate")
    async def _do_batch_upsert(self, *, collection: str, points: list[WeaviateDataObject]) -> None:
        """Weaviate-specific batch upsert implementation with retry.

        Args:
            collection: Collection name to upsert into.
            points: List of WeaviateDataObject instances to upsert.

        Raises:
            Exception: Any Weaviate-specific exceptions (wrapped by base class).
        """

        async def _do_upsert() -> None:
            async with self._client:
                collection_obj = self._client.collections.get(collection)

                objects_to_insert = [
                    DataObject(
                        properties=obj.properties,
                        vector={"default": obj.vector} if obj.vector else None,
                        uuid=obj.uuid or None,
                    )
                    for obj in points
                ]

                result = await collection_obj.data.insert_many(objects_to_insert)
                if hasattr(result, "has_errors") and result.has_errors:
                    error_details = {str(k): str(v) for k, v in list(result.errors.items())[:5]}
                    msg = (
                        f"Weaviate insert_many partial failure: "
                        f"{len(result.errors)}/{len(objects_to_insert)} failed"
                    )
                    logger.warning(msg, extra={"errors": error_details})
                    raise UpstreamError(msg)

        await async_retry_call(
            _do_upsert,
            max_retries=self.max_retries,
            min_wait=self.retry_min_wait,
            max_wait=self.retry_max_wait,
            is_retriable=_weaviate_classifier.is_retriable,
            before_sleep=_weaviate_log_retry,
            client="weaviate",
            operation="batch_upsert_async",
        )

    @override
    async def batch_upsert_async(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        collection: str,
        points: list[WeaviateDataObject],
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

    @override
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
