"""Vector database provider interface and factory.

This module defines the `VectorDBProvider` ABC and factory functions for vector
database providers. Configuration classes are in `pipeline.config`.
Concrete implementations live in the parent `clients` package (e.g., `QdrantClientWrapper`).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias

from config import VectorDBConfig
from core.models import SearchResults, VectorPoint

if TYPE_CHECKING:
    from clients.weaviate import WeaviateDataObject
else:
    # Runtime stub to avoid importing WeaviateDataObject at runtime
    # (avoids circular dependency - registration happens in clients.registries)
    WeaviateDataObject = object  # type: ignore[assignment, misc]

# Type alias for batch_upsert points parameter.  Qdrant uses VectorPoint (our
# wrapper), Weaviate uses WeaviateDataObject
VectorDBPoint: TypeAlias = VectorPoint | WeaviateDataObject


class VectorDBProvider(ABC):
    """Abstract base class for vector database providers.

    This class defines the interface that all vector database providers must implement.
    Each provider (QdrantClientWrapper, WeaviateClient, etc.) inherits from this class
    and implements the required methods.

    Example:
        ```python
        class MyVectorDB(VectorDBProvider):
            async def ensure_collection_async(self, collection: str, vector_size: int) -> None:
                # Implementation
                ...

            async def search_async(
                self, collection: str, query_vector: list[float], limit: int
            ) -> SearchResults:
                # Implementation
                ...

            async def batch_upsert_async(
                self, collection: str, points: list[VECTOR_DB_POINT], vector_size: int
            ) -> None:
                # Implementation
                ...
        ```
    """

    @abstractmethod
    async def ensure_collection_async(self, collection: str, vector_size: int) -> None:
        """Create a collection if it does not already exist.

        This is a dev convenience function that checks for collection existence
        and creates it with sensible defaults if missing. In production, you might
        want to manage collections via infrastructure-as-code or separate tooling.

        Args:
            collection: Collection name to ensure exists.
            vector_size: Dimension of vectors that will be stored in this collection.
                Must match the embedding dimension from your model (e.g., 768 for
                `nomic-embed-text`).

        Note:
            If the collection already exists, this function should do nothing
            (no-op).
        """

    @abstractmethod
    async def search_async(
        self, collection: str, query_vector: list[float], limit: int = 10
    ) -> SearchResults:
        """Search for similar vectors in a collection.

        Args:
            collection: Collection name to search.
            query_vector: Query embedding vector (must match collection dimension).
            limit: Maximum number of results to return.

        Returns:
            A `SearchResults` instance containing:
            - `items`: List of search result items, ordered by similarity (highest first)
            - `total`: Total number of results found

        Raises:
            UpstreamError: If collection doesn't exist or search fails.
        """

    @abstractmethod
    async def batch_upsert_async(
        self,
        collection: str,
        points: list[VectorDBPoint],
        vector_size: int,
    ) -> None:
        """Batch upsert points/objects into a collection.

        This method ensures the collection exists before upserting and performs
        batch upserts for better performance. It's designed for high-throughput
        scenarios like Ray job processing.

        Args:
            collection: Collection name to upsert into.
            points: List of points/objects to upsert. Must not be empty.
                - Qdrant: List of VectorPoint instances (from core.models)
                - Weaviate: List of WeaviateDataObject instances (from clients.weaviate)
            vector_size: Vector dimension (e.g., 768 for nomic-embed-text).
                Used to ensure collection exists with correct dimensions.

        Raises:
            UpstreamError: If vector database operations fail.

        Note:
            The collection is automatically created if it doesn't exist.
            Empty point lists should be ignored (no-op).
        """

    def close(self) -> None:  # noqa: B027
        """Close client connections and cleanup resources.

        This method should be called when the provider is no longer needed
        to properly release connections and prevent resource leaks.

        Note:
            This is not an abstract method because some providers may not
            need cleanup. Subclasses should override if they maintain resources.
        """
        # Default no-op implementation

    @classmethod
    @abstractmethod
    def from_config(cls, config: VectorDBConfig) -> "VectorDBProvider":
        """Create provider instance from VectorDBConfig.

        Each provider class must implement this class method to construct itself
        from the configuration. This allows the factory to instantiate providers
        without knowing provider-specific construction details.

        Args:
            config: Vector database configuration.

        Returns:
            Configured VectorDBProvider instance.

        Raises:
            ValueError: If config is invalid or missing required fields.
        """


# Provider registry: maps provider_type to provider class
# Each provider class must inherit from VectorDBProvider and implement the interface
_PROVIDER_REGISTRY: dict[str, type[VectorDBProvider]] = {}


def register_provider(provider_type: str, provider_class: type[VectorDBProvider]) -> None:
    """Register a vector database provider class.

    This function allows providers to register themselves in the factory registry.
    This makes the factory extensible without needing to modify it for each new provider.

    Args:
        provider_type: Provider type identifier (e.g., "qdrant", "weaviate").
        provider_class: Provider class that inherits from VectorDBProvider.

    Example:
        ```python
        from clients.interfaces.vector_db import VectorDBProvider, register_provider

        class MyProvider(VectorDBProvider):
            # ... implement interface ...

        register_provider("myprovider", MyProvider)
        ```
    """
    _PROVIDER_REGISTRY[provider_type] = provider_class


def create_vector_db_provider(config: VectorDBConfig) -> "VectorDBProvider":
    """Create a vector database provider based on configuration.

    This factory function instantiates the appropriate vector database client based
    on the provider type in the configuration. Providers are registered via
    `register_provider()` and must inherit from `VectorDBProvider`.

    Args:
        config: Vector database configuration.

    Returns:
        VectorDBProvider instance (QdrantClientWrapper, WeaviateClient, etc.).

    Raises:
        ValueError: If provider type is not registered or required config is missing.

    Example:
        ```python
        from clients.interfaces.vector_db import VectorDBConfig, create_vector_db_provider

        config = VectorDBConfig.from_env()
        provider = create_vector_db_provider(config)
        results = await provider.search_async(collection="documents", query_vector=[...], limit=10)
        ```
    """
    provider_class = _PROVIDER_REGISTRY.get(config.provider_type)
    if provider_class is None:
        msg = (
            f"Provider type '{config.provider_type}' is not registered. "
            f"Available providers: {list(_PROVIDER_REGISTRY.keys())}"
        )
        raise ValueError(msg)

    # Instantiate provider - each provider class knows how to construct itself from config
    # This delegates the construction logic to the provider class
    # Since from_config is abstract, all registered providers must implement it
    return provider_class.from_config(config)
