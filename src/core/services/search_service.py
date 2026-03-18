"""Service layer for semantic search operations.

This module provides service classes for semantic search orchestration:
- ImageSearchService: Text-to-image search using embeddings

The services:
1. Generate embedding for query text
2. Search vector database for similar vectors
3. Format and return results
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

from core.exceptions import BadRequestError
from foundation.metrics.registry import (
    SEARCH_EMBEDDING_DURATION,
    SEARCH_RESULT_COUNT,
    SEARCH_VECTOR_QUERY_DURATION,
)

if TYPE_CHECKING:
    from clients.cache import CacheClient
    from clients.interfaces.embedding import EmbeddingProvider
    from clients.interfaces.vector_db import VectorDBProvider
    from core.models import SearchResults


@attrs.define(frozen=True, slots=True)
class ImageSearchService:
    """Service for performing text-to-image search using embeddings.

    This service orchestrates text-to-image search by:
    1. Generating text embedding for the query via EmbeddingProvider
    2. (Optional) Checking the semantic cache for a matching result
    3. Searching image collections in vector database on cache miss
    4. (Optional) Storing the result in the semantic cache

    Attributes:
        embedding_provider: EmbeddingProvider instance for generating text embeddings.
        vector_db_provider: Vector database provider instance.
        embedding_provider_name: String label for the embedding provider used in metrics
            (e.g. ``"clip"``, ``"ollama"``). Defaults to ``"clip"``.
        vector_db_provider_name: String label for the vector DB provider used in metrics
            (e.g. ``"qdrant"``, ``"weaviate"``). Defaults to ``"qdrant"``.
        cache: Optional ``CacheClient`` instance.  When ``None`` (default), the
            search pipeline operates without caching.

    Example:
        ```python
        from core.services.search_service import ImageSearchService
        from clients.interfaces.embedding import create_embedding_provider
        from clients.interfaces.vector_db import create_vector_db_provider

        embedding_provider = create_embedding_provider(EmbeddingConfig.from_env())
        vector_db_provider = create_vector_db_provider(VectorDBConfig.from_env())

        service = ImageSearchService(
            embedding_provider=embedding_provider,
            vector_db_provider=vector_db_provider,
        )
        results = await service.search_images_async(
            collection="documents",
            query="sunset over the ocean",
            limit=10
        )
        ```
    """

    embedding_provider: EmbeddingProvider
    vector_db_provider: VectorDBProvider
    embedding_provider_name: str = attrs.field(default="clip")
    vector_db_provider_name: str = attrs.field(default="qdrant")
    cache: CacheClient | None = attrs.field(default=None)

    async def search_images_async(
        self,
        *,
        collection: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """Perform text-to-image search over image collection.

        Args:
            collection: Base collection name.
            query: Natural language text query (e.g., "a fluffy cat").
            limit: Maximum number of results to return.

        Returns:
            A `SearchResults` instance containing:
            - `items`: List of image search results with s3_key, s3_uri, format, etc.
            - `total`: Total number of results found

        Raises:
            BadRequestError: If query is empty or limit is invalid.
            UpstreamError: If CLIP or vector database operations fail.
        """
        if not query or not query.strip():
            raise BadRequestError("Query string cannot be empty")

        if limit < 1 or limit > 100:
            raise BadRequestError("Limit must be between 1 and 100")

        with SEARCH_EMBEDDING_DURATION.labels(provider=self.embedding_provider_name).time():
            query_embedding = await self.embedding_provider.embed_text(query.strip())

        if self.cache is not None:
            cached = await self.cache.get(query_vector=query_embedding, collection=collection)
            if cached is not None:
                return cached

        with SEARCH_VECTOR_QUERY_DURATION.labels(
            provider=self.vector_db_provider_name, collection=collection
        ).time():
            results = await self.vector_db_provider.search_async(
                collection=collection,
                query_vector=query_embedding,
                limit=limit,
            )

        if self.cache is not None:
            await self.cache.put(query_vector=query_embedding, results=results, collection=collection)

        # Intentionally not observed on error: SEARCH_RESULT_COUNT tracks result
        # distributions for successful queries only. Error-path searches have no
        # result count to record; error rates are tracked separately via
        # CLIENT_ERRORS_TOTAL at the client layer.
        SEARCH_RESULT_COUNT.labels(collection=collection).observe(len(results.items))
        return results
