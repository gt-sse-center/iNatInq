"""Service layer for semantic search operations.

This module provides service classes for semantic search orchestration:
- ImageSearchService: Text-to-image search using embeddings

The services:
1. Generate embedding for query text
2. Search vector database for similar vectors
3. Format and return results
"""

import attrs

from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from core.exceptions import BadRequestError
from core.models import SearchResults


@attrs.define(frozen=True, slots=True)
class ImageSearchService:
    """Service for performing text-to-image search using embeddings.

    This service orchestrates text-to-image search by:
    1. Generating text embedding for the query via EmbeddingProvider
    2. Searching image collections in vector database
    3. Returning formatted results with image metadata and similarity scores

    Attributes:
        embedding_provider: EmbeddingProvider instance for generating text embeddings.
        vector_db_provider: Vector database provider instance.

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

        # 1. Generate text embedding for query (async)
        query_embedding = await self.embedding_provider.embed_text(query.strip())

        # 2. Search image collection (async)
        return await self.vector_db_provider.search_async(
            collection=collection,
            query_vector=query_embedding,
            limit=limit,
        )
