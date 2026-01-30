"""Service layer for semantic search operations.

This module provides service classes for semantic search orchestration:
1. SearchService: Text-to-text document search using text embeddings
2. ImageSearchService: Text-to-image search using CLIP embeddings

Both services:
1. Generate embedding for query text
2. Search vector database for similar vectors
3. Format and return results
"""

import asyncio

import attrs

from clients.clip import CLIPClient
from clients.interfaces.embedding import EmbeddingProvider
from clients.interfaces.vector_db import VectorDBProvider
from core.exceptions import BadRequestError
from core.models import SearchResults


@attrs.define(frozen=True, slots=True)
class SearchService:
    """Service for performing semantic search over documents.

    This service orchestrates semantic search by:
    1. Generating embeddings for query text via embedding provider
    2. Searching vector database for similar vectors using cosine similarity
    3. Returning formatted results with scores and metadata

    Attributes:
        embedding_provider: Embedding provider instance (agnostic to implementation).
        vector_db_provider: Vector database provider instance (agnostic to implementation).

    Example:
        ```python
        from core.services.search_service import SearchService
        from clients.interfaces.embedding import create_embedding_provider, EmbeddingConfig
        from clients.interfaces.vector_db import create_vector_db_provider, VectorDBConfig

        embedding_config = EmbeddingConfig.from_env()
        embedding_provider = create_embedding_provider(embedding_config)
        vector_db_config = VectorDBConfig.from_env()
        vector_db_provider = create_vector_db_provider(vector_db_config)

        service = SearchService(
            embedding_provider=embedding_provider,
            vector_db_provider=vector_db_provider,
        )
        results = service.search_documents(
            collection="documents",
            query="machine learning pipeline",
            limit=10
        )
        ```
    """

    embedding_provider: EmbeddingProvider
    vector_db_provider: VectorDBProvider

    def search_documents(
        self,
        *,
        collection: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """Perform semantic search over documents in vector database.

        This method:
        1. Generates an embedding for the query text via embedding provider
        2. Searches vector database for similar vectors using cosine similarity
        3. Returns formatted results with scores and metadata

        Args:
            collection: Collection name to search.
            query: Natural language query string.
            limit: Maximum number of results to return.

        Returns:
            A `SearchResults` instance containing:
            - `items`: List of search result items, ordered by similarity (highest first)
            - `total`: Total number of results found

        Raises:
            BadRequestError: If query is empty or limit is invalid.
            UpstreamError: If embedding provider or vector database operations fail.

        Example:
            ```python
            service = SearchService(embedding_provider=..., vector_db_provider=...)
            results = service.search_documents(
                collection="documents",
                query="machine learning pipeline",
                limit=10
            )
            ```
        """
        if not query or not query.strip():
            raise BadRequestError("Query string cannot be empty")

        if limit < 1 or limit > 100:
            raise BadRequestError("Limit must be between 1 and 100")

        # 1. Generate embedding for query
        query_embedding = self.embedding_provider.embed(query.strip())

        # 2. Search vector database (async, run in event loop)
        return asyncio.run(
            self.vector_db_provider.search_async(
                collection=collection,
                query_vector=query_embedding,
                limit=limit,
            )
        )

    async def search_documents_async(
        self,
        *,
        collection: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """Perform semantic search over documents in vector database (async).

        This is the async version of `search_documents()` that uses async I/O for
        non-blocking operations. The embedding request is async, and vector database
        operations are run in a thread pool to avoid blocking.

        Args:
            collection: Collection name to search.
            query: Natural language query string.
            limit: Maximum number of results to return.

        Returns:
            A `SearchResults` instance containing:
            - `items`: List of search result items, ordered by similarity (highest first)
            - `total`: Total number of results found

        Raises:
            BadRequestError: If query is empty or limit is invalid.
            UpstreamError: If embedding provider or vector database operations fail.

        Example:
            ```python
            service = SearchService(embedding_provider=..., vector_db_provider=...)
            results = await service.search_documents_async(
                collection="documents",
                query="machine learning pipeline",
                limit=10
            )
            ```
        """
        if not query or not query.strip():
            raise BadRequestError("Query string cannot be empty")

        if limit < 1 or limit > 100:
            raise BadRequestError("Limit must be between 1 and 100")

        # 1. Generate embedding for query (async)
        query_embedding = await self.embedding_provider.embed_async(query.strip())

        # 2. Search vector database (async)
        return await self.vector_db_provider.search_async(
            collection=collection,
            query_vector=query_embedding,
            limit=limit,
        )


@attrs.define(frozen=True, slots=True)
class ImageSearchService:
    """Service for performing text-to-image search using CLIP embeddings.

    CLIP (Contrastive Language-Image Pre-training) embeddings allow searching
    images using natural language text queries. Both text and image embeddings
    live in the same vector space, enabling cross-modal search.

    This service orchestrates text-to-image search by:
    1. Generating CLIP text embedding for the query via CLIPClient
    2. Searching image collections in vector database
    3. Returning formatted results with image metadata and similarity scores

    Attributes:
        clip_client: CLIPClient instance for generating text embeddings.
        vector_db_provider: Vector database provider instance.

    Example:
        ```python
        from core.services.search_service import ImageSearchService
        from clients.clip import CLIPClient
        from clients.interfaces.vector_db import create_vector_db_provider

        clip_client = CLIPClient.from_config(ImageEmbeddingConfig.from_env())
        vector_db_provider = create_vector_db_provider(VectorDBConfig.from_env())

        service = ImageSearchService(
            clip_client=clip_client,
            vector_db_provider=vector_db_provider,
        )
        results = await service.search_images_async(
            collection="documents",
            query="sunset over the ocean",
            limit=10
        )
        ```
    """

    clip_client: CLIPClient
    vector_db_provider: VectorDBProvider

    def search_images(
        self,
        *,
        collection: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """Perform text-to-image search over image collection.

        This method:
        1. Generates a CLIP text embedding for the query
        2. Searches the image collection (named {collection}_images) for similar vectors
        3. Returns formatted results with image metadata and similarity scores

        Args:
            collection: Base collection name. The image collection will be
                searched as {collection}_images.
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

        # 1. Generate CLIP text embedding for query
        query_embedding = self.clip_client.embed_text(query.strip())

        # 2. Search image collection (async, run in event loop)
        image_collection = f"{collection}_images"
        return asyncio.run(
            self.vector_db_provider.search_async(
                collection=image_collection,
                query_vector=query_embedding,
                limit=limit,
            )
        )

    async def search_images_async(
        self,
        *,
        collection: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """Perform text-to-image search over image collection (async).

        This is the async version of `search_images()` that uses async I/O
        for non-blocking operations.

        Args:
            collection: Base collection name. The image collection will be
                searched as {collection}_images.
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

        # 1. Generate CLIP text embedding for query (async)
        query_embedding = await self.clip_client.embed_text_async(query.strip())

        # 2. Search image collection (async)
        image_collection = f"{collection}_images"
        return await self.vector_db_provider.search_async(
            collection=image_collection,
            query_vector=query_embedding,
            limit=limit,
        )
