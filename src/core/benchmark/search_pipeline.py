"""Search pipeline composing embedding provider, vector search, and ID mapping.

Provides an end-to-end pipeline: text query → embedding provider → vector DB
search → document ID extraction.

Example:
    ```python
    from core.benchmark.search_pipeline import SearchPipeline

    pipeline = SearchPipeline(
        embedding_provider=provider,  # EmbeddingProvider (CLIP, Infinity, etc.)
        vector_provider=qdrant_provider,
        id_mapper=mapper,
    )
    result = await pipeline.search("red bird", collection="images", limit=10)
    print(result.doc_ids)  # ["1316621", "42", ...]
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import attrs
from typing_extensions import override

from core.benchmark.id_mapping import S3KeyIDMapper  # noqa: TC001 — used as attrs field type at runtime

if TYPE_CHECKING:
    from clients.interfaces.embedding import EmbeddingProvider
    from clients.interfaces.vector_db import VectorDBProvider
    from core.models import SearchResults


@attrs.define(frozen=True, slots=True)
class SearchPipelineResult:
    """Result of a search pipeline execution.

    Attributes:
        doc_ids: Mapped document IDs (e.g., iNat photo IDs) in ranked order.
        raw_results: The underlying SearchResults from the vector DB.
    """

    doc_ids: list[str]
    raw_results: SearchResults


@attrs.define(frozen=True, slots=True)
class SearchPipeline:
    """Pipeline: text query → embedding → vector search → ID mapping.

    Attributes:
        embedding_provider: EmbeddingProvider for generating text embeddings.
        vector_provider: VectorDBProvider for similarity search.
        id_mapper: S3KeyIDMapper for translating point IDs to doc IDs.
    """

    embedding_provider: EmbeddingProvider
    vector_provider: VectorDBProvider
    id_mapper: S3KeyIDMapper

    async def search(
        self,
        query_text: str,
        *,
        collection: str,
        limit: int = 10,
    ) -> SearchPipelineResult:
        """Execute a full search pipeline.

        1. Embed ``query_text`` via embedding provider.
        2. Search the vector DB collection.
        3. Map point IDs to document IDs via the ID mapper.

        Args:
            query_text: Text query to embed and search.
            collection: Vector database collection name.
            limit: Maximum number of results to return.

        Returns:
            SearchPipelineResult with mapped doc_ids and raw results.
        """
        query_vector = await self.embedding_provider.embed_text(query_text)

        results = await self.vector_provider.search_async(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
        )

        doc_ids = [self.id_mapper.point_id_to_doc_id(item.point_id, item.payload) for item in results.items]

        return SearchPipelineResult(doc_ids=doc_ids, raw_results=results)


@attrs.define(frozen=True, slots=True)
class QdrantSearchPipeline(SearchPipeline):
    """SearchPipeline variant that passes Qdrant-specific ``search_params``.

    Used for quantization benchmarking where different search configurations
    (e.g., rescore, oversampling) need to be tested against the same collection.

    Attributes:
        search_params: Qdrant ``SearchParams`` instance (e.g., for rescore
            or oversampling control). Typed as ``Any`` to avoid coupling
            the domain layer to ``qdrant_client.http.models``.
    """

    search_params: Any = None  # qmodels.SearchParams at runtime

    @override
    async def search(
        self,
        query_text: str,
        *,
        collection: str,
        limit: int = 10,
    ) -> SearchPipelineResult:
        """Execute search with Qdrant-specific search params.

        Embeds the query, then calls ``QdrantClientWrapper.search_async``
        with the configured ``search_params`` for quantization control.

        Args:
            query_text: Text query to embed and search.
            collection: Vector database collection name.
            limit: Maximum number of results to return.

        Returns:
            SearchPipelineResult with mapped doc_ids and raw results.

        Raises:
            TypeError: If ``vector_provider`` is not a ``QdrantClientWrapper``.
        """
        from clients.qdrant import QdrantClientWrapper

        if not isinstance(self.vector_provider, QdrantClientWrapper):
            raise TypeError(
                f"QdrantSearchPipeline requires QdrantClientWrapper, got {type(self.vector_provider).__name__}"
            )

        query_vector = await self.embedding_provider.embed_text(query_text)

        results = await self.vector_provider.search_async(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
            search_params=self.search_params,
        )

        doc_ids = [self.id_mapper.point_id_to_doc_id(item.point_id, item.payload) for item in results.items]

        return SearchPipelineResult(doc_ids=doc_ids, raw_results=results)
