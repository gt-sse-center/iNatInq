"""Search pipeline composing CLIP embedding, vector search, and ID mapping.

Provides an end-to-end pipeline: text query → CLIP embedding → vector DB
search → document ID extraction.

Example:
    ```python
    from core.benchmark.search_pipeline import CLIPSearchPipeline

    pipeline = CLIPSearchPipeline(
        clip_client=clip_client,
        vector_provider=qdrant_provider,
        id_mapper=mapper,
    )
    result = await pipeline.search("red bird", collection="images", limit=10)
    print(result.doc_ids)  # ["1316621", "42", ...]
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

from core.benchmark.id_mapping import S3KeyIDMapper  # noqa: TC001 — used as attrs field type at runtime

if TYPE_CHECKING:
    from clients.clip import CLIPClient
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
class CLIPSearchPipeline:
    """Pipeline: text query → CLIP embed → vector search → ID mapping.

    Attributes:
        clip_client: CLIPClient for generating text embeddings.
        vector_provider: VectorDBProvider for similarity search.
        id_mapper: S3KeyIDMapper for translating point IDs to doc IDs.
    """

    clip_client: CLIPClient
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

        1. Embed ``query_text`` via CLIP.
        2. Search the vector DB collection.
        3. Map point IDs to document IDs via the ID mapper.

        Args:
            query_text: Text query to embed and search.
            collection: Qdrant collection name.
            limit: Maximum number of results to return.

        Returns:
            SearchPipelineResult with mapped doc_ids and raw results.
        """
        query_vector = self.clip_client.embed_text(query_text)

        results = await self.vector_provider.search_async(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
        )

        doc_ids = [self.id_mapper.point_id_to_doc_id(item.point_id, item.payload) for item in results.items]

        return SearchPipelineResult(doc_ids=doc_ids, raw_results=results)
