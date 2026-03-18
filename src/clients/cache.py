"""In-memory semantic cache backed by Qdrant.

Provides a ``CacheClient`` that stores and retrieves search results keyed
on query embedding vectors.  The underlying store is an in-memory Qdrant
instance (``location=":memory:"``), so cached data lives only for the
lifetime of the process.

The cache is fully optional: when no ``CacheConfig`` is provided the rest
of the system operates as if caching does not exist.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from core.models import SearchResultItem, SearchResults

if TYPE_CHECKING:
    from config import CacheConfig

logger = logging.getLogger("clients.cache")

_COLLECTION = "semantic_cache"


class CacheClient:
    """In-memory semantic cache using Qdrant vector similarity.

    Args:
        config: Cache configuration (max_items, threshold, timeout).
        vector_size: Embedding dimension used by the search pipeline.
    """

    def __init__(self, config: CacheConfig, vector_size: int) -> None:
        self._client = AsyncQdrantClient(location=":memory:", timeout=config.timeout)
        self._config = config
        self._vector_size = vector_size
        self._healthy = False

    async def initialize(self) -> None:
        """Create the cache collection.  Call once at startup.

        On failure the client is marked unhealthy and all subsequent
        ``get``/``put`` calls become no-ops so the search pipeline is
        never blocked by a cache problem.
        """
        try:
            await self._client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            self._healthy = True
            logger.info("Semantic cache initialized (max_items=%d)", self._config.max_items)
        except Exception:
            logger.warning("Failed to initialize semantic cache; caching disabled", exc_info=True)

    async def get(
        self,
        query_vector: list[float],
        collection: str,
    ) -> SearchResults | None:
        """Look up cached results for *query_vector*.

        Returns the cached ``SearchResults`` when the best match scores
        at or above ``cache_hit_score_threshold``; otherwise returns
        ``None`` (cache miss).
        """
        if not self._healthy:
            return None
        try:
            hits = await self._client.query_points(
                collection_name=_COLLECTION,
                query=query_vector,
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="collection",
                            match=qmodels.MatchValue(value=collection),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if not hits.points:
                logger.debug("Cache miss (empty)")
                return None
            best = hits.points[0]
            if best.score is not None and best.score >= self._config.cache_hit_score_threshold:
                logger.debug("Cache hit (score=%.4f)", best.score)
                return _deserialize_results(best.payload or {})
            logger.debug(
                "Cache miss (score=%.4f < threshold=%.4f)",
                best.score or 0.0,
                self._config.cache_hit_score_threshold,
            )
            return None
        except Exception:
            logger.warning("Cache get failed; treating as miss", exc_info=True)
            return None

    async def put(
        self,
        query_vector: list[float],
        results: SearchResults,
        collection: str,
    ) -> None:
        """Store *results* in the cache keyed by *query_vector*.

        Evicts the oldest entry (FIFO) when ``max_items`` is reached.
        """
        if not self._healthy:
            return
        try:
            count_result = await self._client.count(collection_name=_COLLECTION, exact=True)
            if count_result.count >= self._config.max_items:
                await self._evict(count_result.count - self._config.max_items + 1)

            point_id = str(uuid.uuid4())
            await self._client.upsert(
                collection_name=_COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload={
                            "collection": collection,
                            "results_json": _serialize_results(results),
                        },
                    )
                ],
            )
        except Exception:
            logger.warning("Cache put failed; skipping", exc_info=True)

    async def _evict(self, n: int) -> None:
        """Delete the *n* oldest entries (by scroll order / FIFO)."""
        scroll_result = await self._client.scroll(
            collection_name=_COLLECTION,
            limit=n,
            with_payload=False,
            with_vectors=False,
        )
        ids = [p.id for p in scroll_result[0]]
        if ids:
            await self._client.delete(
                collection_name=_COLLECTION,
                points_selector=qmodels.PointIdsList(points=ids),
            )
            logger.debug("Evicted %d cache entries", len(ids))

    async def close(self) -> None:
        """Close the underlying Qdrant client."""
        await self._client.close()


def _serialize_results(results: SearchResults) -> str:
    """Serialize ``SearchResults`` to a JSON string for cache storage."""
    return json.dumps(
        {
            "items": [
                {"point_id": item.point_id, "score": item.score, "payload": item.payload}
                for item in results.items
            ],
            "total": results.total,
        }
    )


def _deserialize_results(payload: dict) -> SearchResults:
    """Reconstruct ``SearchResults`` from a cache payload."""
    raw = json.loads(payload["results_json"])
    items = [
        SearchResultItem(
            point_id=item["point_id"],
            score=item["score"],
            payload=item["payload"],
        )
        for item in raw["items"]
    ]
    return SearchResults(items=items, total=raw["total"])
