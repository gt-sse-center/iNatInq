"""In-memory semantic cache backed by Qdrant.

Provides a ``CacheClient`` that stores and retrieves search results keyed
on query embedding vectors.  The underlying store is an in-memory Qdrant
instance (``location=":memory:"``), so cached data lives only for the
lifetime of the process.

Each vector-DB collection gets its own ``cache_{collection_name}``
collection inside the in-memory Qdrant, providing per-collection isolation.

The cache is fully optional: when ``SemanticCacheConfig.enabled`` is
``False``, the ``get_semantic_cache()`` singleton returns ``None`` and the
rest of the system operates as if caching does not exist.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import attrs
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from config import SemanticCacheConfig
from core.models import SearchResultItem, SearchResults

logger = logging.getLogger("clients.cache")


def _cache_collection_name(collection: str) -> str:
    """Return the internal Qdrant collection name for a given user collection."""
    return f"cache_{collection}"


class CacheClient:
    """In-memory semantic cache using Qdrant vector similarity.

    Each user-facing collection is stored in a separate ``cache_{name}``
    collection inside the in-memory Qdrant instance.  Collections are
    lazily created on the first ``store()`` call, so no ``vector_size``
    is required at construction time.

    Args:
        config: Semantic cache configuration.
    """

    def __init__(self, config: SemanticCacheConfig) -> None:
        self._client = AsyncQdrantClient(location=":memory:", timeout=config.timeout_s)
        self._config = config
        self._known_collections: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def config(self) -> SemanticCacheConfig:
        """Return the cache configuration (read-only)."""
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def lookup(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
    ) -> SearchResults | None:
        """Look up cached results for *query_vector*.

        Returns the cached ``SearchResults`` when the best match scores
        at or above ``similarity_threshold``; otherwise returns ``None``
        (cache miss).  Returns ``None`` for non-existent collections.

        Args:
            collection: User-facing collection name.
            query_vector: Query embedding vector.
            limit: Maximum number of result items to return.

        Returns:
            Cached ``SearchResults`` truncated to *limit*, or ``None``.
        """
        cache_col = _cache_collection_name(collection)
        if cache_col not in self._known_collections:
            return None
        try:
            hits = await self._client.query_points(
                collection_name=cache_col,
                query=query_vector,
                limit=1,
                with_payload=True,
            )
            if not hits.points:
                logger.debug("Cache miss (empty)")
                return None

            best = hits.points[0]
            if best.score is not None and best.score >= self._config.similarity_threshold:
                logger.debug("Cache hit (score=%.4f)", best.score)
                results = _deserialize_results(best.payload or {})
                # Truncate to requested limit
                if len(results.items) > limit:
                    return SearchResults(items=results.items[:limit], total=results.total)
                return results

            logger.debug(
                "Cache miss (score=%.4f < threshold=%.4f)",
                best.score or 0.0,
                self._config.similarity_threshold,
            )
            return None
        except Exception:
            logger.warning("Cache lookup failed; treating as miss", exc_info=True)
            return None

    async def store(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str,
        results: SearchResults,
        limit: int,
    ) -> None:
        """Store *results* in the cache keyed by *query_vector*.

        Lazily creates the ``cache_{collection}`` collection on first use.
        Evicts a random entry when ``max_entries_per_collection`` is reached.

        Args:
            collection: User-facing collection name.
            query_vector: Query embedding vector.
            query_text: Original query text.
            results: Search results to cache.
            limit: The limit from the original query.
        """
        try:
            cache_col = _cache_collection_name(collection)
            await self._ensure_collection(cache_col, vector_size=len(query_vector))

            count_result = await self._client.count(collection_name=cache_col, exact=True)
            if count_result.count >= self._config.max_entries_per_collection:
                await self._evict_random(cache_col)

            point_id = str(uuid.uuid4())
            await self._client.upsert(
                collection_name=cache_col,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload={
                            "query_text": query_text,
                            "results_json": _serialize_results(results),
                            "stored_at": time.time(),
                            "original_limit": limit,
                        },
                    )
                ],
            )
        except Exception:
            logger.warning("Cache store failed; skipping", exc_info=True)

    async def invalidate(self, collection: str | None = None) -> None:
        """Invalidate (delete) cached data.

        Args:
            collection: If given, delete only ``cache_{collection}``.
                If ``None``, delete all known cache collections.
        """
        try:
            if collection is not None:
                cache_col = _cache_collection_name(collection)
                if cache_col in self._known_collections:
                    await self._client.delete_collection(collection_name=cache_col)
                    self._known_collections.discard(cache_col)
            else:
                for cache_col in list(self._known_collections):
                    await self._client.delete_collection(collection_name=cache_col)
                self._known_collections.clear()
        except Exception:
            logger.warning("Cache invalidate failed", exc_info=True)

    async def size(self, collection: str) -> int:
        """Return the number of cached entries for *collection*.

        Returns 0 if the collection does not exist.
        """
        cache_col = _cache_collection_name(collection)
        if cache_col not in self._known_collections:
            return 0
        try:
            result = await self._client.count(collection_name=cache_col, exact=True)
            return result.count
        except Exception:
            logger.warning("Cache size failed", exc_info=True)
            return 0

    async def close(self) -> None:
        """Close the underlying Qdrant client."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_collection(self, cache_col: str, vector_size: int) -> None:
        """Lazily create a cache collection, guarded by an asyncio lock."""
        if cache_col in self._known_collections:
            return
        async with self._lock:
            if cache_col in self._known_collections:
                return
            await self._client.create_collection(
                collection_name=cache_col,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            self._known_collections.add(cache_col)
            logger.info("Created cache collection %s (vector_size=%d)", cache_col, vector_size)

    async def _evict_random(self, cache_col: str) -> None:
        """Evict one arbitrary entry from *cache_col*."""
        scroll_result = await self._client.scroll(
            collection_name=cache_col,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        ids = [p.id for p in scroll_result[0]]
        if ids:
            await self._client.delete(
                collection_name=cache_col,
                points_selector=qmodels.PointIdsList(points=ids),
            )
            logger.debug("Evicted 1 cache entry from %s", cache_col)


# =============================================================================
# Serialization helpers
# =============================================================================


def _serialize_results(results: SearchResults) -> str:
    """Serialize ``SearchResults`` to a JSON string for cache storage."""
    return json.dumps(attrs.asdict(results))


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


# =============================================================================
# Module-level singleton
# =============================================================================

_cache_instance: CacheClient | None = None
_cache_initialized: bool = False


def get_semantic_cache() -> CacheClient | None:
    """Return the shared ``CacheClient`` singleton, or ``None`` when disabled.

    Creates the instance on first call.  Subsequent calls return the
    same instance.
    """
    global _cache_instance, _cache_initialized  # noqa: PLW0603
    if _cache_initialized:
        return _cache_instance
    _cache_initialized = True
    config = SemanticCacheConfig.from_env()
    if not config.enabled:
        return None
    _cache_instance = CacheClient(config=config)
    return _cache_instance
