"""Qdrant indexing helpers for bulk ingestion.

Uses the synchronous ``QdrantClient`` directly so the pipeline orchestrator
(which runs in a sync context) can disable / re-enable HNSW indexing around
batch uploads without touching the async wrapper used by Ray tasks.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


from contextlib import contextmanager

logger = logging.getLogger("pipeline.ingestion.qdrant_indexing")


@contextmanager
def qdrant_indexing_disabled(url: str, api_key: str | None, collection: str):
    """Disable HNSW indexing on *collection* for faster bulk uploads.

    Args:
        url: Qdrant server URL.
        api_key: Optional Qdrant API key.
        collection: Collection to modify.
    """
    index_disabled = False
    # TODO: reset collection parameters to config values rather than using defaults
    try:
        original_params = disable_qdrant_indexing(url, api_key, collection)
        index_disabled = True
    except Exception as e:
        logger.exception("Failed to disable Qdrant indexing", extra={"error": str(e)})

    try:
        yield
    finally:
        try:
            if index_disabled:
                enable_qdrant_indexing(
                    url,
                    api_key,
                    collection,
                    indexing_threshold=original_params.optimizer_config.indexing_threshold,
                    hnsw_m=original_params.hnsw_config.m,
                )
            else:
                logger.info("Indexing was already enabled for collection", extra={"collection": collection})
        except Exception as e:
            logger.exception("Failed to enable Qdrant indexing", extra={"error": str(e)})


def disable_qdrant_indexing(
    url: str,
    api_key: str | None,
    collection: str,
) -> qmodels.CollectionInfo:
    """Disable HNSW indexing on *collection* for faster bulk uploads.

    Args:
        url: Qdrant server URL.
        api_key: Optional Qdrant API key.
        collection: Collection to modify.

    Returns:
        The original collection parameters.
    """
    client = QdrantClient(url=url, api_key=api_key)
    try:
        original_params = client.get_collection(collection_name=collection)
        client.update_collection(
            collection_name=collection,
            optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=0),
            hnsw_config=qmodels.HnswConfigDiff(m=0),
        )
        logger.info(
            "Disabled indexing for collection",
            extra={"collection": collection},
        )
    finally:
        client.close()

    return original_params


def enable_qdrant_indexing(
    url: str,
    api_key: str | None,
    collection: str,
    *,
    indexing_threshold: int = 20_000,
    hnsw_m: int = 16,
) -> None:
    """Re-enable HNSW indexing on *collection* after bulk uploads.

    Args:
        url: Qdrant server URL.
        api_key: Optional Qdrant API key.
        collection: Collection to modify.
        indexing_threshold: Point count threshold before indexing starts.
        hnsw_m: HNSW graph connectivity parameter.
    """
    client = QdrantClient(url=url, api_key=api_key)
    try:
        client.update_collection(
            collection_name=collection,
            optimizer_config=qmodels.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
            hnsw_config=qmodels.HnswConfigDiff(m=hnsw_m),
        )
        logger.info(
            "Re-enabled indexing for collection",
            extra={
                "collection": collection,
                "indexing_threshold": indexing_threshold,
                "hnsw_m": hnsw_m,
            },
        )
    finally:
        client.close()
