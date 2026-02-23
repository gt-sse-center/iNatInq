"""Qdrant indexing helpers for bulk ingestion.

Uses the synchronous ``QdrantClient`` directly so the pipeline orchestrator
(which runs in a sync context) can disable / re-enable HNSW indexing around
batch uploads without touching the async wrapper used by Ray tasks.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger("pipeline.ingestion.qdrant_indexing")


def disable_qdrant_indexing(
    url: str,
    api_key: str | None,
    collection: str,
) -> None:
    """Disable HNSW indexing on *collection* for faster bulk uploads.

    Args:
        url: Qdrant server URL.
        api_key: Optional Qdrant API key.
        collection: Collection to modify.
    """
    client = QdrantClient(url=url, api_key=api_key)
    try:
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
