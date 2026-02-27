"""Qdrant indexing helpers for bulk ingestion.

Uses the synchronous ``QdrantClient`` directly so the pipeline orchestrator
(which runs in a sync context) can disable / re-enable HNSW indexing around
batch uploads without touching the async wrapper used by Ray tasks.
"""

from __future__ import annotations

import logging

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client.http import models as qmodels
    from collections.abc import Generator
    from clients.qdrant import QdrantClientWrapper


logger = logging.getLogger("pipeline.ingestion.qdrant_indexing")


@contextmanager
def qdrant_indexing_disabled(
    client: QdrantClientWrapper, collection: str, *, vector_size: int, distance_metric: qmodels.Distance
) -> Generator[None, None, None]:
    """Disable HNSW indexing on *collection* for faster bulk uploads.

    Args:
        client: Qdrant client wrapper.
        collection: Collection to modify.
    """
    index_disabled = False

    try:
        original_params = client.disable_indexing_sync(
            collection=collection, vector_size=vector_size, distance_metric=distance_metric
        )
        index_disabled = True
    except Exception as e:
        logger.exception("Failed to disable Qdrant indexing", extra={"error": str(e)})

    try:
        yield
    finally:
        try:
            if index_disabled:
                indexing_threshold = original_params.optimizer_config.indexing_threshold or None
                hnsw_m = original_params.hnsw_config.m or None
                client.enable_indexing_sync(
                    collection=collection, indexing_threshold=indexing_threshold, hnsw_m=hnsw_m
                )
            else:
                logger.info("Indexing was already enabled for collection", extra={"collection": collection})
        except Exception as e:
            logger.exception("Failed to enable Qdrant indexing", extra={"error": str(e)})
