"""Qdrant vector database adaptor."""

from collections.abc import Sequence

from inatinqperf.adaptors.base import Query, SearchResult, VectorDatabase


class Qdrant(VectorDatabase):
    """Qdrant vector database.

    Qdrant only supports a single dense vector index: HNSW.
    However, it supports indexes on the attributes (aka payload) associated with each vector.
    These payload indexes can greatly improve search efficiency.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def search(self, q: Query, topk: int) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for top-k nearest neighbors."""

        # TODO: update this method to use FastAPI search route

        return []
