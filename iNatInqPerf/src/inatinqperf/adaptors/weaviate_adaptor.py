"""Weaviate adaptor using the v4 client library.

Docs can be found here:
- https://weaviate-python-client.readthedocs.io/en/stable/weaviate.html
- https://docs.weaviate.io/weaviate/guides
"""

from collections.abc import Sequence
from loguru import logger

from inatinqperf.adaptors.base import Query, SearchResult, VectorDatabase


class Weaviate(VectorDatabase):
    """Adaptor to help work with the Weaviate vector database."""

    @logger.catch(reraise=True)
    def __init__(
        self,
    ) -> None:
        """Initialise the adaptor with a dataset template and connectivity details."""
        super().__init__()

    def search(self, q: Query, topk: int) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for the `topk` nearest vectors based on the query point `q`."""

        return []
