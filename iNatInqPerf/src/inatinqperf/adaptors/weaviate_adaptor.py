"""Weaviate adaptor using the v4 client library.

Docs can be found here:
- https://weaviate-python-client.readthedocs.io/en/stable/weaviate.html
- https://docs.weaviate.io/weaviate/guides
"""

from collections.abc import Sequence
from urllib.parse import urlparse

import weaviate
from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.connect import ConnectionParams

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import IndexTypeBase, Metric


class WeaviateIndexType(IndexTypeBase):
    """Enum for various index types supported by Weaviate."""

    FLAT = "flat"
    HNSW = "hnsw"
    DYNAMIC = "dynamic"


class Weaviate(VectorDatabase):
    """Adaptor to help work with the Weaviate vector database."""

    @logger.catch(reraise=True)
    def __init__(
        self,
        metric: Metric,
        index_type: WeaviateIndexType,
        url: str = "http://localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        collection_name: str = "collection_name",
        m: int = 32,
        ef: int = 128,
        **params: object,  # noqa: ARG002
    ) -> None:
        """Initialise the adaptor with a dataset template and connectivity details."""
        super().__init__(metric=metric)

        connection_params = ConnectionParams.from_url(url=f"{url}:{port}", grpc_port=grpc_port)
        # The default timeout is 90 seconds
        self.client = weaviate.WeaviateClient(
            connection_params=connection_params,
            skip_init_checks=False,
        )
        self.client.connect()

        self.collection_name = collection_name
        self.m = m
        # The ef value used during collection construction
        self.ef = ef

        self.index_type_func = self._get_index_type(WeaviateIndexType(index_type))

    @staticmethod
    def _get_index_type(index_type: WeaviateIndexType) -> callable:
        if index_type == WeaviateIndexType.FLAT:
            return Configure.VectorIndex.flat
        if index_type == WeaviateIndexType.HNSW:
            return Configure.VectorIndex.hnsw
        if index_type == WeaviateIndexType.DYNAMIC:
            return Configure.VectorIndex.dynamic

        msg = f"Unsupported index type: '{index_type}'"
        raise ValueError(msg)

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        """Map internal metric names to Weaviate's expected identifiers."""
        if metric == Metric.INNER_PRODUCT:
            return VectorDistances.DOT
        if metric == Metric.COSINE:
            return VectorDistances.COSINE
        if metric == Metric.L2:
            return VectorDistances.L2_SQUARED
        if metric == Metric.MANHATTAN:
            return VectorDistances.MANHATTAN

        msg = f"Unsupported metric '{metric}'"
        raise ValueError(msg)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for the `topk` nearest vectors based on the query point `q`."""

        # TODO: update this method to use FastAPI search route

        collection = self.client.collections.use(self.collection_name)
        response = collection.query.near_vector(
            near_vector=q.vector, limit=topk, return_metadata=MetadataQuery(distance=True, score=True)
        )

        return [SearchResult(id=o.properties["dataset_id"], score=o.metadata.score) for o in response.objects]

    def stats(self) -> dict[str, object]:
        """Return summary statistics sourced via a Weaviate aggregation query."""

        collection = self.client.collections.use(self.collection_name)
        total_count = collection.aggregate.over_all(total_count=True).total_count

        return {
            "ntotal": total_count,
            "metric": self.metric.value,
            "collection_name": self.collection_name,
            "dim": self.dim,
        }


class WeaviateCluster(Weaviate):
    """Adaptor for running benchmarks against a multi-node Weaviate deployment."""

    @logger.catch(reraise=True)
    def __init__(
        self,
        metric: Metric,
        index_type: WeaviateIndexType,
        url: str = "http://localhost",
        port: str = "8080",
        node_urls: Sequence[str] | None = None,
        shard_count: int | None = None,
        replication_factor: int | None = None,
        virtual_per_physical: int | None = None,
        grpc_port: str | int | None = None,
        collection_name: str = "collection_name",
        **params: object,  # noqa: ARG002
    ) -> None:
        """Initialise the adaptor for a sharded Weaviate cluster."""
        VectorDatabase.__init__(self, metric=metric)

        if replication_factor is not None and (shard_count is not None or virtual_per_physical is not None):
            msg = "WeaviateCluster does not support configuring sharding and replication at the same time."
            raise ValueError(msg)

        self.collection_name = collection_name
        self.node_urls = self._resolve_node_urls(node_urls, url, port)
        self.shard_count = shard_count
        self.replication_factor = replication_factor
        self.virtual_per_physical = virtual_per_physical
        self.index_type = index_type
        self.shard_count = shard_count

        grpc = int(grpc_port) if grpc_port is not None else 50051
        connection_params = ConnectionParams.from_url(url=self.node_urls[0], grpc_port=grpc)
        object.__setattr__(connection_params, "grpc_port", grpc)
        self.client = weaviate.WeaviateClient(
            connection_params=connection_params,
            skip_init_checks=False,
        )
        self.client.connect()

    @staticmethod
    def _resolve_node_urls(
        node_urls: Sequence[str] | None,
        url: str,
        port: str,
    ) -> list[str]:
        """Normalise node URLs ensuring scheme + port are always present."""
        if node_urls:
            resolved = [WeaviateCluster._normalise_endpoint(raw, default_port=port) for raw in node_urls]
        else:
            resolved = [WeaviateCluster._normalise_endpoint(url, default_port=port)]

        return resolved

    @staticmethod
    def _normalise_endpoint(raw: str, default_port: str) -> str:
        """Return endpoint with scheme and explicit port."""
        if "://" not in raw:
            raw = f"http://{raw}"

        parsed = urlparse(raw)
        scheme = parsed.scheme or "http"
        host = parsed.hostname
        port = parsed.port or int(default_port)

        if host is None:
            msg = f"Invalid Weaviate node endpoint '{raw}'"
            raise ValueError(msg)

        return f"{scheme}://{host}:{port}"

    def stats(self) -> dict[str, object]:
        """Return cluster stats, including the configured topology."""
        stats = super().stats()
        stats.update(
            {
                "nodes": self.node_urls,
                "shard_count": self.shard_count,
                "replication_factor": self.replication_factor,
                "virtual_per_physical": self.virtual_per_physical,
            }
        )
        return stats
