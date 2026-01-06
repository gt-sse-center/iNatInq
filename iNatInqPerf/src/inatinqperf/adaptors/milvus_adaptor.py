"""Milvus adaptor."""

from collections.abc import Sequence

import numpy as np
from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from pymilvus import (
    DataType,
    MilvusClient,
    connections,
    utility,
)
from tqdm import tqdm

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import IndexTypeBase, Metric


class MilvusIndexType(IndexTypeBase):
    """Enum for various index types supported by Milvus.

    For more details, see https://milvus.io/docs/index.md?tab=floating.
    """

    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    HNSW_SQ = "HNSW_SQ"
    HNSW_PQ = "HNSW_PQ"


class Milvus(VectorDatabase):
    """Adaptor to help work with Milvus vector database."""

    @logger.catch(reraise=True)
    def __init__(
        self,
        metric: Metric,
        index_type: MilvusIndexType,
        index_params: dict | None = None,
        url: str = "localhost",
        port: int = 19530,
        grpc_port: int = 19530,
        collection_name: str = "default_collection",
        **params,  # noqa: ARG002
    ) -> None:
        super().__init__(metric)

        self.index_type = MilvusIndexType(index_type)
        self.index_name: str = f"{collection_name}_index"
        self.index_params = index_params
        self.collection_name = collection_name

        try:
            connections.connect(host=url, port=port, grpc_port=grpc_port)
            server_type = utility.get_server_type()
            logger.info(f"Milvus server is running. Server type: {server_type}")
        except Exception:
            logger.exception("Milvus server is not running or connection failed")

        # NOTE: pymilvus is very slow to connect, takes ~8 seconds as per profiling.
        self.client = MilvusClient(uri=f"http://{url}:{port}")

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        """Translate metric to Milvus metric type."""
        if metric == Metric.INNER_PRODUCT:
            return "IP"
        if metric == Metric.COSINE:
            return "COSINE"
        if metric == Metric.L2:
            return "L2"

        msg = f"{metric} metric specified is not a valid one for Milvus."
        raise ValueError(msg)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for top-k nearest neighbors.

        The score returned in this case is the distance, so smaller is better.
        """
        # TODO: update this method to use FastAPI search route
        results = self.client.search(
            collection_name=self.collection_name,
            anns_field="vector",
            data=[q.vector],
            limit=topk,
            search_params={
                "metric_type": self._translate_metric(self.metric),
            },
        )

        search_results = []

        # `results` should only have a single value
        result = results[0]

        hit_ids = result.ids
        hit_distances = result.distances
        for hit_id, hit_distance in zip(hit_ids, hit_distances):
            search_results.append(SearchResult(id=hit_id, score=hit_distance))

        return search_results

    def stats(self) -> None:
        """Return index statistics."""
        return self.client.describe_index(collection_name=self.collection_name, index_name=self.index_name)
