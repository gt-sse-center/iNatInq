"""Qdrant vector database adaptor."""

import datetime
import enum
import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct
from tqdm import tqdm

from inquire_api.models import FilterData


class Metric(str, enum.Enum):
    """Enum for metrics used to compute vector similarity.

    Inherit from `str` so we can get human-readable metric names.

    More details about FAISS metrics can be found here: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
    """

    INNER_PRODUCT = "ip"  # inner product
    COSINE = "cosine"  # Cosine distance
    L2 = "l2"  # Euclidean L2 distance
    MANHATTAN = "l1"  # Taxicab/L1 distance

    @classmethod
    def _missing_(cls, value: str) -> "Metric | None":
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


@dataclass
class DataPoint:
    """A single data point in the dataset, which includes the embedding vector and additional metadata."""

    id: int
    vector: Sequence[float]
    metadata: dict[str, object]


@dataclass
class SearchResult:
    """The result of a search query.

    Contains the data point ID and the similarity score.
    """

    id: int
    score: float
    metadata: dict


class VectorDatabaseAdaptor:
    """Qdrant vector database adaptor.

    Qdrant only supports a single dense vector index: HNSW.
    However, it supports indexes on the attributes (aka payload) associated with each vector.
    These payload indexes can greatly improve search efficiency.
    """

    def __init__(
        self,
        collection_name: str = "default_collection",
        metric: Metric = Metric.COSINE,
        url: str = "localhost",
        port: int = 7333,
        grpc_port: int = 7334,
        m: int = 32,
        ef: int = 128,
        **params,  # noqa: ARG002
    ) -> None:
        # Will raise exception if `metric` is not valid.
        self.metric = Metric(metric)

        self.client = QdrantClient(
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=True,
            timeout=10,  # extend the timeout to 10 seconds
        )
        self.collection_name = collection_name

        self.m = m
        # The ef value used during collection construction
        self.ef = ef

    @staticmethod
    def translate_metric(metric: Metric) -> Distance:
        """Helper method to convert from Metric enum to Qdrant Distance."""
        if metric == Metric.INNER_PRODUCT:
            return Distance.DOT
        if metric == Metric.COSINE:
            return Distance.COSINE
        if metric == Metric.L2:
            return Distance.EUCLID
        if metric == Metric.MANHATTAN:
            return Distance.MANHATTAN

        msg = f"{metric} metric specified is not a valid one for Qdrant."
        raise ValueError(msg)

    def initialize_collection(self, dataset: HuggingFaceDataset, batch_size: int) -> None:
        """Create a dataset collection and upload data to it."""
        logger.info(f"Creating collection {self.collection_name}")
        dim = dataset.info.features["img_embedding"].length

        vectors_config = models.VectorParams(
            size=dim,
            distance=self.translate_metric(self.metric),
            on_disk=True,  # save to disk immediately
        )
        index_params = models.HnswConfigDiff(
            m=0,  # disable indexing until dataset upload is complete
            ef_construct=self.ef,
            max_indexing_threads=0,
            on_disk=True,  # Store index on disk
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            hnsw_config=index_params,
            shard_number=4,  # reasonable default as per qdrant docs
        )

        # Batch insert dataset
        num_batches = int(np.ceil(len(dataset) / batch_size))
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=num_batches):
            ids = batch["id"]
            vectors = batch["img_embedding"]
            metadata = [
                {
                    "img_url": batch["img_url"][i],
                    "file_name": batch["file_name"][i],
                    "location": self.get_geo_coordinate(batch["latitude"][i], batch["longitude"][i]),
                    "positional_accuracy": batch["positional_accuracy"][i],
                    "observed_on": self.get_rfc339_date(batch["observed_on"][i]),
                    "species": batch["taxon"][i],
                }
                for i in range(len(batch["id"]))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=metadata,
                ),
            )

        # Set the indexing params
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(m=self.m),
        )

        # Log the number of point uploaded
        num_points_in_db = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count
        logger.info(f"Number of points in Qdrant database: {num_points_in_db}")

        logger.info("Waiting for indexing to complete")
        self.wait_for_index_ready(self.collection_name)
        logger.info("Indexing complete!")

    @staticmethod
    def get_geo_coordinate(latitude: float | None, longitude: float | None) -> dict:
        """Helper method to generate the geo coordinate in the correct format.

        If either the latitude or longitude is None, then set it to 0.0.
        """
        return {
            "lat": 0.0 if latitude is None else float(latitude),
            "lon": 0.0 if longitude is None else float(longitude),
        }

    @staticmethod
    def get_rfc339_date(date: datetime.date | None) -> str:
        """Helper method to get the date in RFC 3339 format."""
        if date is None:
            return ""
        return date.isoformat()

    def wait_for_index_ready(self, collection_name: str, poll_interval: float = 5.0) -> None:
        """Wait until Qdrant reports the collection is fully indexed and ready."""
        while True:
            info = self.client.get_collection(collection_name)

            status = info.status
            optimizer_status = info.optimizer_status

            if status == "green" and optimizer_status == "ok":
                logger.info(f"✅ Index for '{collection_name}' is ready!")
                break

            logger.info(f"⏳ Waiting... status={status}, optimizer_status={optimizer_status}")
            time.sleep(poll_interval)

    @staticmethod
    def _points_iterator(data_points: Sequence[DataPoint]) -> Generator[PointStruct]:
        """A generator to help with creating PointStructs."""
        for data_point in data_points:
            yield PointStruct(id=data_point.id, vector=data_point.vector)

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Upsert vectors with given IDs. This also builds the HNSW index."""
        # Qdrant will override points with the same ID if they already exist,
        # which is the same behavior as `upsert`.
        # Hence we use `upload_points` for performance.
        logger.info("Uploading points to database")
        self.client.upload_points(
            collection_name=self.collection_name,
            points=self._points_iterator(data_points=x),
            parallel=4,
            wait=True,
        )

    def search(
        self,
        query_vector: np.ndarray,
        topk: int,
        filters: FilterData,
        **kwargs,
    ) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors."""
        # Has support for attribute filter: https://qdrant.tech/documentation/quickstart/#add-a-filter

        ef = kwargs.get("ef", self.ef)

        conditions = []

        if filters.species:
            conditions.append(
                models.FieldCondition(
                    key="species",
                    match=models.MatchValue(
                        value=filters.species,
                    ),
                )
            )

        if filters.latitude_min and filters.latitude_max and filters.longitude_min and filters.longitude_max:
            conditions.append(
                models.FieldCondition(
                    key="location",
                    geo_bounding_box=models.GeoBoundingBox(
                        bottom_right=models.GeoPoint(
                            lon=filters.longitude_max,
                            lat=filters.latitude_min,
                        ),
                        top_left=models.GeoPoint(
                            lon=filters.longitude_min,
                            lat=filters.latitude_max,
                        ),
                    ),
                ),
            )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(must=conditions),
            with_payload=True,
            with_vectors=True,
            limit=topk,
            search_params=models.SearchParams(hnsw_ef=ef, exact=False),
        )

        return [SearchResult(point.id, point.score, metadata=point.payload) for point in search_result.points]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

    def delete_collection(self) -> None:
        """Delete the collection associated with this adaptor instance."""
        logger.info(f"Deleting collection {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
        return {
            "metric": self.metric.value,
            "m": self.m,
            "ef": self.ef,
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, "client") and self.client:
            self.client.close()

    def __del__(self) -> None:
        """Destructor method, which automatically closes any open connections."""
        self.close()
