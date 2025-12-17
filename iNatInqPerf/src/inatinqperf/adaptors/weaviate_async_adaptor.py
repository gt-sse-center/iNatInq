"""Weaviate async adaptor using the v4 async client library.

Docs can be found here:
- https://weaviate-python-client.readthedocs.io/en/stable/weaviate.html
- https://docs.weaviate.io/weaviate/guides
"""

import asyncio
from collections.abc import Sequence
from urllib.parse import urlparse

import weaviate
from weaviate.client import WeaviateAsyncClient
from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.connect import ConnectionParams

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import IndexTypeBase, Metric
from inatinqperf.adaptors.weaviate_adaptor import WeaviateIndexType


class WeaviateAsync(VectorDatabase):
    """Async adaptor to help work with the Weaviate vector database."""

    @logger.catch(reraise=True)
    def __init__(
        self,
        metric: Metric,
        index_type: WeaviateIndexType,
        url: str = "http://localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        collection_name: str = "collection_name",
        **params: object,  # noqa: ARG002
    ) -> None:
        """Initialise the adaptor with a dataset template and connectivity details."""
        super().__init__(metric=metric)

        self.collection_name = collection_name
        self.url = url
        self.port = port
        self.grpc_port = grpc_port

        connection_params = ConnectionParams.from_url(
            url=f"{url}:{port}", grpc_port=grpc_port
        )
        object.__setattr__(connection_params, "grpc_port", grpc_port)
        self._connection_params = connection_params
        self.client: WeaviateAsyncClient | None = None

        self.index_type_func = self._get_index_type(WeaviateIndexType(index_type))

    async def _ensure_client_connected(self) -> None:
        """Ensure the client is connected, creating it if necessary."""
        # Always create a new client since each asyncio.run() creates a new event loop
        # and async clients are tied to their event loop
        # Close old client if it exists (from a previous event loop)
        old_client = self.client
        self.client = weaviate.WeaviateAsyncClient(
            connection_params=self._connection_params,
            skip_init_checks=False,
        )
        await self.client.connect()
        # Close old client after new one is connected (in case old one was from different loop)
        if old_client is not None:
            try:
                await old_client.close()
            except Exception:
                # Ignore errors when closing old client from different event loop
                pass

    def _upload_collection(
        self, dataset: HuggingFaceDataset, batch_size: int = 1024
    ) -> None:
        """Create a dataset collection and upload data to it."""

        async def _run_and_cleanup():
            try:
                await self._upload_collection_async(dataset, batch_size)
            finally:
                # Ensure client is closed before event loop ends
                if self.client is not None:
                    try:
                        await self.client.close()
                    except Exception:
                        pass
                    finally:
                        self.client = None

        asyncio.run(_run_and_cleanup())

    async def _upload_collection_async(
        self, dataset: HuggingFaceDataset, batch_size: int = 1024
    ) -> None:
        """Create a dataset collection and upload data to it (async implementation)."""

        await self._ensure_client_connected()

        ## Create collection. If it exists, then log warning and return
        if await self.client.collections.exists(self.collection_name):
            logger.warning("Specified collection already exists, exiting...")
            return

        # The `id` and `vector` properties are created by default
        await self.client.collections.create(
            self.collection_name,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=self.index_type_func(
                    distance_metric=self._translate_metric(self.metric)
                ),
            ),
            properties=[
                Property(
                    name="dataset_id",
                    data_type=DataType.INT,
                    description="The original ID of this data point in the dataset",
                )
            ],
        )

        # Upload the dataset to prepare the database.
        await self._upload_dataset(dataset, batch_size)

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

    async def _upload_dataset(
        self, dataset: HuggingFaceDataset, batch_size: int
    ) -> None:
        """Upload the HuggingFaceDataset to the vector database."""

        await self._ensure_client_connected()

        # Avoid materialising embeddings up front; the client batching API will iterate the dataset.
        num_vectors = len(dataset)
        if num_vectors == 0:
            logger.warning("Dataset contains no embeddings; skipping ingest")
            return

        # Validate dataset schema early so we fail before issuing network calls.
        if "id" not in dataset.column_names or "embedding" not in dataset.column_names:
            msg = "Dataset must contain both 'id' and 'embedding' columns for Weaviate upload."
            raise ValueError(msg)

        collection = self.client.collections.use(self.collection_name)

        async with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for data_row in dataset:
                dataset_id = data_row["id"]
                uuid = weaviate.util.generate_uuid5(dataset_id)
                vector = data_row["embedding"]

                await batch.add_object(
                    uuid=uuid, vector=vector, properties={"dataset_id": dataset_id}
                )

                if batch.number_errors >= 10:  # noqa: PLR2004
                    raise RuntimeError("Batch import stopped due to excessive errors.")

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors and associated metadata."""

        async def _run_and_cleanup():
            try:
                await self.upsert_async(x)
            finally:
                # Ensure client is closed before event loop ends
                if self.client is not None:
                    try:
                        await self.client.close()
                    except Exception:
                        pass
                    finally:
                        self.client = None

        asyncio.run(_run_and_cleanup())

    async def upsert_async(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors and associated metadata (async version)."""

        await self._ensure_client_connected()

        collection = self.client.collections.use(self.collection_name)

        for dp in x:
            uuid = weaviate.util.generate_uuid5(dp.id)

            # Check if data point exists in database. If not, returns None
            data_object = await collection.query.fetch_object_by_id(uuid)

            if data_object is None:
                await collection.data.insert(
                    uuid=uuid,
                    vector=dp.vector,
                    properties={"dataset_id": dp.id},
                )
            else:
                await collection.data.replace(
                    uuid=uuid,
                    vector=dp.vector,
                    properties={"dataset_id": dp.id},
                )

    def search(
        self, q: Query, topk: int, **kwargs
    ) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for the `topk` nearest vectors based on the query point `q`."""

        async def _run_and_cleanup():
            try:
                return await self.search_async(q, topk, **kwargs)
            finally:
                # Ensure client is closed before event loop ends
                if self.client is not None:
                    try:
                        await self.client.close()
                    except Exception:
                        pass
                    finally:
                        self.client = None

        return asyncio.run(_run_and_cleanup())

    async def search_async(
        self, q: Query, topk: int, **kwargs
    ) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for the `topk` nearest vectors based on the query point `q` (async version)."""

        await self._ensure_client_connected()

        collection = self.client.collections.use(self.collection_name)
        response = await collection.query.near_vector(
            near_vector=q.vector,
            limit=topk,
            return_metadata=MetadataQuery(distance=True, score=True),
        )

        return [
            SearchResult(id=o.properties["dataset_id"], score=o.metadata.score)
            for o in response.objects
        ]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete objects corresponding to the provided `ids`.

        If an ID is not present, nothing happens, i.e. it is an idempotent operation.
        """

        async def _run_and_cleanup():
            try:
                await self.delete_async(ids)
            finally:
                # Ensure client is closed before event loop ends
                if self.client is not None:
                    try:
                        await self.client.close()
                    except Exception:
                        pass
                    finally:
                        self.client = None

        asyncio.run(_run_and_cleanup())

    async def delete_async(self, ids: Sequence[int]) -> None:
        """Delete objects corresponding to the provided `ids` (async version).

        If an ID is not present, nothing happens, i.e. it is an idempotent operation.
        """

        await self._ensure_client_connected()

        collection = self.client.collections.get(self.collection_name)

        # https://docs.weaviate.io/weaviate/manage-objects/delete#delete-multiple-objects-by-id
        ids_to_delete = [weaviate.util.generate_uuid5(i) for i in ids]
        await collection.data.delete_many(
            where=Filter.by_id().contains_any(ids_to_delete)
        )

    def stats(self) -> dict[str, object]:
        """Return summary statistics sourced via a Weaviate aggregation query."""

        async def _run_and_cleanup():
            try:
                return await self.stats_async()
            finally:
                # Ensure client is closed before event loop ends
                if self.client is not None:
                    try:
                        await self.client.close()
                    except Exception:
                        pass
                    finally:
                        self.client = None

        return asyncio.run(_run_and_cleanup())

    async def stats_async(self) -> dict[str, object]:
        """Return summary statistics sourced via a Weaviate aggregation query (async version)."""

        await self._ensure_client_connected()

        collection = self.client.collections.use(self.collection_name)
        total_count = (
            await collection.aggregate.over_all(total_count=True)
        ).total_count

        return {
            "ntotal": total_count,
            "metric": self.metric.value,
            "collection_name": self.collection_name,
            "dim": self.dim,
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, "client") and self.client is not None:
            try:
                # Try to get the current event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in a running loop, we can't use asyncio.run()
                    # Just set client to None and let it be cleaned up
                    self.client = None
                except RuntimeError:
                    # No running loop, try to close properly
                    try:
                        asyncio.run(self.close_async())
                    except (RuntimeError, Exception):
                        # Event loop issues, just clear the reference
                        self.client = None
            except Exception:
                # Any other error, just clear the reference
                self.client = None

    async def close_async(self) -> None:
        """Close database connection (async version)."""
        if hasattr(self, "client") and self.client is not None:
            try:
                client_close = getattr(self.client, "close", None)
                if client_close is not None:
                    await client_close()
            except Exception:
                # Ignore errors during shutdown
                pass
            finally:
                self.client = None
