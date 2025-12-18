"""Spark-based benchmark orchestrator."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from datasets import Dataset as HuggingFaceDataset
from loguru import logger

from inatinqperf.adaptors import VECTORDBS, DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.configuration import Config
from inatinqperf.container import container_context
from inatinqperf.utils import (
    Profiler,
    embed_images,
    embed_text,
    export_images,
    get_table,
    load_huggingface_dataset,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


class SparkBenchmarker:
    """Spark-driven benchmark driver with the same stages as the local runner."""

    def __init__(self, config_file: Path, base_path: Path | None = None) -> None:
        logger.patch(lambda r: r.update(function="constructor")).info(f"Loading config: {config_file}")

        with config_file.open("r") as f:
            raw_cfg = yaml.safe_load(f)

        self.spark_cfg = raw_cfg.pop("spark", {}) if isinstance(raw_cfg, dict) else {}
        self.cfg = Config(**raw_cfg)

        if base_path is None:
            self.base_path = Path(__file__).resolve().parent.parent
        else:
            self.base_path = base_path

        self.container_configs = list(self.cfg.containers)
        self._spark: SparkSession | None = None

    def __enter__(self) -> "SparkBenchmarker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        """Stop the Spark session when done."""
        if self._spark is not None:
            self._spark.stop()
            self._spark = None

    def _spark_session(self) -> SparkSession:
        if self._spark is None:
            from pyspark.sql import SparkSession

            if self.spark_cfg.get("clear_hadoop_conf", True):
                os.environ.pop("HADOOP_CONF_DIR", None)
                os.environ.pop("YARN_CONF_DIR", None)

            app_name = self.spark_cfg.get("app_name", "inatinqperf-spark")
            builder = SparkSession.builder.appName(app_name)
            master = self.spark_cfg.get("master")
            if master:
                builder = builder.master(master)
            conf = dict(self.spark_cfg.get("conf", {}))
            force_local_fs = self.spark_cfg.get("force_local_fs", True)
            if force_local_fs:
                conf.setdefault("spark.hadoop.fs.defaultFS", "file:///")
                conf.setdefault(
                    "spark.hadoop.fs.viewfs.impl",
                    "org.apache.hadoop.fs.local.LocalFileSystem",
                )
                conf.setdefault(
                    "spark.hadoop.fs.AbstractFileSystem.viewfs.impl",
                    "org.apache.hadoop.fs.local.LocalFileSystem",
                )
                conf.setdefault("spark.hadoop.fs.viewfs.impl.disable.cache", "true")
            for key, value in conf.items():
                builder = builder.config(key, value)
            self._spark = builder.getOrCreate()
            self._spark.sparkContext.setLogLevel(self.spark_cfg.get("log_level", "WARN"))
            if force_local_fs:
                hconf = self._spark.sparkContext._jsc.hadoopConfiguration()
                hconf.set("fs.defaultFS", "file:///")
                hconf.set("fs.viewfs.impl", "org.apache.hadoop.fs.local.LocalFileSystem")
                hconf.set(
                    "fs.AbstractFileSystem.viewfs.impl",
                    "org.apache.hadoop.fs.local.LocalFileSystem",
                )
        return self._spark

    def _embeddings_parquet_path(self) -> Path:
        return self.base_path / self.cfg.embedding.directory / "spark_embeddings.parquet"

    def _ensure_embeddings_dir(self) -> Path:
        embeddings_dir = self.base_path / self.cfg.embedding.directory
        if embeddings_dir.exists():
            return embeddings_dir
        self._embed_driver()
        return embeddings_dir

    def _load_queries(self) -> list[str]:
        queries_file = self.base_path / "src"/ "inatinqperf" / self.cfg.search.queries_file
        queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]
        limit = len(queries) if self.cfg.search.limit < 0 else self.cfg.search.limit
        return queries[:limit]

    def _vectordb_params(self) -> tuple[str, dict[str, object], object]:
        vdb_type = self.cfg.vectordb.type
        params = self.cfg.vectordb.params.to_dict()
        metric = params.pop("metric")
        return vdb_type, params, metric

    def download(self) -> None:
        """Download HF dataset and optionally export images."""
        dataset_id = self.cfg.dataset.dataset_id
        dataset_dir = self.base_path / self.cfg.dataset.directory

        if dataset_dir.exists():
            logger.info(f"Dataset already exists at {dataset_dir}, continuing...")
            return

        dataset_dir.mkdir(parents=True, exist_ok=True)
        export_raw_images = self.cfg.dataset.export_images
        splits = self.cfg.dataset.splits

        with Profiler(
            f"download-{dataset_id.split('/')[-1]}-{splits}",
            containers=self.container_configs,
        ):
            ds = load_huggingface_dataset(dataset_id, splits)
            ds.save_to_disk(dataset_dir)

            if export_raw_images:
                export_dir = dataset_dir / "images"
                manifest = export_images(ds, export_dir)
                logger.info(f"Exported images to: {export_dir}\nManifest: {manifest}")

        logger.info(f"Downloaded HuggingFace dataset to: {dataset_dir}")

    def _embed_driver(self) -> HuggingFaceDataset:
        """Run single-node embedding and return HF dataset."""
        embeddings_dir = self.base_path / self.cfg.embedding.directory
        if embeddings_dir.exists():
            logger.info(f"Embeddings found at {embeddings_dir}, loading instead of computing")
            return HuggingFaceDataset.load_from_disk(dataset_path=embeddings_dir)

        model_id = self.cfg.embedding.model_id
        batch_size = self.cfg.embedding.batch_size
        dataset_dir = self.base_path / self.cfg.dataset.directory
        logger.info(f"Generating embeddings on driver with model={model_id}")

        with Profiler("embed-images", containers=self.container_configs):
            ds = embed_images(dataset_dir, model_id, batch_size)

        embeddings_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(embeddings_dir)
        return ds

    def embed(self) -> Path:
        """Create or reuse embeddings in a Spark-readable parquet format."""
        embeddings_path = self._embeddings_parquet_path()
        if embeddings_path.exists():
            logger.info(f"Embeddings parquet already exists at {embeddings_path}")
            return embeddings_path

        embeddings_dir = self.base_path / self.cfg.embedding.directory
        if embeddings_dir.exists():
            logger.info(
                f"Found existing HuggingFace embeddings at {embeddings_dir}; converting to parquet for Spark."
            )
            ds = HuggingFaceDataset.load_from_disk(dataset_path=embeddings_dir)
            ds.to_parquet(str(embeddings_path))
            return embeddings_path

        if self.spark_cfg.get("embed_with_spark"):
            logger.info("Spark embedding via image manifests is disabled; using driver embedding.")

        logger.info("Falling back to driver embedding for parquet generation.")
        ds = self._embed_driver()
        ds.to_parquet(str(embeddings_path))
        return embeddings_path

    def _load_embeddings_df(self) -> DataFrame:
        spark = self._spark_session()
        use_parquet = self.spark_cfg.get("use_parquet_embeddings", True)
        if use_parquet:
            embeddings_path = self.embed()
            logger.info(f"Loading embeddings parquet from: {embeddings_path}")
            try:
                df = spark.read.parquet(str(embeddings_path))
                df = df.select("id", "embedding")
                if partitions := self.spark_cfg.get("partitions"):
                    df = df.repartition(int(partitions))
                return df
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Spark parquet read failed ({exc}); falling back to HuggingFace dataset.")

        embeddings_dir = self._ensure_embeddings_dir()
        embeddings_dir_str = str(embeddings_dir)
        partitions = int(
            self.spark_cfg.get("partitions") or (spark.sparkContext.defaultParallelism or 1)
        )

        def _load_partition(partition_idx: int, iterator: Iterable[object]) -> Iterable[tuple[int, list[float]]]:
            from datasets import Dataset as HFDataset

            _ = list(iterator)
            ds = HFDataset.load_from_disk(embeddings_dir_str)
            shard = ds.shard(num_shards=partitions, index=partition_idx, contiguous=True)
            if "embedding" not in shard.column_names:
                msg = "Embeddings dataset must contain an 'embedding' column."
                raise ValueError(msg)
            has_id = "id" in shard.column_names
            for idx, row in enumerate(shard):
                row_id = int(row["id"]) if has_id else idx
                emb = row["embedding"]
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                yield row_id, emb

        seed_rdd = spark.sparkContext.parallelize(range(partitions), partitions)
        embeddings_rdd = seed_rdd.mapPartitionsWithIndex(_load_partition)

        from pyspark.sql.types import ArrayType, FloatType, LongType, StructField, StructType

        schema = StructType(
            [
                StructField("id", LongType(), nullable=False),
                StructField("embedding", ArrayType(FloatType()), nullable=False),
            ]
        )
        return spark.createDataFrame(embeddings_rdd, schema=schema)

    def _create_seed_dataset(self, embeddings_df: DataFrame) -> HuggingFaceDataset:
        sample = embeddings_df.limit(1).collect()
        if not sample:
            msg = "Embeddings parquet is empty; cannot create collection."
            raise RuntimeError(msg)
        row = sample[0]
        return HuggingFaceDataset.from_dict({"id": [int(row.id)], "embedding": [row.embedding]})

    def get_vector_db(self) -> VectorDatabase:
        vdb_type, params, metric = self._vectordb_params()
        vectordb_cls = VECTORDBS[vdb_type.lower()]
        return vectordb_cls(metric=metric, **params)

    def build(self) -> None:
        """Create the collection and upload embeddings using Spark tasks."""
        vdb_type, params, metric = self._vectordb_params()
        embeddings_df = self._load_embeddings_df()
        seed = self._create_seed_dataset(embeddings_df)
        batch_size = self.cfg.vectordb.params.batch_size

        with Profiler(f"spark-build-{vdb_type}", containers=self.container_configs):
            vdb = self.get_vector_db()
            vdb.initialize_collection(seed, batch_size=batch_size)
            _safe_close(vdb)

            def _upsert_partition(rows: Iterable[object]) -> Iterable[int]:
                vdb_local = VECTORDBS[vdb_type.lower()](metric=metric, **params)
                batch: list[DataPoint] = []
                count = 0
                for row in rows:
                    batch.append(DataPoint(id=int(row.id), vector=row.embedding, metadata={}))
                    if len(batch) >= batch_size:
                        vdb_local.upsert(batch)
                        count += len(batch)
                        batch.clear()
                if batch:
                    vdb_local.upsert(batch)
                    count += len(batch)
                _safe_close(vdb_local)
                yield count

            counts = embeddings_df.rdd.mapPartitions(_upsert_partition).collect()

        stats_vdb = self.get_vector_db()
        try:
            logger.info(f"Stats: {stats_vdb.stats()}")
        finally:
            _safe_close(stats_vdb)
        logger.info(f"Upserted {sum(counts)} vectors into {vdb_type}")

    def search(self, baseline_results_path: Path | None = None) -> None:
        """Run parallel search with Spark and compute latency stats."""
        vdb_type, params, metric = self._vectordb_params()
        topk = self.cfg.search.topk
        queries = self._load_queries()
        if not queries:
            logger.warning("No queries available for Spark search")
            return

        q = embed_text(queries, self.cfg.embedding.model_id)
        spark = self._spark_session()

        query_rows = [(idx, q[idx].tolist()) for idx in range(q.shape[0])]
        query_df = spark.createDataFrame(query_rows, ["query_index", "embedding"])
        if partitions := self.spark_cfg.get("partitions"):
            query_df = query_df.repartition(int(partitions))

        def _search_partition(rows: Iterable[object]) -> Iterable[tuple[int, float, list[int]]]:
            vdb_local = VECTORDBS[vdb_type.lower()](metric=metric, **params)
            for row in rows:
                t0 = time.perf_counter()
                results = vdb_local.search(Query(row.embedding), topk, **params)
                latency = (time.perf_counter() - t0) * 1000.0
                ids = [int(r.id) for r in results]
                if len(ids) < topk:
                    ids.extend([-1] * (topk - len(ids)))
                yield int(row.query_index), latency, ids
            _safe_close(vdb_local)

        results_rdd = query_df.rdd.mapPartitions(_search_partition)

        from pyspark.sql.types import ArrayType, FloatType, LongType, StructField, StructType

        schema = StructType(
            [
                StructField("query_index", LongType(), nullable=False),
                StructField("latency_ms", FloatType(), nullable=False),
                StructField("ids", ArrayType(LongType()), nullable=False),
            ]
        )
        results_df = spark.createDataFrame(results_rdd, schema=schema)

        from pyspark.sql import functions as F

        lat_df = results_df.select("latency_ms")
        lat_ms_avg = float(lat_df.agg(F.avg("latency_ms")).collect()[0][0])
        p50, p95 = lat_df.approxQuantile("latency_ms", [0.5, 0.95], 0.01)

        stats = {
            "vectordb": vdb_type,
            "index_type": self.cfg.vectordb.params.index_type,
            "topk": topk,
            "lat_ms_avg": lat_ms_avg,
            "lat_ms_p50": float(p50),
            "lat_ms_p95": float(p95),
        }

        if self.cfg.compute_recall and baseline_results_path is not None:
            i0 = np.load(baseline_results_path)
            ordered = results_df.orderBy("query_index").select("ids").collect()
            i1 = np.asarray([row.ids for row in ordered], dtype=float)
            if i0.shape != i1.shape:
                msg = "Baseline search is not the correct shape, results may be incorrect."
                raise RuntimeError(msg)
            stats["recall@k"] = recall_at_k(i1, i0, topk)

        table = get_table(stats)
        logger.info(f"\n\n{table}\n\n")

    def update(self) -> None:
        """Upsert + delete small batch and re-search."""
        vdb_type, params, metric = self._vectordb_params()
        add_n = self.cfg.update["add_count"]
        del_n = self.cfg.update["delete_count"]
        embeddings_df = self._load_embeddings_df()

        from pyspark.sql import functions as F

        max_id_row = embeddings_df.agg(F.max("id")).collect()[0][0]
        max_existing_id = int(max_id_row) if max_id_row is not None else -1
        next_id = max_existing_id + 1

        add_vectors = embeddings_df.select("embedding").limit(add_n).collect()
        if not add_vectors:
            logger.warning("No embeddings available for update; skipping.")
            return
        if len(add_vectors) < add_n:
            add_n = len(add_vectors)
        base_vecs = np.asarray([row.embedding for row in add_vectors], dtype=np.float32)
        rng = np.random.default_rng(42)
        add_vecs = base_vecs + rng.normal(0, 0.01, size=base_vecs.shape).astype(np.float32)
        add_ids = list(range(next_id, next_id + add_n))

        spark = self._spark_session()
        add_rows = [(add_ids[i], add_vecs[i].tolist()) for i in range(add_n)]
        add_df = spark.createDataFrame(add_rows, ["id", "embedding"])

        def _upsert_partition(rows: Iterable[object]) -> None:
            vdb_local = VECTORDBS[vdb_type.lower()](metric=metric, **params)
            batch: list[DataPoint] = []
            for row in rows:
                batch.append(DataPoint(id=int(row.id), vector=row.embedding, metadata={}))
            if batch:
                vdb_local.upsert(batch)
            _safe_close(vdb_local)

        logger.info(f"Performing update with {add_n} additions and {del_n} deletions.")
        with Profiler(f"spark-update-add-{vdb_type}", containers=self.container_configs):
            add_df.foreachPartition(_upsert_partition)

        del_ids = add_ids[:del_n]
        if del_ids:
            del_df = spark.createDataFrame([(i,) for i in del_ids], ["id"])

            def _delete_partition(rows: Iterable[object]) -> None:
                vdb_local = VECTORDBS[vdb_type.lower()](metric=metric, **params)
                ids = [int(row.id) for row in rows]
                if ids:
                    vdb_local.delete(ids)
                _safe_close(vdb_local)

            with Profiler(f"spark-update-delete-{vdb_type}", containers=self.container_configs):
                del_df.foreachPartition(_delete_partition)

    def update_and_search(self) -> None:
        """Run update workflow then search again to capture post-update performance."""
        self.update()
        self.search(self.cfg.baseline.results_post_update)

    def run(self) -> None:
        """Run end-to-end Spark benchmark."""
        self.download()
        self.embed()

        with container_context(self.cfg):
            self.build()
            self.search(self.cfg.baseline.results)
            self.update_and_search()


def recall_at_k(approx_i: np.ndarray, exact_i: np.ndarray, k: int) -> float:
    """Compute recall@K between two sets of indices."""
    hits = 0
    for i in range(approx_i.shape[0]):
        hits += len(set(approx_i[i, :k]).intersection(set(exact_i[i, :k])))
    return hits / float(approx_i.shape[0] * k)


def _safe_close(vdb: VectorDatabase) -> None:
    """Close adapters that are safe to close inside Spark tasks."""
    if vdb.__class__.__name__.lower().startswith("milvus"):
        return
    vdb.close()
