"""Vector database-agnostic benchmark orchestrator."""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

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
    from inatinqperf.adaptors.enums import Metric


class Benchmarker:
    """Class to encapsulate all benchmarking operations."""

    def __init__(self, config_file: Path, base_path: Path | None = None) -> None:
        """Construct the benchmark orchestrator.

        Args:
            config_file (Path): Path to the config file with the parameters required to run the benchmark.
            base_path (Path | None, optional): The path to which all data will be saved.
                If None, it will be set to the root directory of the project.
        """
        logger.patch(lambda r: r.update(function="constructor")).info(f"Loading config: {config_file}")

        with config_file.open("r") as f:
            cfg = yaml.safe_load(f)
        # Load into Config class to validate properties
        self.cfg = Config(**cfg)

        if base_path is None:
            self.base_path = Path(__file__).resolve().parent.parent
        else:
            self.base_path = base_path
        self.container_configs = list(self.cfg.containers)

        self.ntotal = 0

    def get_vector_db(self) -> VectorDatabase:
        """Method to initialize the vector database."""
        vdb_type = self.cfg.vectordb.type
        logger.info(f"Building {vdb_type} vector database")

        vectordb_cls = self._resolve_vectordb_class(vdb_type)
        init_params = self.cfg.vectordb.params.to_dict()
        metric: Metric = init_params.pop("metric")
        return vectordb_cls(metric=metric, **init_params)

    @staticmethod
    def _resolve_vectordb_class(vdb_type: str) -> type[VectorDatabase]:
        """Return the adaptor class associated with `vdb_type`."""
        return VECTORDBS[vdb_type.lower()]

    @staticmethod
    def _dataset_to_datapoints(dataset: Dataset) -> list[DataPoint]:
        """Convert a HuggingFace dataset to a list of DataPoint objects."""

        # TODO: add metadata info from dataset if available
        return [
            DataPoint(
                id=int(row_id),
                vector=vector,
                metadata={},
            )
            for idx, (row_id, vector) in enumerate(zip(dataset["id"], dataset["embedding"], strict=True))
        ]

    def search(self, vectordb: VectorDatabase, baseline_results_path: Path | None = None) -> None:
        """Profile search and compute recall@K vs exact baseline."""
        params = self.cfg.vectordb.params
        model_id = self.cfg.embedding.model_id

        topk = self.cfg.search.topk

        queries_file = Path(__file__).resolve().parent.parent / self.cfg.search.queries_file
        queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]

        # Limit the queries
        # If limit is negative, use the full query set
        limit = len(queries) if self.cfg.search.limit < 0 else self.cfg.search.limit
        queries = queries[:limit]

        q = embed_text(queries, model_id)
        logger.info("Embedded all queries")

        # Compute search latencies
        logger.info(f"Performing search on {self.cfg.vectordb.type}")
        with Profiler(f"search-{self.cfg.vectordb.type}", containers=self.container_configs) as p:
            latencies = []
            for i in tqdm(range(q.shape[0])):
                t0 = time.perf_counter()
                vectordb.search(Query(q[i]), topk, **params.to_dict())
                latencies.append((time.perf_counter() - t0) * 1000.0)

            p.sample()

        stats = {
            "vectordb": self.cfg.vectordb.type,
            "index_type": self.cfg.vectordb.params.index_type,
            "topk": topk,
            "lat_ms_avg": float(np.mean(latencies)),
            "lat_ms_p50": float(np.percentile(latencies, 50)),
            "lat_ms_p95": float(np.percentile(latencies, 95)),
        }

        if self.cfg.compute_recall:
            with Path.open(baseline_results_path, mode="rb+") as baseline_results:
                i0 = np.load(baseline_results)

            if i0.shape != (q.shape[0], topk):
                raise RuntimeWarning("Baseline search is not the correct shape, results may be incorrect.")

            logger.info("recall@K (compare last retrieved to baseline per query")
            # For simplicity compute approximate on whole Q at once:
            i1 = np.full((q.shape[0], topk), -1.0, dtype=float)
            for i in tqdm(range(q.shape[0])):
                results = vectordb.search(Query(q[i]), topk, **params.to_dict())
                padded = _ids_to_fixed_array(results, topk)
                i1[i] = padded
            rec = recall_at_k(i1, i0, topk)

            stats["recall@k"] = rec

        # Make values as lists so `tabulate` can print properly.
        table = get_table(stats)
        logger.info(f"\n\n{table}\n\n")

    def run(self) -> None:
        """Run end-to-end benchmark with all steps."""
        with container_context(self.cfg):
            vectordb = None  # TODO: should be adaptor for vectorDB. Will need to update search method to use FastAPI search route

            # Perform search
            self.search(vectordb, self.cfg.baseline.results)


def ensure_dir(p: Path) -> Path:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def recall_at_k(approx_i: np.ndarray, exact_i: np.ndarray, k: int) -> float:
    """Compute recall@K between two sets of indices."""
    hits = 0
    for i in range(approx_i.shape[0]):
        hits += len(set(approx_i[i, :k]).intersection(set(exact_i[i, :k])))
    return hits / float(approx_i.shape[0] * k)


def _ids_to_fixed_array(results: Sequence[SearchResult], topk: int) -> np.ndarray:
    """Convert a list of SearchResult objects into a fixed-length array."""

    arr = np.full(topk, -1.0, dtype=float)
    if not results:
        return arr

    count = min(topk, len(results))
    arr[:count] = [float(results[i].id) for i in range(count)]
    return arr
