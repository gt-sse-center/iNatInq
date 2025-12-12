"""Benchmark orchestrator for FLAT index based vectordb."""

from pathlib import Path

import numpy as np
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from inatinqperf.adaptors import (
    Faiss,
    Query,
    VectorDatabase,
)
from inatinqperf.utils import (
    Profiler,
    embed_text,
)

from .benchmark import Benchmarker, _ids_to_fixed_array


class BaselineBenchmarker(Benchmarker):
    """Class to encapsulate all benchmarking operations."""

    def build(self, dataset: Dataset) -> VectorDatabase:
        """Build the FAISS vector database with a `IndexFlat` index as a baseline."""
        metric = self.cfg.vectordb.params.metric.lower()

        # Create exact baseline
        faiss_flat_db = Faiss(dataset, metric=metric, index_type="FLAT")
        logger.info("Created exact baseline index")

        return faiss_flat_db

    def search(self, vectordb: VectorDatabase, baseline_results_path: Path) -> None:
        """Profile search for exact baseline and save the results for later computing recall@k."""
        model_id = self.cfg.embedding.model_id

        topk = self.cfg.search.topk

        dataset_dir = self.base_path / self.cfg.dataset.directory
        ds = Dataset.load_from_disk(dataset_dir)
        if "query" in ds.column_names:
            queries = ds["query"]

        else:
            queries_file = Path(__file__).resolve().parent.parent / self.cfg.search.queries_file
            queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines() if q.strip()]

        # Limit the queries
        # If limit is negative, use the full query set
        limit = len(queries) if self.cfg.search.limit < 0 else self.cfg.search.limit
        queries = queries[:limit]

        q = embed_text(queries, model_id)
        logger.info("Embedded all queries")

        logger.info("Performing search on baseline")
        with Profiler("search-baseline-FaissFlat", containers=self.container_configs):
            i0 = np.full((q.shape[0], topk), -1.0, dtype=float)
            for i in tqdm(range(q.shape[0])):
                assert vectordb is not None
                base_results = vectordb.search(Query(q[i]), topk)  # exact
                padded = _ids_to_fixed_array(base_results, topk)
                i0[i] = padded

            # Save i0 to baseline_results_path
            with Path.open(baseline_results_path, mode="wb+") as baseline_results:
                np.save(baseline_results, i0)
