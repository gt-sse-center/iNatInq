"""Default benchmark runner implementation.

This module provides a concrete BenchmarkRunner that executes queries against
a vector DB provider, measures latency, and computes IR metrics.

Example:
    ```python
    from core.benchmark.runner.default import DefaultBenchmarkRunner
    from core.benchmark.metrics.ir import PrecisionAtK, RecallAtK

    runner = DefaultBenchmarkRunner()
    result = await runner.run(
        provider=qdrant_provider,
        dataset=gold_standard,
        metrics=[PrecisionAtK(k=10), RecallAtK(k=10)],
        limit=10,
        warmup_queries=5,
    )
    print(result.metrics)  # {"precision@k": 0.85, "recall@k": 0.72}
    print(result.latency)  # {"p50_ms": 12.5, ...}
    ```
"""

from __future__ import annotations

import logging
import time
from itertools import chain, islice
from typing import TYPE_CHECKING

from core.benchmark.metrics.latency import LatencyStats
from core.benchmark.runner.base import BenchmarkResult, BenchmarkRunner

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from clients.interfaces.vector_db import VectorDBProvider
    from core.benchmark.datasets.base import Dataset, Query
    from core.benchmark.metrics.base import Metric
    from core.benchmark.search_pipeline import CLIPSearchPipeline, SearchPipelineResult
    from core.models import SearchResults

logger = logging.getLogger("benchmark.runner.default")


class DefaultBenchmarkRunner(BenchmarkRunner):
    """Default benchmark runner that executes queries and computes metrics.

    Runs a warmup phase (untimed), then a measurement phase where each
    query's latency is recorded and IR metrics are computed per-query
    and aggregated as means.

    Queries are consumed lazily from the dataset iterator in batches to
    avoid materializing all queries into memory at once.

    When a ``search_pipeline`` is provided, queries are embedded via CLIP
    and searched through the pipeline. Otherwise, the runner falls back
    to the provider's ``search_async`` with a placeholder vector.
    """

    def __init__(
        self,
        *,
        batch_size: int = 100,
        search_pipeline: CLIPSearchPipeline | None = None,
        collection: str | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            batch_size: Number of queries to process per batch during
                the measurement phase (default: 100).
            search_pipeline: Optional CLIPSearchPipeline for embed→search→map.
                When set, queries are embedded via CLIP and searched through
                the pipeline. When None, falls back to provider.search_async
                with a placeholder vector.
            collection: Optional collection name override. When set, this
                overrides ``dataset.name`` as the Qdrant collection name.
                Useful when the collection name differs from the dataset name.
        """
        self._batch_size = batch_size
        self._search_pipeline = search_pipeline
        self._collection = collection

    async def run(
        self,
        provider: VectorDBProvider,
        dataset: Dataset,
        metrics: Sequence[Metric],
        *,
        limit: int = 10,
        warmup_queries: int = 5,
    ) -> BenchmarkResult:
        """Execute a benchmark run.

        Args:
            provider: Vector DB provider to benchmark.
            dataset: Benchmark dataset with queries and relevance judgments.
            metrics: Sequence of metrics to compute.
            limit: Number of results to retrieve per query (default: 10).
            warmup_queries: Number of warmup queries to run before timing
                (default: 5).

        Returns:
            BenchmarkResult with aggregated metrics and latency statistics.
        """
        query_iter: Iterator[Query] = dataset.queries()

        collection = self._collection or dataset.name

        # Warmup phase — pull warmup queries lazily from the iterator
        warmup_count = min(warmup_queries, len(dataset))
        warmup_batch = list(islice(query_iter, warmup_count))
        for query in warmup_batch:
            await self._execute_query(provider, dataset, query, limit, collection)

        logger.info(
            "Warmup complete",
            extra={"warmup_queries": len(warmup_batch), "dataset": dataset.name},
        )

        # Measurement phase — chain warmup queries back with remaining iterator,
        # then consume in batches of self._batch_size to limit memory usage.
        measurement_iter: Iterator[Query] = chain(warmup_batch, query_iter)
        latency_stats = LatencyStats()
        per_query_scores: dict[str, list[float]] = {m.name: [] for m in metrics}

        while batch := list(islice(measurement_iter, self._batch_size)):
            for query in batch:
                start = time.perf_counter()
                results = await self._execute_query(provider, dataset, query, limit, collection)
                elapsed = time.perf_counter() - start
                latency_stats.add_sample(elapsed)

                if isinstance(results, tuple):
                    # Pipeline returns (SearchResults, list[str])
                    _, retrieved = results
                else:
                    retrieved = [item.point_id for item in results.items]

                for metric in metrics:
                    score = metric.compute(
                        retrieved,
                        set(query.relevant),
                        graded=query.graded_relevance or None,
                    )
                    per_query_scores[metric.name].append(score)

        # Aggregate: mean across all queries
        aggregated_metrics: dict[str, float] = {}
        for metric_name, scores in per_query_scores.items():
            aggregated_metrics[metric_name] = sum(scores) / len(scores) if scores else 0.0

        return BenchmarkResult(
            provider=type(provider).__name__,
            dataset=dataset.name,
            metrics=aggregated_metrics,
            latency=latency_stats.to_dict(),
            config={"limit": limit, "warmup_queries": warmup_queries, "batch_size": self._batch_size},
        )

    async def _execute_query(
        self,
        provider: VectorDBProvider,
        dataset: Dataset,
        query: Query,
        limit: int,
        collection: str,
    ) -> SearchResults | tuple[SearchResults, list[str]]:
        """Execute a single search query against the provider.

        When ``search_pipeline`` is set, uses it to embed the query via
        CLIP, search the vector DB, and map point IDs to document IDs.
        Returns a ``(SearchResults, doc_ids)`` tuple.

        Without a pipeline, falls back to ``provider.search_async`` with
        a placeholder vector and returns raw ``SearchResults``.

        Args:
            provider: Vector DB provider.
            dataset: Benchmark dataset.
            query: Query to execute.
            limit: Maximum results to retrieve.
            collection: Collection name to search.

        Returns:
            SearchResults (no pipeline) or (SearchResults, doc_ids) tuple.
        """
        if self._search_pipeline is not None:
            pipeline_result: SearchPipelineResult = await self._search_pipeline.search(
                query.text,
                collection=collection,
                limit=limit,
            )
            return pipeline_result.raw_results, pipeline_result.doc_ids

        return await provider.search_async(
            collection=collection,
            query_vector=[],  # placeholder — real impl needs embedding client
            limit=limit,
        )
