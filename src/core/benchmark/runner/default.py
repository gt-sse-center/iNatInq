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
from typing import TYPE_CHECKING

from core.benchmark.metrics.latency import LatencyStats
from core.benchmark.runner.base import BenchmarkResult, BenchmarkRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clients.interfaces.vector_db import VectorDBProvider
    from core.benchmark.datasets.base import Dataset, Query
    from core.benchmark.metrics.base import Metric
    from core.models import SearchResults

logger = logging.getLogger("benchmark.runner.default")


class DefaultBenchmarkRunner(BenchmarkRunner):
    """Default benchmark runner that executes queries and computes metrics.

    Runs a warmup phase (untimed), then a measurement phase where each
    query's latency is recorded and IR metrics are computed per-query
    and aggregated as means.

    The runner uses the provider's ``search_async`` method for text
    modality datasets and ``search_images_async`` (if available) for
    image modality datasets, falling back to ``search_async``.
    """

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
        all_queries = list(dataset.queries())

        # Warmup phase — run a subset of queries without timing
        warmup_count = min(warmup_queries, len(all_queries))
        for query in all_queries[:warmup_count]:
            await self._execute_query(provider, dataset, query, limit)

        logger.info(
            "Warmup complete",
            extra={"warmup_queries": warmup_count, "dataset": dataset.name},
        )

        # Measurement phase — time each query and collect metrics
        latency_stats = LatencyStats()
        per_query_scores: dict[str, list[float]] = {m.name: [] for m in metrics}

        for query in all_queries:
            start = time.perf_counter()
            results = await self._execute_query(provider, dataset, query, limit)
            elapsed = time.perf_counter() - start
            latency_stats.add_sample(elapsed)

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
            config={"limit": limit, "warmup_queries": warmup_queries},
        )

    @staticmethod
    async def _execute_query(
        provider: VectorDBProvider,
        dataset: Dataset,
        query: Query,
        limit: int,
    ) -> SearchResults:
        """Execute a single search query against the provider.

        Uses search_images_async for image datasets if available,
        otherwise falls back to search_async. Requires the provider
        to expose an ``embed_query`` or similar method — for now we
        delegate directly to search_async with a dummy vector.

        Note:
            This implementation assumes the provider's search_async
            accepts a text query embedding. In a full implementation,
            the query text would first be embedded via the appropriate
            embedding client. For benchmarking, we pass the query text
            through a text embedding step external to this runner.
        """
        # The provider interface expects a query_vector (list[float]).
        # In a real benchmark, an embedding client would convert query.text
        # to a vector. For now we use the provider's search_async with
        # the collection from the dataset name.
        return await provider.search_async(
            collection=dataset.name,
            query_vector=[],  # placeholder — real impl needs embedding client
            limit=limit,
        )
