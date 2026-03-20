"""Comparison runner for benchmarking multiple providers side-by-side.

This module provides a runner that evaluates multiple vector DB providers
against the same dataset and metrics, then feeds results to reporters.

Example:
    ```python
    from core.benchmark.runner.comparison import ComparisonRunner

    runner = ComparisonRunner(
        providers=[qdrant_provider],
        reporters=[console_reporter, json_reporter],
    )
    results = await runner.compare(dataset, metrics=[PrecisionAtK(), RecallAtK()])
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.benchmark.metrics.ir import MRR, NDCG, MeanAveragePrecision, PrecisionAtK, RecallAtK
from core.benchmark.runner.default import DefaultBenchmarkRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clients.interfaces.vector_db import VectorDBProvider
    from core.benchmark.datasets.base import Dataset
    from core.benchmark.metrics.base import Metric
    from core.benchmark.reporters.base import Reporter
    from core.benchmark.runner.base import BenchmarkResult

logger = logging.getLogger("benchmark.runner.comparison")

DEFAULT_METRICS: list[Metric] = [
    PrecisionAtK(),
    RecallAtK(),
    MeanAveragePrecision(),
    NDCG(),
    MRR(),
]


class ComparisonRunner:
    """Runs benchmarks across multiple providers and reports results.

    Iterates over each provider, delegates to DefaultBenchmarkRunner,
    and calls all configured reporters with the collected results.

    Attributes:
        providers: List of vector DB providers to benchmark.
        reporters: Optional list of reporters to call with results.
    """

    def __init__(
        self,
        providers: Sequence[VectorDBProvider],
        reporters: Sequence[Reporter] | None = None,
    ) -> None:
        self._providers = list(providers)
        self._reporters = list(reporters) if reporters else []
        self._runner = DefaultBenchmarkRunner()

    async def compare(
        self,
        dataset: Dataset,
        metrics: Sequence[Metric] | None = None,
        *,
        limit: int = 10,
        warmup_queries: int = 5,
    ) -> dict[str, BenchmarkResult]:
        """Run benchmarks on all providers and report results.

        Args:
            dataset: Benchmark dataset with queries and relevance judgments.
            metrics: Metrics to compute. Defaults to DEFAULT_METRICS.
            limit: Number of results to retrieve per query.
            warmup_queries: Number of warmup queries before timing.

        Returns:
            Mapping of provider name to BenchmarkResult.
        """
        active_metrics = list(metrics) if metrics is not None else DEFAULT_METRICS

        results: dict[str, BenchmarkResult] = {}
        for provider in self._providers:
            provider_name = type(provider).__name__
            logger.info("Benchmarking provider: %s", provider_name)

            result = await self._runner.run(
                provider=provider,
                dataset=dataset,
                metrics=active_metrics,
                limit=limit,
                warmup_queries=warmup_queries,
            )
            results[provider_name] = result

        for reporter in self._reporters:
            await reporter.report(results)

        return results
