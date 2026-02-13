"""Base classes for benchmark runners.

This module provides the foundational abstractions for benchmark execution:
- BenchmarkResult: An attrs data class holding all benchmark output data
- BenchmarkRunner: An abstract base class defining the runner interface

Example:
    ```python
    from core.benchmark.runner.base import BenchmarkRunner, BenchmarkResult

    class MyRunner(BenchmarkRunner):
        async def run(self, provider, dataset, metrics, *, limit=10, warmup_queries=5):
            # ... run benchmark ...
            return BenchmarkResult(
                provider="qdrant",
                dataset="gold-standard",
                metrics={"precision@k": 0.85},
                latency={"p50_ms": 12.5},
            )
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from clients.interfaces.vector_db import VectorDBProvider
    from core.benchmark.datasets.base import Dataset
    from core.benchmark.metrics.base import Metric


@attrs.define
class BenchmarkResult:
    """Result of a benchmark run against a single provider.

    Attributes:
        provider: Name of the vector DB provider benchmarked.
        dataset: Name of the dataset used.
        metrics: Mapping of metric name to aggregated score.
        latency: Mapping of latency stat name to value (e.g., p50_ms, p95_ms).
        timestamp: UTC timestamp of when the benchmark was run.
        config: Optional dict of configuration used for the benchmark.

    Example:
        >>> result = BenchmarkResult(
        ...     provider="qdrant",
        ...     dataset="gold-standard",
        ...     metrics={"precision@k": 0.85, "recall@k": 0.72},
        ...     latency={"p50_ms": 12.5, "p95_ms": 45.0, "qps": 80.0},
        ... )
    """

    provider: str
    dataset: str
    metrics: dict[str, float]
    latency: dict[str, float]
    timestamp: datetime = attrs.field(factory=lambda: datetime.now(UTC))
    config: dict[str, Any] = attrs.Factory(dict)


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners.

    Subclasses must implement the `run()` method to execute a benchmark
    against a vector DB provider using a dataset and set of metrics.
    """

    @abstractmethod
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
        ...
