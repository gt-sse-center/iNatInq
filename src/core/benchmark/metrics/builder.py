"""Build Metric instances from BenchmarkConfig.

Translates configuration (metric names and k values) into concrete
Metric instances ready for use by the benchmark runner.

Example:
    ```python
    from core.benchmark.config import BenchmarkConfig
    from core.benchmark.metrics.builder import build_metrics_from_config

    config = BenchmarkConfig(k_values=[5, 10], metrics=["precision@k", "map"])
    metrics = build_metrics_from_config(config)
    # [PrecisionAtK(k=5), PrecisionAtK(k=10), MeanAveragePrecision()]
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.benchmark.metrics.ir import MRR, NDCG, MeanAveragePrecision, PrecisionAtK, RecallAtK

if TYPE_CHECKING:
    from core.benchmark.config import BenchmarkConfig
    from core.benchmark.metrics.base import Metric

logger = logging.getLogger("benchmark.metrics.builder")

# Metrics that get instantiated once per k value
_K_DEPENDENT_METRICS: dict[str, type] = {
    "precision@k": PrecisionAtK,
    "recall@k": RecallAtK,
    "ndcg": NDCG,
}

# Metrics that are instantiated once (k-independent)
_SCALAR_METRICS: dict[str, type] = {
    "map": MeanAveragePrecision,
    "mrr": MRR,
}


def build_metrics_from_config(config: BenchmarkConfig) -> list[Metric]:
    """Build a list of Metric instances from a BenchmarkConfig.

    For each metric name in ``config.metrics``:
    - K-dependent metrics (precision@k, recall@k, ndcg) are instantiated
      once per k value in ``config.k_values``.
    - Scalar metrics (map, mrr) are instantiated once.
    - Unknown metric names are logged and skipped.

    Args:
        config: Benchmark configuration with metric names and k values.

    Returns:
        List of Metric instances ready for computation.
    """
    metrics: list[Metric] = []

    for metric_name in config.metrics:
        if metric_name in _K_DEPENDENT_METRICS:
            cls = _K_DEPENDENT_METRICS[metric_name]
            metrics.extend(cls(k=k) for k in config.k_values)
        elif metric_name in _SCALAR_METRICS:
            cls = _SCALAR_METRICS[metric_name]
            metrics.append(cls())
        else:
            logger.warning("Unknown metric '%s' in config — skipping", metric_name)

    return metrics
