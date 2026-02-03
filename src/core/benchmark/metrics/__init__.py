"""Benchmark metrics package.

Provides metric abstractions and implementations for evaluating
information retrieval and latency performance.
"""

from core.benchmark.metrics.base import Metric, MetricRegistry
from core.benchmark.metrics.ir import MRR, NDCG, MeanAveragePrecision, PrecisionAtK, RecallAtK

__all__ = ["MRR", "NDCG", "MeanAveragePrecision", "Metric", "MetricRegistry", "PrecisionAtK", "RecallAtK"]
