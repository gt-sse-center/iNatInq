"""Benchmark metrics package.

Provides metric abstractions and implementations for evaluating
information retrieval and latency performance.
"""

from core.benchmark.metrics.base import Metric, MetricRegistry
from core.benchmark.metrics.ir import PrecisionAtK, RecallAtK

__all__ = ["Metric", "MetricRegistry", "PrecisionAtK", "RecallAtK"]
