"""Benchmark metrics package.

Provides metric abstractions and implementations for evaluating
information retrieval and latency performance.
"""

from core.benchmark.metrics.base import Metric, MetricRegistry

__all__ = ["Metric", "MetricRegistry"]
