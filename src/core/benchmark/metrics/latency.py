"""Latency statistics for benchmark evaluation.

This module provides latency metrics collection and percentile calculations
for benchmarking query performance.

Example:
    ```python
    from core.benchmark.metrics.latency import LatencyStats

    # Collect latency samples (in seconds)
    stats = LatencyStats(samples=[0.1, 0.15, 0.12, 0.2, 0.11])

    print(f"p50: {stats.p50}ms")  # Median latency
    print(f"p95: {stats.p95}ms")  # 95th percentile
    print(f"p99: {stats.p99}ms")  # 99th percentile
    print(f"mean: {stats.mean}ms")  # Average latency
    print(f"qps: {stats.qps}")  # Queries per second

    # Get all stats as a dict
    all_stats = stats.to_dict()
    ```
"""

import attrs
import numpy as np


@attrs.define
class LatencyStats:
    """Latency statistics collection and percentile calculations.

    Collects latency samples and computes standard percentile metrics
    used for benchmarking query performance.

    Attributes:
        samples: List of latency samples in seconds.
    """

    samples: list[float] = attrs.field(factory=list)

    @property
    def p50(self) -> float:
        """Return the 50th percentile (median) latency in milliseconds.

        Returns:
            Median latency in milliseconds, or 0.0 if no samples.
        """
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, 50)) * 1000

    @property
    def p95(self) -> float:
        """Return the 95th percentile latency in milliseconds.

        Returns:
            95th percentile latency in milliseconds, or 0.0 if no samples.
        """
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, 95)) * 1000

    @property
    def p99(self) -> float:
        """Return the 99th percentile latency in milliseconds.

        Returns:
            99th percentile latency in milliseconds, or 0.0 if no samples.
        """
        if not self.samples:
            return 0.0
        return float(np.percentile(self.samples, 99)) * 1000

    @property
    def mean(self) -> float:
        """Return the mean (average) latency in milliseconds.

        Returns:
            Mean latency in milliseconds, or 0.0 if no samples.
        """
        if not self.samples:
            return 0.0
        return float(np.mean(self.samples)) * 1000

    @property
    def qps(self) -> float:
        """Return queries per second (throughput).

        Calculated as: count / total_time

        Returns:
            Queries per second, or 0.0 if no samples or total time is 0.
        """
        if not self.samples:
            return 0.0
        total_time = sum(self.samples)
        if total_time == 0:
            return 0.0
        return len(self.samples) / total_time

    def to_dict(self) -> dict[str, float]:
        """Return all statistics as a dictionary.

        Latency values are suffixed with '_ms' to indicate milliseconds.

        Returns:
            Dictionary with all latency stats and throughput.
        """
        return {
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "mean_ms": self.mean,
            "qps": self.qps,
            "count": len(self.samples),
        }

    def add_sample(self, latency: float) -> None:
        """Add a latency sample.

        Args:
            latency: Latency value in seconds.
        """
        self.samples.append(latency)
