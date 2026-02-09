"""Unit tests for core.benchmark.metrics.latency module.

This file tests the LatencyStats class for benchmark latency collection:
- Percentile calculations (p50, p95, p99)
- Mean latency calculation
- Queries per second (QPS) throughput
- to_dict() serialization

# Test Coverage

The tests cover:
  - Basic percentile calculations with known inputs
  - Edge cases: empty samples, single sample, zero latencies
  - QPS calculation
  - to_dict() output format
  - add_sample() method

# Running Tests

Run with: uv run pytest tests/unit/core/benchmark/metrics/test_latency.py -v
"""

import pytest

from core.benchmark.metrics.latency import LatencyStats


class TestLatencyStats:
    """Test suite for LatencyStats class."""

    def test_p50_basic_calculation(self):
        """Test p50 (median) calculation with known inputs.

        **Why this test is important:**
          - Validates median calculation
          - Core latency metric for benchmarking

        **What it tests:**
          - Odd number of samples → middle value
          - Values in seconds, output in milliseconds
        """
        # Samples in seconds: [0.1, 0.2, 0.3, 0.4, 0.5]
        stats = LatencyStats(samples=[0.1, 0.2, 0.3, 0.4, 0.5])

        # Median of [0.1, 0.2, 0.3, 0.4, 0.5] = 0.3 seconds = 300ms
        assert stats.p50 == pytest.approx(300.0)

    def test_p50_even_samples(self):
        """Test p50 with even number of samples.

        **Why this test is important:**
          - Even sample count requires interpolation
          - Validates numpy percentile behavior

        **What it tests:**
          - Even number of samples → interpolated median
        """
        # Samples: [0.1, 0.2, 0.3, 0.4]
        stats = LatencyStats(samples=[0.1, 0.2, 0.3, 0.4])

        # Median interpolated between 0.2 and 0.3 = 0.25 seconds = 250ms
        assert stats.p50 == pytest.approx(250.0)

    def test_p95_calculation(self):
        """Test p95 calculation with known inputs.

        **Why this test is important:**
          - p95 is critical for SLA monitoring
          - Validates 95th percentile calculation

        **What it tests:**
          - 95th percentile of sample distribution
        """
        # 100 samples from 0.01 to 1.0 seconds
        samples = [i / 100 for i in range(1, 101)]
        stats = LatencyStats(samples=samples)

        # p95 should be around 0.95 seconds = 950ms
        assert stats.p95 == pytest.approx(950.5, rel=0.01)

    def test_p99_calculation(self):
        """Test p99 calculation with known inputs.

        **Why this test is important:**
          - p99 captures tail latency
          - Critical for worst-case performance analysis

        **What it tests:**
          - 99th percentile of sample distribution
        """
        # 100 samples from 0.01 to 1.0 seconds
        samples = [i / 100 for i in range(1, 101)]
        stats = LatencyStats(samples=samples)

        # p99 should be around 0.99 seconds = 990ms
        assert stats.p99 == pytest.approx(990.1, rel=0.01)

    def test_mean_calculation(self):
        """Test mean (average) calculation.

        **Why this test is important:**
          - Mean provides overall latency picture
          - Simple to verify mathematically

        **What it tests:**
          - Average of samples converted to milliseconds
        """
        # Samples: [0.1, 0.2, 0.3] → mean = 0.2 seconds = 200ms
        stats = LatencyStats(samples=[0.1, 0.2, 0.3])

        assert stats.mean == pytest.approx(200.0)

    def test_qps_calculation(self):
        """Test queries per second calculation.

        **Why this test is important:**
          - QPS measures throughput
          - Critical for capacity planning

        **What it tests:**
          - QPS = count / total_time
        """
        # 5 queries, each taking 0.1 seconds → total = 0.5s → QPS = 10
        stats = LatencyStats(samples=[0.1, 0.1, 0.1, 0.1, 0.1])

        assert stats.qps == pytest.approx(10.0)

    def test_qps_varied_latencies(self):
        """Test QPS with varied latencies.

        **Why this test is important:**
          - Real workloads have varied latencies
          - Validates QPS formula

        **What it tests:**
          - QPS with non-uniform sample times
        """
        # 4 queries: 0.1 + 0.2 + 0.3 + 0.4 = 1.0s total → QPS = 4
        stats = LatencyStats(samples=[0.1, 0.2, 0.3, 0.4])

        assert stats.qps == pytest.approx(4.0)

    def test_empty_samples_p50_returns_zero(self):
        """Test p50 returns 0.0 for empty samples.

        **Why this test is important:**
          - Edge case: no samples collected
          - Must not raise exception

        **What it tests:**
          - Empty samples → p50 = 0.0
        """
        stats = LatencyStats(samples=[])

        assert stats.p50 == 0.0

    def test_empty_samples_p95_returns_zero(self):
        """Test p95 returns 0.0 for empty samples.

        **Why this test is important:**
          - Edge case: no samples collected
          - Must not raise exception

        **What it tests:**
          - Empty samples → p95 = 0.0
        """
        stats = LatencyStats(samples=[])

        assert stats.p95 == 0.0

    def test_empty_samples_p99_returns_zero(self):
        """Test p99 returns 0.0 for empty samples.

        **Why this test is important:**
          - Edge case: no samples collected
          - Must not raise exception

        **What it tests:**
          - Empty samples → p99 = 0.0
        """
        stats = LatencyStats(samples=[])

        assert stats.p99 == 0.0

    def test_empty_samples_mean_returns_zero(self):
        """Test mean returns 0.0 for empty samples.

        **Why this test is important:**
          - Edge case: no samples collected
          - Must not raise exception

        **What it tests:**
          - Empty samples → mean = 0.0
        """
        stats = LatencyStats(samples=[])

        assert stats.mean == 0.0

    def test_empty_samples_qps_returns_zero(self):
        """Test QPS returns 0.0 for empty samples.

        **Why this test is important:**
          - Edge case: no samples collected
          - Must not raise exception

        **What it tests:**
          - Empty samples → QPS = 0.0
        """
        stats = LatencyStats(samples=[])

        assert stats.qps == 0.0

    def test_single_sample(self):
        """Test all metrics with single sample.

        **Why this test is important:**
          - Edge case: only one query
          - All percentiles should equal that sample

        **What it tests:**
          - Single sample → all percentiles equal sample value
        """
        stats = LatencyStats(samples=[0.5])

        assert stats.p50 == pytest.approx(500.0)
        assert stats.p95 == pytest.approx(500.0)
        assert stats.p99 == pytest.approx(500.0)
        assert stats.mean == pytest.approx(500.0)
        assert stats.qps == pytest.approx(2.0)  # 1 query / 0.5s = 2 QPS

    def test_zero_latency_samples(self):
        """Test handling of zero latency samples.

        **Why this test is important:**
          - Edge case: instant responses
          - QPS should handle division by zero

        **What it tests:**
          - All zero latencies → percentiles = 0, QPS = 0
        """
        stats = LatencyStats(samples=[0.0, 0.0, 0.0])

        assert stats.p50 == 0.0
        assert stats.mean == 0.0
        assert stats.qps == 0.0  # Division by zero protection

    def test_to_dict_structure(self):
        """Test to_dict() returns correct structure.

        **Why this test is important:**
          - Serialization format for reporting
          - Must include all expected keys

        **What it tests:**
          - All keys present with _ms suffix for latencies
        """
        stats = LatencyStats(samples=[0.1, 0.2, 0.3])

        result = stats.to_dict()

        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert "mean_ms" in result
        assert "qps" in result
        assert "count" in result

    def test_to_dict_values(self):
        """Test to_dict() returns correct values.

        **Why this test is important:**
          - Values must match property calculations
          - Validates consistency

        **What it tests:**
          - Dict values equal property values
        """
        stats = LatencyStats(samples=[0.1, 0.2, 0.3])

        result = stats.to_dict()

        assert result["p50_ms"] == stats.p50
        assert result["p95_ms"] == stats.p95
        assert result["p99_ms"] == stats.p99
        assert result["mean_ms"] == stats.mean
        assert result["qps"] == stats.qps
        assert result["count"] == 3

    def test_to_dict_empty_samples(self):
        """Test to_dict() with empty samples.

        **Why this test is important:**
          - Edge case: no data collected
          - Should return zeros, not error

        **What it tests:**
          - Empty samples → all zeros in dict
        """
        stats = LatencyStats(samples=[])

        result = stats.to_dict()

        assert result["p50_ms"] == 0.0
        assert result["p95_ms"] == 0.0
        assert result["p99_ms"] == 0.0
        assert result["mean_ms"] == 0.0
        assert result["qps"] == 0.0
        assert result["count"] == 0

    def test_add_sample(self):
        """Test add_sample() method.

        **Why this test is important:**
          - Dynamic sample collection during benchmark
          - Must append to samples list

        **What it tests:**
          - add_sample() appends to samples
        """
        stats = LatencyStats()

        stats.add_sample(0.1)
        stats.add_sample(0.2)
        stats.add_sample(0.3)

        assert len(stats.samples) == 3
        assert stats.samples == [0.1, 0.2, 0.3]

    def test_add_sample_updates_metrics(self):
        """Test that add_sample() affects calculated metrics.

        **Why this test is important:**
          - Metrics should reflect all added samples
          - Validates dynamic calculation

        **What it tests:**
          - Adding samples changes metric values
        """
        stats = LatencyStats()

        stats.add_sample(0.1)
        assert stats.mean == pytest.approx(100.0)

        stats.add_sample(0.3)
        assert stats.mean == pytest.approx(200.0)  # (0.1 + 0.3) / 2 = 0.2s = 200ms

    def test_default_empty_samples(self):
        """Test default initialization has empty samples.

        **Why this test is important:**
          - Default constructor behavior
          - Should start with no samples

        **What it tests:**
          - LatencyStats() has empty samples list
        """
        stats = LatencyStats()

        assert stats.samples == []
        assert len(stats.samples) == 0

    def test_large_sample_set(self):
        """Test with large number of samples.

        **Why this test is important:**
          - Performance with realistic sample sizes
          - Validates numpy handles large arrays

        **What it tests:**
          - 10000 samples computed correctly
        """
        # 10000 samples uniformly distributed
        samples = [i / 10000 for i in range(1, 10001)]
        stats = LatencyStats(samples=samples)

        # Mean should be ~0.5 seconds = 500ms
        assert stats.mean == pytest.approx(500.05, rel=0.01)
        # p50 should be ~0.5 seconds = 500ms
        assert stats.p50 == pytest.approx(500.05, rel=0.01)

    def test_millisecond_conversion(self):
        """Test that output is correctly converted to milliseconds.

        **Why this test is important:**
          - Input is seconds, output must be milliseconds
          - Common source of bugs

        **What it tests:**
          - 1 second input → 1000ms output
        """
        stats = LatencyStats(samples=[1.0])

        assert stats.p50 == pytest.approx(1000.0)
        assert stats.mean == pytest.approx(1000.0)
