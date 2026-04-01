"""Unit tests for benchmark runner base classes.

Tests the abstract BenchmarkRunner interface and runner orchestration lifecycle
to ensure correct dataset loading, metric computation, and result reporting.
This is important because runners coordinate all benchmark components into a
complete evaluation pipeline.
"""

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from core.benchmark.datasets.base import Dataset, Query
from core.benchmark.metrics.base import Metric
from core.benchmark.runner.base import BenchmarkResult, BenchmarkRunner


class TestBenchmarkResult:
    """Tests for BenchmarkResult attrs class."""

    def test_required_fields(self):
        """BenchmarkResult stores provider, dataset, metrics, latency."""
        result = BenchmarkResult(
            provider="qdrant",
            dataset="gold-standard",
            metrics={"precision@k": 0.85},
            latency={"p50_ms": 12.5},
        )

        assert result.provider == "qdrant"
        assert result.dataset == "gold-standard"
        assert result.metrics == {"precision@k": 0.85}
        assert result.latency == {"p50_ms": 12.5}

    def test_timestamp_defaults_to_utc_now(self):
        """timestamp defaults to current UTC time."""
        before = datetime.now(timezone.utc)
        result = BenchmarkResult(
            provider="qdrant",
            dataset="ds",
            metrics={},
            latency={},
        )
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after
        assert result.timestamp.tzinfo is not None

    def test_config_defaults_to_empty_dict(self):
        """config defaults to an empty dict."""
        result = BenchmarkResult(
            provider="qdrant",
            dataset="ds",
            metrics={},
            latency={},
        )

        assert result.config == {}

    def test_custom_config(self):
        """config can be set to a custom dict."""
        result = BenchmarkResult(
            provider="qdrant",
            dataset="ds",
            metrics={},
            latency={},
            config={"k": 10, "warmup": 5},
        )

        assert result.config == {"k": 10, "warmup": 5}

    def test_custom_timestamp(self):
        """timestamp can be set explicitly."""
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = BenchmarkResult(
            provider="qdrant",
            dataset="ds",
            metrics={},
            latency={},
            timestamp=ts,
        )

        assert result.timestamp == ts

    def test_multiple_metrics(self):
        """BenchmarkResult holds multiple metrics."""
        result = BenchmarkResult(
            provider="pgvector",
            dataset="gold",
            metrics={
                "precision@k": 0.85,
                "recall@k": 0.72,
                "ndcg": 0.91,
            },
            latency={"p50_ms": 10.0, "p95_ms": 45.0, "qps": 100.0},
        )

        assert len(result.metrics) == 3
        assert result.metrics["ndcg"] == 0.91
        assert result.latency["qps"] == 100.0


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner ABC."""

    def test_runner_is_abstract(self):
        """BenchmarkRunner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BenchmarkRunner()

    def test_incomplete_runner_raises(self):
        """A runner missing run() cannot be instantiated."""

        class IncompleteRunner(BenchmarkRunner):
            pass

        with pytest.raises(TypeError):
            IncompleteRunner()

    def test_concrete_runner_can_be_instantiated(self):
        """A concrete runner with run() can be instantiated."""

        class ConcreteRunner(BenchmarkRunner):
            async def run(self, provider, dataset, metrics, *, limit=10, warmup_queries=5):
                return BenchmarkResult(
                    provider="test",
                    dataset="test",
                    metrics={},
                    latency={},
                )

        runner = ConcreteRunner()
        assert runner is not None

    @pytest.mark.asyncio
    async def test_run_returns_benchmark_result(self):
        """run() returns a BenchmarkResult."""

        class StubRunner(BenchmarkRunner):
            async def run(self, provider, dataset, metrics, *, limit=10, warmup_queries=5):
                return BenchmarkResult(
                    provider="stub",
                    dataset="test-ds",
                    metrics={"precision@k": 0.5},
                    latency={"p50_ms": 10.0},
                )

        runner = StubRunner()
        result = await runner.run(
            provider=None,  # type: ignore[arg-type]
            dataset=None,  # type: ignore[arg-type]
            metrics=[],
        )

        assert isinstance(result, BenchmarkResult)
        assert result.provider == "stub"
        assert result.metrics["precision@k"] == 0.5

    @pytest.mark.asyncio
    async def test_run_accepts_limit_and_warmup(self):
        """run() accepts limit and warmup_queries keyword arguments."""

        class KwargsRunner(BenchmarkRunner):
            async def run(self, provider, dataset, metrics, *, limit=10, warmup_queries=5):
                return BenchmarkResult(
                    provider="test",
                    dataset="test",
                    metrics={},
                    latency={},
                    config={"limit": limit, "warmup_queries": warmup_queries},
                )

        runner = KwargsRunner()
        result = await runner.run(
            provider=None,  # type: ignore[arg-type]
            dataset=None,  # type: ignore[arg-type]
            metrics=[],
            limit=20,
            warmup_queries=3,
        )

        assert result.config["limit"] == 20
        assert result.config["warmup_queries"] == 3
