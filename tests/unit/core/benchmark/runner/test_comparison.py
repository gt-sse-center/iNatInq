"""Unit tests for ComparisonRunner."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from core.benchmark.datasets.base import Dataset, Query
from core.benchmark.metrics.base import Metric
from core.benchmark.reporters.base import Reporter
from core.benchmark.runner.base import BenchmarkResult
from core.benchmark.runner.comparison import DEFAULT_METRICS, ComparisonRunner
from core.models import SearchResultItem, SearchResults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubComparisonDataset(Dataset):
    """In-memory dataset for comparison runner testing."""

    def __init__(self, queries: list[Query]) -> None:
        self._queries = queries

    @property
    def name(self) -> str:
        return "comparison-test-dataset"

    @property
    def modality(self) -> Literal["text", "image"]:
        return "text"

    def queries(self) -> Iterator[Query]:
        return iter(self._queries)

    def __len__(self) -> int:
        return len(self._queries)


class StubComparisonMetric(Metric):
    """Metric that returns a fixed score for comparison testing."""

    name = "stub_comparison_metric"
    description = "Stub for comparison testing"

    def __init__(self, fixed_score: float = 0.7) -> None:
        self.fixed_score = fixed_score

    def compute(self, retrieved, relevant, graded=None) -> float:
        return self.fixed_score


def _make_search_results(*point_ids: str) -> SearchResults:
    items = [
        SearchResultItem(point_id=pid, score=1.0 - i * 0.1, payload={}) for i, pid in enumerate(point_ids)
    ]
    return SearchResults(items=items, total=len(items))


@pytest.fixture()
def sample_queries() -> list[Query]:
    return [
        Query(id="q1", text="red bird", relevant={"d1", "d3"}),
        Query(id="q2", text="blue fish", relevant={"d2"}),
    ]


@pytest.fixture()
def dataset(sample_queries: list[Query]) -> StubComparisonDataset:
    return StubComparisonDataset(sample_queries)


@pytest.fixture()
def mock_provider_a() -> AsyncMock:
    provider = AsyncMock()
    provider.__class__.__name__ = "ProviderA"
    provider.search_async.return_value = _make_search_results("d1", "d2", "d3")
    return provider


@pytest.fixture()
def mock_provider_b() -> AsyncMock:
    provider = AsyncMock()
    provider.__class__.__name__ = "ProviderB"
    provider.search_async.return_value = _make_search_results("d2", "d3", "d4")
    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComparisonRunner:
    """Tests for ComparisonRunner."""

    @pytest.mark.asyncio
    async def test_returns_results_for_all_providers(
        self, mock_provider_a: AsyncMock, mock_provider_b: AsyncMock, dataset: StubComparisonDataset
    ):
        """compare() returns a result for each provider."""
        runner = ComparisonRunner(
            providers=[mock_provider_a, mock_provider_b],
        )
        results = await runner.compare(dataset, metrics=[StubComparisonMetric()], warmup_queries=0)

        assert len(results) == 2
        assert "ProviderA" in results
        assert "ProviderB" in results

    @pytest.mark.asyncio
    async def test_each_result_is_benchmark_result(
        self, mock_provider_a: AsyncMock, dataset: StubComparisonDataset
    ):
        """Each result is a BenchmarkResult instance."""
        runner = ComparisonRunner(providers=[mock_provider_a])
        results = await runner.compare(dataset, metrics=[StubComparisonMetric()], warmup_queries=0)

        for result in results.values():
            assert isinstance(result, BenchmarkResult)

    @pytest.mark.asyncio
    async def test_uses_default_metrics_when_none(
        self, mock_provider_a: AsyncMock, dataset: StubComparisonDataset
    ):
        """compare() uses DEFAULT_METRICS when metrics is None."""
        runner = ComparisonRunner(providers=[mock_provider_a])
        results = await runner.compare(dataset, warmup_queries=0)

        result = results["ProviderA"]
        # All default metric names should appear in the result
        for metric in DEFAULT_METRICS:
            assert metric.name in result.metrics

    @pytest.mark.asyncio
    async def test_default_metrics_contains_expected(self):
        """DEFAULT_METRICS contains P@K, R@K, MAP, NDCG, MRR."""
        names = {m.name for m in DEFAULT_METRICS}
        assert names == {"precision@k", "recall@k", "map", "ndcg", "mrr"}

    @pytest.mark.asyncio
    async def test_reporters_called_with_results(
        self, mock_provider_a: AsyncMock, dataset: StubComparisonDataset
    ):
        """Reporters receive the results dict after benchmarking."""
        mock_reporter = AsyncMock(spec=Reporter)
        runner = ComparisonRunner(
            providers=[mock_provider_a],
            reporters=[mock_reporter],
        )
        results = await runner.compare(dataset, metrics=[StubComparisonMetric()], warmup_queries=0)

        mock_reporter.report.assert_awaited_once_with(results)

    @pytest.mark.asyncio
    async def test_multiple_reporters_all_called(
        self, mock_provider_a: AsyncMock, dataset: StubComparisonDataset
    ):
        """All reporters are called when multiple are configured."""
        reporter_1 = AsyncMock(spec=Reporter)
        reporter_2 = AsyncMock(spec=Reporter)
        runner = ComparisonRunner(
            providers=[mock_provider_a],
            reporters=[reporter_1, reporter_2],
        )
        await runner.compare(dataset, metrics=[StubComparisonMetric()], warmup_queries=0)

        reporter_1.report.assert_awaited_once()
        reporter_2.report.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_reporters_does_not_error(
        self, mock_provider_a: AsyncMock, dataset: StubComparisonDataset
    ):
        """compare() works without any reporters."""
        runner = ComparisonRunner(providers=[mock_provider_a])
        results = await runner.compare(dataset, metrics=[StubComparisonMetric()], warmup_queries=0)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_custom_metrics_used(self, mock_provider_a: AsyncMock, dataset: StubComparisonDataset):
        """Custom metrics are used instead of defaults when provided."""
        runner = ComparisonRunner(providers=[mock_provider_a])
        results = await runner.compare(
            dataset,
            metrics=[StubComparisonMetric(fixed_score=0.9)],
            warmup_queries=0,
        )

        result = results["ProviderA"]
        assert result.metrics["stub_comparison_metric"] == pytest.approx(0.9)
        # Default metrics should NOT be present
        assert "precision@k" not in result.metrics

    @pytest.mark.asyncio
    async def test_limit_and_warmup_passed_through(
        self, mock_provider_a: AsyncMock, dataset: StubComparisonDataset
    ):
        """limit and warmup_queries are passed to the underlying runner."""
        runner = ComparisonRunner(providers=[mock_provider_a])
        results = await runner.compare(
            dataset,
            metrics=[StubComparisonMetric()],
            limit=20,
            warmup_queries=3,
        )

        result = results["ProviderA"]
        assert result.config["limit"] == 20
        assert result.config["warmup_queries"] == 3
