"""Unit tests for DefaultBenchmarkRunner with search pipeline integration.

Tests end-to-end benchmark execution with real SearchPipeline integration,
result caching, and metric aggregation. This is important because the default
runner provides the primary workflow for evaluating search quality.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.benchmark.datasets.base import Dataset, Query
from core.benchmark.id_mapping import S3KeyIDMapper
from core.benchmark.metrics.base import Metric
from core.benchmark.runner.base import BenchmarkResult
from core.benchmark.runner.default import DefaultBenchmarkRunner
from core.benchmark.search_pipeline import SearchPipeline, SearchPipelineResult
from core.models import SearchResultItem, SearchResults


class StubDataset(Dataset):
    """In-memory dataset for testing."""

    def __init__(self, queries: list[Query], modality: str = "image") -> None:
        self._queries = queries
        self._modality = modality

    @property
    def name(self) -> str:
        return "test-dataset"

    @property
    def modality(self) -> Literal["text", "image"]:
        return self._modality  # type: ignore[return-value]

    def queries(self) -> Iterator[Query]:
        return iter(self._queries)

    def __len__(self) -> int:
        return len(self._queries)


class StubMetric(Metric):
    """Metric that records retrieved IDs for assertions."""

    name = "stub_pipeline_metric"
    description = "Stub for pipeline testing"

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], set[str]]] = []

    def compute(self, retrieved, relevant, graded=None) -> float:
        self.calls.append((list(retrieved), set(relevant)))
        return 0.5


def _make_pipeline_result(*doc_ids: str) -> SearchPipelineResult:
    """Build a SearchPipelineResult with given doc IDs."""
    items = [
        SearchResultItem(point_id=f"uuid-{did}", score=1.0 - i * 0.1, payload={"s3_key": did})
        for i, did in enumerate(doc_ids)
    ]
    return SearchPipelineResult(
        doc_ids=list(doc_ids),
        raw_results=SearchResults(items=items, total=len(items)),
    )


@pytest.fixture()
def sample_queries() -> list[Query]:
    return [
        Query(id="q1", text="red bird", relevant={"100", "200"}),
        Query(id="q2", text="blue fish", relevant={"300"}),
    ]


@pytest.fixture()
def dataset(sample_queries: list[Query]) -> StubDataset:
    return StubDataset(sample_queries)


class TestDefaultBenchmarkRunnerWithPipeline:
    """Tests for DefaultBenchmarkRunner with search_pipeline set."""

    @pytest.mark.asyncio
    async def test_uses_pipeline_doc_ids_for_metrics(self, dataset: StubDataset):
        """When search_pipeline is set, metrics receive doc_ids from pipeline."""
        mock_pipeline = AsyncMock(spec=SearchPipeline)
        mock_pipeline.search.return_value = _make_pipeline_result("100", "300")

        spy_metric = StubMetric()
        mock_provider = AsyncMock()

        runner = DefaultBenchmarkRunner(
            search_pipeline=mock_pipeline,
            collection="test-collection",
        )
        await runner.run(
            provider=mock_provider,
            dataset=dataset,
            metrics=[spy_metric],
            limit=10,
            warmup_queries=0,
        )

        # Metrics should have received doc IDs from pipeline, not point IDs
        assert len(spy_metric.calls) == 2
        for retrieved, _ in spy_metric.calls:
            assert retrieved == ["100", "300"]

    @pytest.mark.asyncio
    async def test_pipeline_uses_collection_override(self, dataset: StubDataset):
        """Pipeline is called with the collection override, not dataset.name."""
        mock_pipeline = AsyncMock(spec=SearchPipeline)
        mock_pipeline.search.return_value = _make_pipeline_result("100")

        runner = DefaultBenchmarkRunner(
            search_pipeline=mock_pipeline,
            collection="my-override-collection",
        )
        await runner.run(
            provider=AsyncMock(),
            dataset=dataset,
            metrics=[StubMetric()],
            limit=5,
            warmup_queries=0,
        )

        # Verify collection name passed to pipeline
        for call in mock_pipeline.search.call_args_list:
            assert call.kwargs["collection"] == "my-override-collection"

    @pytest.mark.asyncio
    async def test_pipeline_uses_dataset_name_without_collection_override(self, dataset: StubDataset):
        """Without collection override, pipeline uses dataset.name."""
        mock_pipeline = AsyncMock(spec=SearchPipeline)
        mock_pipeline.search.return_value = _make_pipeline_result("100")

        runner = DefaultBenchmarkRunner(search_pipeline=mock_pipeline)
        await runner.run(
            provider=AsyncMock(),
            dataset=dataset,
            metrics=[StubMetric()],
            limit=5,
            warmup_queries=0,
        )

        for call in mock_pipeline.search.call_args_list:
            assert call.kwargs["collection"] == "test-dataset"

    @pytest.mark.asyncio
    async def test_without_pipeline_uses_provider_directly(self):
        """Without search_pipeline, runner falls back to provider.search_async."""
        mock_provider = AsyncMock()
        mock_provider.search_async.return_value = SearchResults(
            items=[SearchResultItem(point_id="p1", score=0.9, payload={})],
            total=1,
        )

        queries = [Query(id="q1", text="test", relevant={"p1"})]
        ds = StubDataset(queries)
        spy_metric = StubMetric()

        runner = DefaultBenchmarkRunner()
        await runner.run(
            provider=mock_provider,
            dataset=ds,
            metrics=[spy_metric],
            limit=10,
            warmup_queries=0,
        )

        # Should have called provider directly
        mock_provider.search_async.assert_called()
        # Metrics receive raw point_ids
        assert spy_metric.calls[0][0] == ["p1"]

    @pytest.mark.asyncio
    async def test_pipeline_result_returns_benchmark_result(self, dataset: StubDataset):
        """Runner still returns a proper BenchmarkResult with pipeline."""
        mock_pipeline = AsyncMock(spec=SearchPipeline)
        mock_pipeline.search.return_value = _make_pipeline_result("100")

        runner = DefaultBenchmarkRunner(
            search_pipeline=mock_pipeline,
            collection="col",
        )
        result = await runner.run(
            provider=AsyncMock(),
            dataset=dataset,
            metrics=[StubMetric()],
            limit=10,
            warmup_queries=0,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.dataset == "test-dataset"
        assert "stub_pipeline_metric" in result.metrics
        assert result.latency["count"] == 2
