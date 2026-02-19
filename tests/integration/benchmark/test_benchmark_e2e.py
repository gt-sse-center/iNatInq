"""End-to-end integration test for the benchmark framework.

Tests the full pipeline: Dataset -> ComparisonRunner -> Reporters -> Results
using mock providers and synthetic data, plus integration tests against real
INQUIRE benchmark datasets and fixture images.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from clients.s3 import S3ClientWrapper
from core.benchmark.cli import app as benchmark_app
from core.benchmark.datasets.base import Dataset, Query
from core.benchmark.datasets.json_dataset import JSONDataset
from core.benchmark.metrics.base import Metric
from core.benchmark.reporters.console import ConsoleReporter
from core.benchmark.reporters.json_reporter import JSONReporter
from core.benchmark.runner.base import BenchmarkResult
from core.benchmark.runner.comparison import ComparisonRunner
from core.models import SearchResultItem, SearchResults

from tests.integration.benchmark.conftest import (
    INQUIRE_FIXTURES_DIR,
    INQUIRE_VAL_PATH,
    SAMPLE_GOLD_PATH,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

QUERIES = [
    Query(
        id="q1",
        text="red cardinal bird",
        relevant={"d1", "d2", "d3"},
        graded_relevance={"d1": 3, "d2": 2, "d3": 1},
    ),
    Query(
        id="q2",
        text="monarch butterfly on milkweed",
        relevant={"d4", "d5"},
        graded_relevance={"d4": 3, "d5": 2},
    ),
    Query(
        id="q3",
        text="eastern box turtle",
        relevant={"d6", "d7", "d8"},
        graded_relevance={"d6": 3, "d7": 2, "d8": 1},
    ),
]


class SyntheticDataset(Dataset):
    """In-memory dataset for integration testing."""

    @property
    def name(self) -> str:
        return "synthetic-e2e"

    @property
    def modality(self) -> Literal["text", "image"]:
        return "text"

    def queries(self) -> Iterator[Query]:
        return iter(QUERIES)

    def __len__(self) -> int:
        return len(QUERIES)


def _make_search_results(point_ids: list[str]) -> SearchResults:
    items = [
        SearchResultItem(point_id=pid, score=1.0 - i * 0.1, payload={}) for i, pid in enumerate(point_ids)
    ]
    return SearchResults(items=items, total=len(items))


def _make_good_provider() -> AsyncMock:
    """Provider that returns mostly relevant results."""
    provider = AsyncMock()
    provider.__class__.__name__ = "GoodProvider"
    # Returns d1, d2, d3 — all relevant for q1, partial for others
    provider.search_async.return_value = _make_search_results(["d1", "d2", "d3", "d4", "d5"])
    return provider


def _make_poor_provider() -> AsyncMock:
    """Provider that returns mostly irrelevant results."""
    provider = AsyncMock()
    provider.__class__.__name__ = "PoorProvider"
    # Returns mostly irrelevant docs
    provider.search_async.return_value = _make_search_results(["d99", "d98", "d1"])
    return provider


class StubE2EMetric(Metric):
    """Simple metric for e2e testing that measures hit rate."""

    name = "stub_e2e_hit_rate"
    description = "Fraction of retrieved docs that are relevant"

    def compute(self, retrieved, relevant, graded=None) -> float:
        if not retrieved:
            return 0.0
        hits = sum(1 for doc in retrieved if doc in relevant)
        return hits / len(retrieved)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestBenchmarkE2E:
    """Full-stack integration tests for the benchmark framework."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_comparison_runner(self):
        """ComparisonRunner produces results for all providers."""
        good = _make_good_provider()
        poor = _make_poor_provider()
        dataset = SyntheticDataset()

        runner = ComparisonRunner(providers=[good, poor])
        results = await runner.compare(
            dataset,
            metrics=[StubE2EMetric()],
            warmup_queries=0,
        )

        assert len(results) == 2
        assert "GoodProvider" in results
        assert "PoorProvider" in results

        for result in results.values():
            assert isinstance(result, BenchmarkResult)
            assert "stub_e2e_hit_rate" in result.metrics
            assert result.dataset == "synthetic-e2e"
            assert "p50_ms" in result.latency

    @pytest.mark.asyncio
    async def test_good_provider_beats_poor_provider(self):
        """Good provider should score higher than poor provider."""
        good = _make_good_provider()
        poor = _make_poor_provider()
        dataset = SyntheticDataset()

        runner = ComparisonRunner(providers=[good, poor])
        results = await runner.compare(
            dataset,
            metrics=[StubE2EMetric()],
            warmup_queries=0,
        )

        good_score = results["GoodProvider"].metrics["stub_e2e_hit_rate"]
        poor_score = results["PoorProvider"].metrics["stub_e2e_hit_rate"]
        assert good_score > poor_score

    @pytest.mark.asyncio
    async def test_console_reporter_e2e(self):
        """Console reporter produces output for comparison results."""
        good = _make_good_provider()
        stream = io.StringIO()
        console = ConsoleReporter(stream=stream)
        dataset = SyntheticDataset()

        runner = ComparisonRunner(
            providers=[good],
            reporters=[console],
        )
        await runner.compare(
            dataset,
            metrics=[StubE2EMetric()],
            warmup_queries=0,
        )

        output = stream.getvalue()
        assert "GoodProvider" in output
        assert "stub_e2e_hit_rate" in output
        assert "p50_ms" in output

    @pytest.mark.asyncio
    async def test_json_reporter_e2e(self, tmp_path):
        """JSON reporter writes valid results file."""
        good = _make_good_provider()
        poor = _make_poor_provider()
        output_path = tmp_path / "results.json"
        json_reporter = JSONReporter(output_path=output_path)
        dataset = SyntheticDataset()

        runner = ComparisonRunner(
            providers=[good, poor],
            reporters=[json_reporter],
        )
        await runner.compare(
            dataset,
            metrics=[StubE2EMetric()],
            warmup_queries=0,
        )

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "GoodProvider" in data
        assert "PoorProvider" in data
        assert data["GoodProvider"]["metrics"]["stub_e2e_hit_rate"] > 0

    @pytest.mark.asyncio
    async def test_both_reporters_e2e(self, tmp_path):
        """Both console and JSON reporters work together."""
        good = _make_good_provider()
        stream = io.StringIO()
        console = ConsoleReporter(stream=stream)
        json_path = tmp_path / "results.json"
        json_reporter = JSONReporter(output_path=json_path)
        dataset = SyntheticDataset()

        runner = ComparisonRunner(
            providers=[good],
            reporters=[console, json_reporter],
        )
        results = await runner.compare(
            dataset,
            metrics=[StubE2EMetric()],
            warmup_queries=0,
        )

        # Console output present
        assert "GoodProvider" in stream.getvalue()

        # JSON file present and valid
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "GoodProvider" in data

        # Results returned
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_latency_stats_populated_e2e(self):
        """Latency statistics are populated in end-to-end run."""
        good = _make_good_provider()
        dataset = SyntheticDataset()

        runner = ComparisonRunner(providers=[good])
        results = await runner.compare(
            dataset,
            metrics=[StubE2EMetric()],
            warmup_queries=0,
        )

        latency = results["GoodProvider"].latency
        assert latency["count"] == 3  # 3 queries
        assert latency["p50_ms"] >= 0
        assert latency["p95_ms"] >= 0
        assert latency["qps"] > 0

    @pytest.mark.asyncio
    async def test_warmup_queries_e2e(self):
        """Warmup queries run without affecting metrics."""
        good = _make_good_provider()
        dataset = SyntheticDataset()

        runner = ComparisonRunner(providers=[good])
        results = await runner.compare(
            dataset,
            metrics=[StubE2EMetric()],
            warmup_queries=2,
        )

        # 2 warmup + 3 measurement = 5 total search calls
        assert good.search_async.call_count == 5
        # But only 3 queries counted in latency
        assert results["GoodProvider"].latency["count"] == 3

    @pytest.mark.asyncio
    async def test_multiple_metrics_e2e(self):
        """Multiple metrics are computed in end-to-end run."""

        class StubE2EPrecision(Metric):
            name = "stub_e2e_precision"
            description = "Stub precision"

            def compute(self, retrieved, relevant, graded=None) -> float:
                return 0.75

        good = _make_good_provider()
        dataset = SyntheticDataset()

        runner = ComparisonRunner(providers=[good])
        results = await runner.compare(
            dataset,
            metrics=[StubE2EMetric(), StubE2EPrecision()],
            warmup_queries=0,
        )

        result = results["GoodProvider"]
        assert "stub_e2e_hit_rate" in result.metrics
        assert "stub_e2e_precision" in result.metrics
        assert result.metrics["stub_e2e_precision"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# CLI tests against real benchmark datasets
# ---------------------------------------------------------------------------


class TestCLIWithRealDatasets:
    """Exercise CLI ``validate`` and ``metrics`` commands against real files."""

    def test_validate_inquire_val(self, cli_runner):
        """CLI validates inquire-val.json and reports dataset name and query count."""
        if not INQUIRE_VAL_PATH.exists():
            pytest.skip("inquire-val.json not found")

        result = cli_runner.invoke(benchmark_app, ["validate", str(INQUIRE_VAL_PATH)])

        assert result.exit_code == 0
        assert "inquire-val" in result.output
        assert "50" in result.output

    def test_validate_inquire_val_modality(self, cli_runner):
        """CLI validate output confirms image modality."""
        if not INQUIRE_VAL_PATH.exists():
            pytest.skip("inquire-val.json not found")

        result = cli_runner.invoke(benchmark_app, ["validate", str(INQUIRE_VAL_PATH)])

        assert result.exit_code == 0
        assert "image" in result.output

    def test_validate_sample_gold(self, cli_runner):
        """CLI validates sample-gold.json successfully."""
        if not SAMPLE_GOLD_PATH.exists():
            pytest.skip("sample-gold.json not found")

        result = cli_runner.invoke(benchmark_app, ["validate", str(SAMPLE_GOLD_PATH)])

        assert result.exit_code == 0
        assert "inat-sample-gold" in result.output

    def test_metrics_command(self, cli_runner):
        """CLI lists all registered IR metrics."""
        result = cli_runner.invoke(benchmark_app, ["metrics"])

        assert result.exit_code == 0
        for metric_name in ("precision@k", "recall@k", "map", "ndcg", "mrr"):
            assert metric_name in result.output


# ---------------------------------------------------------------------------
# INQUIRE dataset integration tests
# ---------------------------------------------------------------------------


class TestInquireDatasetIntegration:
    """Load and inspect real INQUIRE benchmark files."""

    def test_inquire_val_loads(self, inquire_val_dataset: JSONDataset):
        """Dataset name and modality are correct."""
        assert inquire_val_dataset.name == "inquire-val"
        assert inquire_val_dataset.modality == "image"

    def test_inquire_val_query_count(self, inquire_val_dataset: JSONDataset):
        """INQUIRE validation split contains 50 queries."""
        assert len(inquire_val_dataset) == 50

    def test_query_127_present(self, inquire_val_dataset: JSONDataset):
        """Query 127 exists with 52 relevant images and expected text."""
        query_127 = None
        for q in inquire_val_dataset.queries():
            if q.id == "127":
                query_127 = q
                break

        assert query_127 is not None, "Query 127 not found in dataset"
        assert len(query_127.relevant) == 52
        assert "peach-faced Lovebird" in query_127.text

    def test_relevant_ids_are_numeric_strings(self, inquire_val_dataset: JSONDataset):
        """All relevant IDs across all queries are digit strings (iNat image IDs)."""
        for q in inquire_val_dataset.queries():
            for rid in q.relevant:
                assert rid.isdigit(), f"Non-numeric relevant ID '{rid}' in query {q.id}"


# ---------------------------------------------------------------------------
# INQUIRE image fixture tests
# ---------------------------------------------------------------------------


class TestInquireImageFixtures:
    """Validate fixture images on disk and in MinIO testcontainer."""

    def test_fixture_images_exist(self, inquire_images_dir: Path):
        """5 JPEG fixture images exist on disk."""
        jpgs = list(inquire_images_dir.glob("*.jpg"))
        assert len(jpgs) == 5

    def test_fixture_images_are_valid(self, inquire_images_dir: Path):
        """All fixture images are non-empty and start with JPEG magic bytes."""
        jpgs = sorted(inquire_images_dir.glob("*.jpg"))
        for jpg in jpgs:
            data = jpg.read_bytes()
            assert len(data) > 0, f"{jpg.name} is empty"
            assert data[:2] == b"\xff\xd8", f"{jpg.name} missing JPEG magic bytes"

    def test_images_loaded_in_minio(
        self,
        inquire_bucket: str,
        minio_client: S3ClientWrapper,
    ):
        """5 objects are present in the testcontainer MinIO bucket."""
        keys = minio_client.list_objects(bucket=inquire_bucket)
        assert len(keys) == 5

    def test_image_roundtrip(
        self,
        inquire_bucket: str,
        inquire_images_dir: Path,
        minio_client: S3ClientWrapper,
    ):
        """An image fetched from MinIO matches the local fixture byte-for-byte."""
        # Pick the first fixture alphabetically
        local_jpg = sorted(inquire_images_dir.glob("*.jpg"))[0]
        image_id = local_jpg.stem

        remote_bytes = minio_client.get_object(bucket=inquire_bucket, key=image_id)
        local_bytes = local_jpg.read_bytes()

        assert remote_bytes == local_bytes

    def test_fixture_images_are_relevant_to_query_127(
        self,
        inquire_bucket: str,
        inquire_val_dataset: JSONDataset,
        minio_client: S3ClientWrapper,
    ):
        """All fixture image IDs in MinIO are a subset of query 127's relevant set."""
        query_127 = None
        for q in inquire_val_dataset.queries():
            if q.id == "127":
                query_127 = q
                break

        assert query_127 is not None

        keys = set(minio_client.list_objects(bucket=inquire_bucket))
        assert len(keys) == 5
        assert keys <= query_127.relevant, f"Unexpected IDs: {keys - query_127.relevant}"
