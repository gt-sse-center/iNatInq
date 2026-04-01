"""Unit tests for JSONReporter.

Tests JSON schema compliance, metric serialization, and file output for
benchmark results. This is important because JSON output enables CI/CD
integration and historical metric tracking.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from core.benchmark.reporters.json_reporter import JSONReporter
from core.benchmark.runner.base import BenchmarkResult


def _make_result(
    provider: str = "TestProvider",
    timestamp: datetime | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        provider=provider,
        dataset="test-ds",
        metrics={"precision@k": 0.85, "recall@k": 0.72},
        latency={"p50_ms": 12.5, "p95_ms": 45.0},
        timestamp=timestamp or datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
        config={"limit": 10},
    )


class TestJSONReporter:
    """Tests for JSONReporter."""

    @pytest.mark.asyncio
    async def test_writes_valid_json(self, tmp_path):
        """Output file contains valid JSON."""
        output = tmp_path / "results.json"
        reporter = JSONReporter(output_path=output)

        await reporter.report({"qdrant": _make_result()})

        data = json.loads(output.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_all_fields_present(self, tmp_path):
        """Output contains all expected fields per provider."""
        output = tmp_path / "results.json"
        reporter = JSONReporter(output_path=output)

        await reporter.report({"qdrant": _make_result()})

        data = json.loads(output.read_text(encoding="utf-8"))
        entry = data["qdrant"]
        assert entry["provider"] == "TestProvider"
        assert entry["dataset"] == "test-ds"
        assert entry["metrics"]["precision@k"] == 0.85
        assert entry["latency"]["p50_ms"] == 12.5
        assert entry["config"]["limit"] == 10
        assert "timestamp" in entry

    @pytest.mark.asyncio
    async def test_datetime_iso_format(self, tmp_path):
        """Datetime values are serialized as ISO 8601 strings."""
        output = tmp_path / "results.json"
        reporter = JSONReporter(output_path=output)
        ts = datetime(2026, 6, 15, 14, 30, 0, tzinfo=UTC)

        await reporter.report({"qdrant": _make_result(timestamp=ts)})

        data = json.loads(output.read_text(encoding="utf-8"))
        timestamp_str = data["qdrant"]["timestamp"]
        # Should be parseable back to datetime
        parsed = datetime.fromisoformat(timestamp_str)
        assert parsed == ts

    @pytest.mark.asyncio
    async def test_multiple_providers(self, tmp_path):
        """Output contains entries for multiple providers."""
        output = tmp_path / "results.json"
        reporter = JSONReporter(output_path=output)

        results = {
            "qdrant": _make_result(provider="qdrant"),
            "pgvector": _make_result(provider="pgvector"),
        }
        await reporter.report(results)

        data = json.loads(output.read_text(encoding="utf-8"))
        assert "qdrant" in data
        assert "pgvector" in data

    @pytest.mark.asyncio
    async def test_empty_results(self, tmp_path):
        """Output is valid JSON even with no results."""
        output = tmp_path / "results.json"
        reporter = JSONReporter(output_path=output)

        await reporter.report({})

        data = json.loads(output.read_text(encoding="utf-8"))
        assert data == {}

    @pytest.mark.asyncio
    async def test_output_file_created(self, tmp_path):
        """Output file is created at the specified path."""
        output = tmp_path / "subdir" / "results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        reporter = JSONReporter(output_path=output)

        await reporter.report({"qdrant": _make_result()})

        assert output.exists()
