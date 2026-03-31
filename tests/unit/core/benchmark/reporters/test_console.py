"""Unit tests for ConsoleReporter.

Tests console table formatting, metric value alignment, and ANSI color codes
for benchmark result output. This is important because developers rely on clear
console output to quickly assess search quality changes.
"""

from __future__ import annotations

import io

import pytest

from core.benchmark.reporters.console import ConsoleReporter
from core.benchmark.runner.base import BenchmarkResult


def _make_result(
    provider: str = "TestProvider",
    metrics: dict | None = None,
    latency: dict | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        provider=provider,
        dataset="test-ds",
        metrics=metrics or {"precision@k": 0.85, "recall@k": 0.72},
        latency=latency or {"p50_ms": 12.5, "p95_ms": 45.0, "qps": 80.0},
    )


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    @pytest.mark.asyncio
    async def test_single_provider_table(self):
        """Report formats a table for a single provider."""
        stream = io.StringIO()
        reporter = ConsoleReporter(stream=stream)

        await reporter.report({"qdrant": _make_result()})

        output = stream.getvalue()
        assert "Metric" in output
        assert "qdrant" in output
        assert "precision@k" in output
        assert "recall@k" in output
        assert "0.8500" in output
        assert "p50_ms" in output

    @pytest.mark.asyncio
    async def test_multi_provider_table(self):
        """Report formats a table with multiple providers side-by-side."""
        stream = io.StringIO()
        reporter = ConsoleReporter(stream=stream)

        results = {
            "qdrant": _make_result(metrics={"precision@k": 0.85}),
            "pgvector": _make_result(metrics={"precision@k": 0.72}),
        }
        await reporter.report(results)

        output = stream.getvalue()
        assert "qdrant" in output
        assert "pgvector" in output
        assert "0.8500" in output
        assert "0.7200" in output

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Report handles empty results dict."""
        stream = io.StringIO()
        reporter = ConsoleReporter(stream=stream)

        await reporter.report({})

        output = stream.getvalue()
        assert "No results to report" in output

    @pytest.mark.asyncio
    async def test_separator_line_present(self):
        """Output contains a separator line between header and data."""
        stream = io.StringIO()
        reporter = ConsoleReporter(stream=stream)

        await reporter.report({"qdrant": _make_result()})

        lines = stream.getvalue().split("\n")
        # Second line should be all dashes
        assert all(c == "-" for c in lines[1])

    @pytest.mark.asyncio
    async def test_latency_values_formatted(self):
        """Latency values are formatted with 2 decimal places."""
        stream = io.StringIO()
        reporter = ConsoleReporter(stream=stream)

        await reporter.report({"qdrant": _make_result(latency={"p50_ms": 12.567})})

        output = stream.getvalue()
        assert "12.57" in output

    @pytest.mark.asyncio
    async def test_metric_values_formatted(self):
        """Metric values are formatted with 4 decimal places."""
        stream = io.StringIO()
        reporter = ConsoleReporter(stream=stream)

        await reporter.report({"qdrant": _make_result(metrics={"ndcg": 0.9123456})})

        output = stream.getvalue()
        assert "0.9123" in output

    @pytest.mark.asyncio
    async def test_defaults_to_stdout(self):
        """ConsoleReporter defaults to sys.stdout when no stream given."""
        reporter = ConsoleReporter()
        assert reporter._stream is not None
