"""Unit tests for Reporter ABC."""

from __future__ import annotations

import pytest

from core.benchmark.reporters.base import Reporter
from core.benchmark.runner.base import BenchmarkResult


class TestReporter:
    """Tests for Reporter abstract base class."""

    def test_reporter_is_abstract(self):
        """Reporter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Reporter()

    def test_incomplete_reporter_raises(self):
        """A reporter missing report() cannot be instantiated."""

        class IncompleteReporter(Reporter):
            pass

        with pytest.raises(TypeError):
            IncompleteReporter()

    def test_concrete_reporter_can_be_instantiated(self):
        """A concrete reporter with report() can be instantiated."""

        class ConcreteReporter(Reporter):
            async def report(self, results):
                pass

        reporter = ConcreteReporter()
        assert reporter is not None

    @pytest.mark.asyncio
    async def test_report_receives_results(self):
        """report() receives a dict of provider name to BenchmarkResult."""
        received = {}

        class CapturingReporter(Reporter):
            async def report(self, results):
                received.update(results)

        reporter = CapturingReporter()
        result = BenchmarkResult(
            provider="test-provider",
            dataset="test-ds",
            metrics={"precision@k": 0.85},
            latency={"p50_ms": 10.0},
        )
        await reporter.report({"test-provider": result})

        assert "test-provider" in received
        assert received["test-provider"].metrics["precision@k"] == 0.85
