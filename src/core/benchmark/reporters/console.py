"""Console reporter for benchmark results.

Prints a plain-text table of metrics and latency per provider.

Example:
    ```python
    from core.benchmark.reporters.console import ConsoleReporter

    reporter = ConsoleReporter()
    await reporter.report({"qdrant": result_a, "weaviate": result_b})
    ```
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from core.benchmark.reporters.base import Reporter

if TYPE_CHECKING:
    from io import TextIOBase

    from core.benchmark.runner.base import BenchmarkResult


class ConsoleReporter(Reporter):
    """Reports benchmark results as a plain-text table to a stream.

    Attributes:
        _stream: Output stream (defaults to sys.stdout).
    """

    def __init__(self, stream: TextIOBase | None = None) -> None:
        self._stream = stream or sys.stdout

    async def report(self, results: dict[str, BenchmarkResult]) -> None:
        """Print a plain-text comparison table.

        Args:
            results: Mapping of provider name to BenchmarkResult.
        """
        if not results:
            self._stream.write("No results to report.\n")
            return

        providers = list(results.keys())

        # Collect all metric and latency keys across providers
        metric_keys: list[str] = []
        latency_keys: list[str] = []
        for result in results.values():
            for k in result.metrics:
                if k not in metric_keys:
                    metric_keys.append(k)
            for k in result.latency:
                if k not in latency_keys:
                    latency_keys.append(k)

        # Build rows: label + values per provider
        rows: list[tuple[str, list[str]]] = []
        for key in metric_keys:
            values = []
            for provider in providers:
                val = results[provider].metrics.get(key)
                values.append(f"{val:.4f}" if val is not None else "N/A")
            rows.append((key, values))

        for key in latency_keys:
            values = []
            for provider in providers:
                val = results[provider].latency.get(key)
                values.append(f"{val:.2f}" if val is not None else "N/A")
            rows.append((key, values))

        # Calculate column widths
        label_width = max(len(r[0]) for r in rows) if rows else 10
        label_width = max(label_width, 10)
        col_width = max((len(p) for p in providers), default=10)
        col_width = max(col_width, 10)

        # Print header
        header = f"{'Metric':<{label_width}}"
        for provider in providers:
            header += f"  {provider:>{col_width}}"
        self._stream.write(header + "\n")
        self._stream.write("-" * len(header) + "\n")

        # Print rows
        for label, values in rows:
            line = f"{label:<{label_width}}"
            for val in values:
                line += f"  {val:>{col_width}}"
            self._stream.write(line + "\n")
