"""JSON reporter for benchmark results.

Writes benchmark results to a JSON file with ISO 8601 datetime serialization.

Example:
    ```python
    from pathlib import Path
    from core.benchmark.reporters.json_reporter import JSONReporter

    reporter = JSONReporter(output_path=Path("results.json"))
    await reporter.report({"qdrant": result})
    ```
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from core.benchmark.reporters.base import Reporter
from typing_extensions import override

if TYPE_CHECKING:
    from pathlib import Path

    from core.benchmark.runner.base import BenchmarkResult


class _BenchmarkEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""

    @override
    def default(self, o: object) -> object:
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class JSONReporter(Reporter):
    """Reports benchmark results as a JSON file.

    Attributes:
        _output_path: Path to the output JSON file.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    @override
    async def report(self, results: dict[str, BenchmarkResult]) -> None:
        """Write benchmark results to a JSON file.

        Args:
            results: Mapping of provider name to BenchmarkResult.
        """
        data = {}
        for provider_name, result in results.items():
            data[provider_name] = {
                "provider": result.provider,
                "dataset": result.dataset,
                "metrics": result.metrics,
                "latency": result.latency,
                "timestamp": result.timestamp,
                "config": result.config,
            }

        self._output_path.write_text(
            json.dumps(data, cls=_BenchmarkEncoder, indent=2) + "\n",
            encoding="utf-8",
        )
