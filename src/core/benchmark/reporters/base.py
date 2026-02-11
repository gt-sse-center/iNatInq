"""Base class for benchmark reporters.

This module provides the abstract Reporter interface that all benchmark
reporters must implement.

Example:
    ```python
    from core.benchmark.reporters.base import Reporter

    class MyReporter(Reporter):
        async def report(self, results):
            for provider, result in results.items():
                print(f"{provider}: {result.metrics}")
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.benchmark.runner.base import BenchmarkResult


class Reporter(ABC):
    """Abstract base class for benchmark reporters.

    Subclasses must implement the `report()` method to present
    benchmark results in a specific format (console, JSON, etc.).
    """

    @abstractmethod
    async def report(self, results: dict[str, BenchmarkResult]) -> None:
        """Report benchmark results.

        Args:
            results: Mapping of provider name to BenchmarkResult.
        """
        ...
