"""Base classes for benchmark metrics.

This module provides the foundational abstractions for benchmark metrics:
- MetricRegistry: A metaclass that auto-registers metric subclasses
- Metric: An abstract base class defining the metric interface

Example:
    ```python
    class PrecisionAtK(Metric):
        name = "precision@k"
        description = "Fraction of top-K results that are relevant"

        def __init__(self, k: int = 10):
            self.k = k

        def compute(self, retrieved, relevant, graded=None):
            top_k = retrieved[:self.k]
            return len(set(top_k) & relevant) / self.k if self.k > 0 else 0.0

    # Auto-registered, retrieve by name
    metric_cls = MetricRegistry.get("precision@k")
    ```
"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import ClassVar


class MetricRegistry(ABCMeta):
    """Metaclass that auto-registers metric subclasses.

    Any class using this metaclass that defines a `name` class variable
    will be automatically registered in the internal registry.

    Class Methods:
        get: Retrieve a metric class by name.
        all_metrics: List all registered metric names.
    """

    _registry: ClassVar[dict[str, type]] = {}

    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        """Create a new class and register it if it has a `name` attribute."""
        cls = super().__new__(mcs, name, bases, namespace)

        # Register if the class has a 'name' class variable (not inherited abstract)
        if "name" in namespace and namespace["name"] is not None:
            metric_name = namespace["name"]
            mcs._registry[metric_name] = cls

        return cls

    @classmethod
    def get(cls, metric_name: str) -> type | None:
        """Retrieve a metric class by name.

        Args:
            metric_name: The registered name of the metric.

        Returns:
            The metric class, or None if not found.
        """
        return cls._registry.get(metric_name)

    @classmethod
    def all_metrics(cls) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of registered metric names.
        """
        return sorted(cls._registry.keys())


class Metric(ABC, metaclass=MetricRegistry):
    """Abstract base class for benchmark metrics.

    Subclasses must define:
        - name: A unique identifier for the metric (used for registration)
        - description: A human-readable description of the metric
        - compute(): Method to calculate the metric value

    Attributes:
        name: Unique metric identifier (class variable).
        description: Human-readable description (class variable).
    """

    name: str | None = None
    description: str = ""

    @abstractmethod
    def compute(
        self,
        retrieved: Sequence[str],
        relevant: set[str],
        graded: dict[str, int] | None = None,
    ) -> float:
        """Compute the metric value.

        Args:
            retrieved: Ordered sequence of retrieved document IDs.
            relevant: Set of relevant document IDs.
            graded: Optional dict mapping document IDs to relevance grades.
                   Higher grades indicate more relevant documents.

        Returns:
            The computed metric value as a float.
        """
        ...
