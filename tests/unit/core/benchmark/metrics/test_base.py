"""Unit tests for benchmark metrics base classes."""

import pytest
from collections.abc import Sequence

from core.benchmark.metrics.base import Metric, MetricRegistry


class TestMetricRegistry:
    """Tests for MetricRegistry metaclass."""

    def test_registry_exists(self):
        """MetricRegistry has an internal _registry dict."""
        assert hasattr(MetricRegistry, "_registry")
        assert isinstance(MetricRegistry._registry, dict)

    def test_auto_registers_subclass_with_name(self):
        """Subclasses with a `name` attribute are auto-registered."""

        class TestMetricA(Metric):
            name = "test_metric_a"
            description = "A test metric"

            def compute(self, retrieved, relevant, graded=None):
                return 1.0

        assert "test_metric_a" in MetricRegistry._registry
        assert MetricRegistry._registry["test_metric_a"] is TestMetricA

    def test_does_not_register_without_name(self):
        """Subclasses without a `name` attribute are not registered."""
        initial_count = len(MetricRegistry._registry)

        class TestMetricNoName(Metric):
            description = "No name metric"

            def compute(self, retrieved, relevant, graded=None):
                return 0.0

        # Should not add to registry (name is None from parent)
        assert len(MetricRegistry._registry) == initial_count

    def test_get_returns_registered_metric(self):
        """MetricRegistry.get() returns the metric class by name."""

        class TestMetricB(Metric):
            name = "test_metric_b"
            description = "Another test metric"

            def compute(self, retrieved, relevant, graded=None):
                return 0.5

        result = MetricRegistry.get("test_metric_b")
        assert result is TestMetricB

    def test_get_returns_none_for_unknown(self):
        """MetricRegistry.get() returns None for unknown metric names."""
        result = MetricRegistry.get("nonexistent_metric_xyz")
        assert result is None

    def test_all_metrics_returns_sorted_names(self):
        """MetricRegistry.all_metrics() returns sorted list of names."""

        class ZetaMetric(Metric):
            name = "zeta_metric"
            description = "Zeta"

            def compute(self, retrieved, relevant, graded=None):
                return 0.0

        class AlphaMetric(Metric):
            name = "alpha_metric"
            description = "Alpha"

            def compute(self, retrieved, relevant, graded=None):
                return 0.0

        metrics = MetricRegistry.all_metrics()
        assert isinstance(metrics, list)
        assert metrics == sorted(metrics)
        assert "alpha_metric" in metrics
        assert "zeta_metric" in metrics


class TestMetric:
    """Tests for Metric ABC."""

    def test_metric_is_abstract(self):
        """Metric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Metric()

    def test_metric_has_name_class_variable(self):
        """Metric defines a name class variable."""
        assert hasattr(Metric, "name")
        assert Metric.name == ""

    def test_metric_has_description_class_variable(self):
        """Metric defines a description class variable."""
        assert hasattr(Metric, "description")
        assert Metric.description == ""

    def test_compute_is_abstract(self):
        """compute() is an abstract method that must be implemented."""

        class IncompleteMetric(Metric):
            name = "incomplete"
            description = "Missing compute"

        with pytest.raises(TypeError):
            IncompleteMetric()

    def test_concrete_metric_can_be_instantiated(self):
        """A concrete metric with compute() can be instantiated."""

        class ConcreteMetric(Metric):
            name = "concrete_metric"
            description = "A concrete metric"

            def compute(
                self,
                retrieved: Sequence[str],
                relevant: set[str],
                graded: dict[str, int] | None = None,
            ) -> float:
                return 0.75

        metric = ConcreteMetric()
        assert metric is not None

    def test_compute_signature(self):
        """compute() accepts the correct arguments."""

        class SignatureMetric(Metric):
            name = "signature_metric"
            description = "Tests signature"

            def compute(self, retrieved, relevant, graded=None):
                # Verify we can access arguments correctly
                assert isinstance(retrieved, Sequence)
                assert isinstance(relevant, set)
                return len(set(retrieved) & relevant) / len(retrieved) if retrieved else 0.0

        metric = SignatureMetric()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3", "doc5"}

        result = metric.compute(retrieved, relevant)
        assert result == pytest.approx(2 / 3)

    def test_compute_with_graded_relevance(self):
        """compute() can use graded relevance scores."""

        class GradedMetric(Metric):
            name = "graded_metric"
            description = "Uses graded relevance"

            def compute(self, retrieved, relevant, graded=None):
                if graded is None:
                    return 0.0
                total = sum(graded.get(doc, 0) for doc in retrieved)
                return float(total)

        metric = GradedMetric()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        graded = {"doc1": 3, "doc2": 2, "doc3": 0}

        result = metric.compute(retrieved, relevant, graded)
        assert result == 5.0
