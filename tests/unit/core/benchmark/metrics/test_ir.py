"""Unit tests for core.benchmark.metrics.ir module.

This file tests the Information Retrieval metrics for benchmark evaluation:
- PrecisionAtK: Fraction of top-K results that are relevant
- RecallAtK: Fraction of relevant documents in top-K results

# Test Coverage

The tests cover:
  - Basic computation with known inputs and expected outputs
  - Edge cases: k=0, empty relevant set, empty retrieved list
  - Auto-registration with MetricRegistry
  - Default k value (10)
  - Custom k values

# Running Tests

Run with: uv run pytest tests/unit/core/benchmark/metrics/test_ir.py -v
"""

import pytest

from core.benchmark.metrics.base import MetricRegistry
from core.benchmark.metrics.ir import PrecisionAtK, RecallAtK


class TestPrecisionAtK:
    """Test suite for PrecisionAtK metric."""

    def test_precision_basic_computation(self):
        """Test basic Precision@K calculation with known inputs.

        **Why this test is important:**
          - Validates core precision calculation logic
          - Ensures correct intersection counting
          - Verifies division by k

        **What it tests:**
          - 2 relevant docs in top-5 → precision = 2/5 = 0.4
        """
        precision = PrecisionAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc7"}

        result = precision.compute(retrieved, relevant)

        assert result == pytest.approx(0.4)

    def test_precision_perfect_score(self):
        """Test Precision@K when all top-K are relevant.

        **Why this test is important:**
          - Validates upper bound (1.0) is achievable
          - Ensures no off-by-one errors

        **What it tests:**
          - All 3 retrieved docs are relevant → precision = 1.0
        """
        precision = PrecisionAtK(k=3)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3", "doc4"}

        result = precision.compute(retrieved, relevant)

        assert result == pytest.approx(1.0)

    def test_precision_zero_score(self):
        """Test Precision@K when no top-K are relevant.

        **Why this test is important:**
          - Validates lower bound (0.0) is returned correctly
          - Ensures no false positives in counting

        **What it tests:**
          - No overlap between retrieved and relevant → precision = 0.0
        """
        precision = PrecisionAtK(k=3)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5", "doc6"}

        result = precision.compute(retrieved, relevant)

        assert result == pytest.approx(0.0)

    def test_precision_k_zero_returns_zero(self):
        """Test Precision@K returns 0.0 when k=0.

        **Why this test is important:**
          - Division by zero edge case
          - Must return 0.0 not raise exception

        **What it tests:**
          - k=0 → returns 0.0
        """
        precision = PrecisionAtK(k=0)
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}

        result = precision.compute(retrieved, relevant)

        assert result == 0.0

    def test_precision_k_negative_returns_zero(self):
        """Test Precision@K returns 0.0 when k is negative.

        **Why this test is important:**
          - Invalid k value edge case
          - Must handle gracefully

        **What it tests:**
          - k=-5 → returns 0.0
        """
        precision = PrecisionAtK(k=-5)
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}

        result = precision.compute(retrieved, relevant)

        assert result == 0.0

    def test_precision_default_k_is_ten(self):
        """Test PrecisionAtK uses k=10 by default.

        **Why this test is important:**
          - Validates default parameter value
          - Common use case without specifying k

        **What it tests:**
          - Default k=10 is used when not specified
        """
        precision = PrecisionAtK()

        assert precision.k == 10

    def test_precision_fewer_results_than_k(self):
        """Test Precision@K when retrieved list is shorter than k.

        **Why this test is important:**
          - Retrieved list may be shorter than k
          - Should still compute correctly

        **What it tests:**
          - 2 docs retrieved, k=5, 1 relevant → precision = 1/5 = 0.2
        """
        precision = PrecisionAtK(k=5)
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}

        result = precision.compute(retrieved, relevant)

        # Precision is still divided by k, not len(retrieved)
        assert result == pytest.approx(0.2)

    def test_precision_auto_registers_with_registry(self):
        """Test PrecisionAtK auto-registers with MetricRegistry.

        **Why this test is important:**
          - Metrics must be discoverable by name
          - Enables dynamic metric selection

        **What it tests:**
          - MetricRegistry.get("precision@k") returns PrecisionAtK class
        """
        metric_cls = MetricRegistry.get("precision@k")

        assert metric_cls is not None
        assert metric_cls.__name__ == "PrecisionAtK"
        # Verify it can be instantiated and compute
        instance = metric_cls(k=5)
        assert instance.compute(["a", "b"], {"a"}) == pytest.approx(0.2)

    def test_precision_graded_parameter_ignored(self):
        """Test that graded parameter doesn't affect Precision@K.

        **Why this test is important:**
          - Precision@K is binary, not graded
          - graded param exists for interface compatibility

        **What it tests:**
          - Same result with or without graded parameter
        """
        precision = PrecisionAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}
        graded = {"doc1": 3, "doc3": 2, "doc7": 1}

        result_with_graded = precision.compute(retrieved, relevant, graded)
        result_without_graded = precision.compute(retrieved, relevant)

        assert result_with_graded == result_without_graded


class TestRecallAtK:
    """Test suite for RecallAtK metric."""

    def test_recall_basic_computation(self):
        """Test basic Recall@K calculation with known inputs.

        **Why this test is important:**
          - Validates core recall calculation logic
          - Ensures correct intersection counting
          - Verifies division by relevant set size

        **What it tests:**
          - 2 of 3 relevant docs in top-5 → recall = 2/3 ≈ 0.667
        """
        recall = RecallAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc7"}

        result = recall.compute(retrieved, relevant)

        assert result == pytest.approx(2 / 3)

    def test_recall_perfect_score(self):
        """Test Recall@K when all relevant docs are in top-K.

        **Why this test is important:**
          - Validates upper bound (1.0) is achievable
          - All relevant documents found

        **What it tests:**
          - All 3 relevant docs in top-5 → recall = 1.0
        """
        recall = RecallAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3"}

        result = recall.compute(retrieved, relevant)

        assert result == pytest.approx(1.0)

    def test_recall_zero_score(self):
        """Test Recall@K when no relevant docs in top-K.

        **Why this test is important:**
          - Validates lower bound (0.0) is returned correctly
          - No relevant documents found

        **What it tests:**
          - No overlap → recall = 0.0
        """
        recall = RecallAtK(k=3)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5", "doc6"}

        result = recall.compute(retrieved, relevant)

        assert result == pytest.approx(0.0)

    def test_recall_empty_relevant_returns_zero(self):
        """Test Recall@K returns 0.0 when relevant set is empty.

        **Why this test is important:**
          - Division by zero edge case
          - Must return 0.0 not raise exception

        **What it tests:**
          - Empty relevant set → returns 0.0
        """
        recall = RecallAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: set[str] = set()

        result = recall.compute(retrieved, relevant)

        assert result == 0.0

    def test_recall_k_zero_returns_zero(self):
        """Test Recall@K returns 0.0 when k=0.

        **Why this test is important:**
          - Edge case with k=0
          - Must return 0.0

        **What it tests:**
          - k=0 → returns 0.0
        """
        recall = RecallAtK(k=0)
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}

        result = recall.compute(retrieved, relevant)

        assert result == 0.0

    def test_recall_k_negative_returns_zero(self):
        """Test Recall@K returns 0.0 when k is negative.

        **Why this test is important:**
          - Invalid k value edge case
          - Must handle gracefully

        **What it tests:**
          - k=-5 → returns 0.0
        """
        recall = RecallAtK(k=-5)
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1"}

        result = recall.compute(retrieved, relevant)

        assert result == 0.0

    def test_recall_default_k_is_ten(self):
        """Test RecallAtK uses k=10 by default.

        **Why this test is important:**
          - Validates default parameter value
          - Common use case without specifying k

        **What it tests:**
          - Default k=10 is used when not specified
        """
        recall = RecallAtK()

        assert recall.k == 10

    def test_recall_auto_registers_with_registry(self):
        """Test RecallAtK auto-registers with MetricRegistry.

        **Why this test is important:**
          - Metrics must be discoverable by name
          - Enables dynamic metric selection

        **What it tests:**
          - MetricRegistry.get("recall@k") returns RecallAtK class
        """
        metric_cls = MetricRegistry.get("recall@k")

        assert metric_cls is not None
        assert metric_cls.__name__ == "RecallAtK"
        # Verify it can be instantiated and compute
        instance = metric_cls(k=5)
        assert instance.compute(["a", "b"], {"a", "c"}) == pytest.approx(0.5)

    def test_recall_graded_parameter_ignored(self):
        """Test that graded parameter doesn't affect Recall@K.

        **Why this test is important:**
          - Recall@K is binary, not graded
          - graded param exists for interface compatibility

        **What it tests:**
          - Same result with or without graded parameter
        """
        recall = RecallAtK(k=5)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc7"}
        graded = {"doc1": 3, "doc3": 2, "doc7": 1}

        result_with_graded = recall.compute(retrieved, relevant, graded)
        result_without_graded = recall.compute(retrieved, relevant)

        assert result_with_graded == result_without_graded

    def test_recall_fewer_results_than_k(self):
        """Test Recall@K when retrieved list is shorter than k.

        **Why this test is important:**
          - Retrieved list may be shorter than k
          - Should still compute correctly based on what's available

        **What it tests:**
          - 2 docs retrieved, k=5, 1 of 2 relevant found → recall = 1/2 = 0.5
        """
        recall = RecallAtK(k=5)
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc3"}

        result = recall.compute(retrieved, relevant)

        assert result == pytest.approx(0.5)


class TestMetricRegistryIntegration:
    """Test MetricRegistry integration with IR metrics."""

    def test_all_ir_metrics_registered(self):
        """Test that all IR metrics are registered.

        **Why this test is important:**
          - Ensures metrics are discoverable
          - Validates auto-registration works

        **What it tests:**
          - Both precision@k and recall@k are in registry
        """
        all_metrics = MetricRegistry.all_metrics()

        assert "precision@k" in all_metrics
        assert "recall@k" in all_metrics

    def test_registry_returns_correct_classes(self):
        """Test that registry returns correct metric classes.

        **Why this test is important:**
          - Validates registry correctness
          - Ensures get() returns proper types

        **What it tests:**
          - get() returns classes with correct names that can be instantiated
        """
        precision_cls = MetricRegistry.get("precision@k")
        recall_cls = MetricRegistry.get("recall@k")

        assert precision_cls is not None
        assert recall_cls is not None
        assert precision_cls.__name__ == "PrecisionAtK"
        assert recall_cls.__name__ == "RecallAtK"

        # Verify both can be instantiated
        p = precision_cls(k=5)
        r = recall_cls(k=5)
        assert hasattr(p, "compute")
        assert hasattr(r, "compute")
