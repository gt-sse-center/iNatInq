"""Unit tests for core.benchmark.metrics.ir module.

This file tests the Information Retrieval metrics for benchmark evaluation:
- PrecisionAtK: Fraction of top-K results that are relevant
- RecallAtK: Fraction of relevant documents in top-K results
- MeanAveragePrecision: Average precision at each relevant hit
- MRR: Reciprocal rank of the first relevant result

# Test Coverage

The tests cover:
  - Basic computation with known inputs and expected outputs
  - Edge cases: k=0, empty relevant set, empty retrieved list
  - Auto-registration with MetricRegistry
  - Default k value (10)
  - Custom k values
  - MAP calculation with multiple relevant documents
  - MRR calculation for first relevant result

# Running Tests

Run with: uv run pytest tests/unit/core/benchmark/metrics/test_ir.py -v
"""

import pytest

from core.benchmark.metrics.base import MetricRegistry
from core.benchmark.metrics.ir import MRR, MeanAveragePrecision, PrecisionAtK, RecallAtK


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

    def test_precision_k_zero_raises_value_error(self):
        """Test Precision@K raises ValueError when k=0.

        **Why this test is important:**
          - k=0 is semantically invalid for Precision@K
          - Fail fast with clear error

        **What it tests:**
          - k=0 → raises ValueError
        """
        with pytest.raises(ValueError, match="k must be at least 1"):
            PrecisionAtK(k=0)

    def test_precision_k_negative_raises_value_error(self):
        """Test Precision@K raises ValueError when k is negative.

        **Why this test is important:**
          - Negative k is semantically invalid
          - Fail fast with clear error

        **What it tests:**
          - k=-5 → raises ValueError
        """
        with pytest.raises(ValueError, match="k must be at least 1"):
            PrecisionAtK(k=-5)

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

    def test_recall_k_zero_raises_value_error(self):
        """Test Recall@K raises ValueError when k=0.

        **Why this test is important:**
          - k=0 is semantically invalid for Recall@K
          - Fail fast with clear error

        **What it tests:**
          - k=0 → raises ValueError
        """
        with pytest.raises(ValueError, match="k must be at least 1"):
            RecallAtK(k=0)

    def test_recall_k_negative_raises_value_error(self):
        """Test Recall@K raises ValueError when k is negative.

        **Why this test is important:**
          - Negative k is semantically invalid
          - Fail fast with clear error

        **What it tests:**
          - k=-5 → raises ValueError
        """
        with pytest.raises(ValueError, match="k must be at least 1"):
            RecallAtK(k=-5)

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


class TestMeanAveragePrecision:
    """Test suite for MeanAveragePrecision (MAP) metric."""

    def test_map_basic_computation(self):
        """Test basic MAP calculation with known inputs.

        **Why this test is important:**
          - Validates core MAP calculation logic
          - Ensures precision is calculated at each relevant hit
          - Verifies averaging over all relevant documents

        **What it tests:**
          - Retrieved: [rel, non, rel, non, non] with 3 total relevant
          - At position 1: precision = 1/1, at position 3: precision = 2/3
          - MAP = (1/1 + 2/3) / 3 = 5/9 ≈ 0.556
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc7"}  # doc7 not in retrieved

        result = map_metric.compute(retrieved, relevant)

        # Precision at doc1 (rank 1) = 1/1 = 1.0
        # Precision at doc3 (rank 3) = 2/3 ≈ 0.667
        # MAP = (1.0 + 0.667) / 3 ≈ 0.556
        expected = (1.0 + 2 / 3) / 3
        assert result == pytest.approx(expected)

    def test_map_perfect_ranking(self):
        """Test MAP when all relevant docs are ranked first.

        **Why this test is important:**
          - Validates perfect ranking scenario
          - MAP should be 1.0 when ranking is optimal

        **What it tests:**
          - All 3 relevant docs in positions 1, 2, 3
          - MAP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3"}

        result = map_metric.compute(retrieved, relevant)

        # Precision at each position: 1/1, 2/2, 3/3 = 1.0, 1.0, 1.0
        # MAP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert result == pytest.approx(1.0)

    def test_map_worst_ranking(self):
        """Test MAP when relevant docs are ranked last.

        **Why this test is important:**
          - Validates worst-case ranking scenario
          - MAP should be low but non-zero

        **What it tests:**
          - Relevant docs at positions 4 and 5
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc4", "doc5"}

        result = map_metric.compute(retrieved, relevant)

        # Precision at doc4 (rank 4) = 1/4 = 0.25
        # Precision at doc5 (rank 5) = 2/5 = 0.4
        # MAP = (0.25 + 0.4) / 2 = 0.325
        expected = (1 / 4 + 2 / 5) / 2
        assert result == pytest.approx(expected)

    def test_map_no_relevant_docs_found(self):
        """Test MAP returns 0.0 when no relevant docs in results.

        **Why this test is important:**
          - Validates zero-score scenario
          - No relevant documents should yield 0.0

        **What it tests:**
          - No overlap between retrieved and relevant
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5", "doc6"}

        result = map_metric.compute(retrieved, relevant)

        assert result == pytest.approx(0.0)

    def test_map_empty_relevant_returns_zero(self):
        """Test MAP returns 0.0 when relevant set is empty.

        **Why this test is important:**
          - Division by zero edge case
          - Must return 0.0 not raise exception

        **What it tests:**
          - Empty relevant set → returns 0.0
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: set[str] = set()

        result = map_metric.compute(retrieved, relevant)

        assert result == 0.0

    def test_map_single_relevant_at_first_position(self):
        """Test MAP with single relevant doc at position 1.

        **Why this test is important:**
          - Simple case for validation
          - Single relevant doc at top = perfect score

        **What it tests:**
          - One relevant doc at rank 1 → MAP = 1.0
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}

        result = map_metric.compute(retrieved, relevant)

        # Precision at doc1 (rank 1) = 1/1 = 1.0
        # MAP = 1.0 / 1 = 1.0
        assert result == pytest.approx(1.0)

    def test_map_single_relevant_at_last_position(self):
        """Test MAP with single relevant doc at last position.

        **Why this test is important:**
          - Validates late-position calculation
          - Single relevant doc at bottom = low score

        **What it tests:**
          - One relevant doc at rank 5 → MAP = 0.2
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc5"}

        result = map_metric.compute(retrieved, relevant)

        # Precision at doc5 (rank 5) = 1/5 = 0.2
        # MAP = 0.2 / 1 = 0.2
        assert result == pytest.approx(0.2)

    def test_map_auto_registers_with_registry(self):
        """Test MeanAveragePrecision auto-registers with MetricRegistry.

        **Why this test is important:**
          - Metrics must be discoverable by name
          - Enables dynamic metric selection

        **What it tests:**
          - MetricRegistry.get("map") returns MeanAveragePrecision class
        """
        metric_cls = MetricRegistry.get("map")

        assert metric_cls is not None
        assert metric_cls.__name__ == "MeanAveragePrecision"
        # Verify it can be instantiated and compute
        instance = metric_cls()
        assert instance.compute(["a", "b"], {"a"}) == pytest.approx(1.0)

    def test_map_graded_parameter_ignored(self):
        """Test that graded parameter doesn't affect MAP.

        **Why this test is important:**
          - MAP is binary, not graded
          - graded param exists for interface compatibility

        **What it tests:**
          - Same result with or without graded parameter
        """
        map_metric = MeanAveragePrecision()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}
        graded = {"doc1": 3, "doc3": 2, "doc7": 1}

        result_with_graded = map_metric.compute(retrieved, relevant, graded)
        result_without_graded = map_metric.compute(retrieved, relevant)

        assert result_with_graded == result_without_graded


class TestMRR:
    """Test suite for MRR (Mean Reciprocal Rank) metric."""

    def test_mrr_relevant_at_first_position(self):
        """Test MRR when first relevant doc is at position 1.

        **Why this test is important:**
          - Best case scenario for MRR
          - Validates 1/1 = 1.0

        **What it tests:**
          - Relevant doc at rank 1 → MRR = 1.0
        """
        mrr = MRR()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3"}

        result = mrr.compute(retrieved, relevant)

        assert result == pytest.approx(1.0)

    def test_mrr_relevant_at_second_position(self):
        """Test MRR when first relevant doc is at position 2.

        **Why this test is important:**
          - Common scenario validation
          - Validates 1/2 = 0.5

        **What it tests:**
          - First relevant doc at rank 2 → MRR = 0.5
        """
        mrr = MRR()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2", "doc3"}

        result = mrr.compute(retrieved, relevant)

        assert result == pytest.approx(0.5)

    def test_mrr_relevant_at_third_position(self):
        """Test MRR when first relevant doc is at position 3.

        **Why this test is important:**
          - Validates reciprocal calculation
          - 1/3 ≈ 0.333

        **What it tests:**
          - First relevant doc at rank 3 → MRR = 1/3
        """
        mrr = MRR()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc3"}

        result = mrr.compute(retrieved, relevant)

        assert result == pytest.approx(1 / 3)

    def test_mrr_relevant_at_tenth_position(self):
        """Test MRR when first relevant doc is at position 10.

        **Why this test is important:**
          - Tests deeper ranking positions
          - Validates 1/10 = 0.1

        **What it tests:**
          - First relevant doc at rank 10 → MRR = 0.1
        """
        mrr = MRR()
        retrieved = [f"doc{i}" for i in range(1, 11)]
        relevant = {"doc10"}

        result = mrr.compute(retrieved, relevant)

        assert result == pytest.approx(0.1)

    def test_mrr_no_relevant_docs_returns_zero(self):
        """Test MRR returns 0.0 when no relevant docs in results.

        **Why this test is important:**
          - Validates zero-score scenario
          - No relevant documents should yield 0.0

        **What it tests:**
          - No overlap between retrieved and relevant → MRR = 0.0
        """
        mrr = MRR()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5", "doc6"}

        result = mrr.compute(retrieved, relevant)

        assert result == pytest.approx(0.0)

    def test_mrr_empty_relevant_returns_zero(self):
        """Test MRR returns 0.0 when relevant set is empty.

        **Why this test is important:**
          - Edge case with no relevant documents defined
          - Should return 0.0, not error

        **What it tests:**
          - Empty relevant set → MRR = 0.0
        """
        mrr = MRR()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: set[str] = set()

        result = mrr.compute(retrieved, relevant)

        assert result == 0.0

    def test_mrr_empty_retrieved_returns_zero(self):
        """Test MRR returns 0.0 when retrieved list is empty.

        **Why this test is important:**
          - Edge case with no results
          - Should return 0.0, not error

        **What it tests:**
          - Empty retrieved list → MRR = 0.0
        """
        mrr = MRR()
        retrieved: list[str] = []
        relevant = {"doc1", "doc2"}

        result = mrr.compute(retrieved, relevant)

        assert result == 0.0

    def test_mrr_only_considers_first_relevant(self):
        """Test MRR only considers the first relevant document.

        **Why this test is important:**
          - MRR is about first relevant, not all relevant
          - Multiple relevant docs should not change score

        **What it tests:**
          - First relevant at rank 2, others later → MRR = 0.5
        """
        mrr = MRR()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4", "doc5"}  # Multiple relevant

        result = mrr.compute(retrieved, relevant)

        # Only doc2 at rank 2 matters
        assert result == pytest.approx(0.5)

    def test_mrr_auto_registers_with_registry(self):
        """Test MRR auto-registers with MetricRegistry.

        **Why this test is important:**
          - Metrics must be discoverable by name
          - Enables dynamic metric selection

        **What it tests:**
          - MetricRegistry.get("mrr") returns MRR class
        """
        metric_cls = MetricRegistry.get("mrr")

        assert metric_cls is not None
        assert metric_cls.__name__ == "MRR"
        # Verify it can be instantiated and compute
        instance = metric_cls()
        assert instance.compute(["a", "b"], {"b"}) == pytest.approx(0.5)

    def test_mrr_graded_parameter_ignored(self):
        """Test that graded parameter doesn't affect MRR.

        **Why this test is important:**
          - MRR is binary, not graded
          - graded param exists for interface compatibility

        **What it tests:**
          - Same result with or without graded parameter
        """
        mrr = MRR()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4"}
        graded = {"doc2": 3, "doc4": 2}

        result_with_graded = mrr.compute(retrieved, relevant, graded)
        result_without_graded = mrr.compute(retrieved, relevant)

        assert result_with_graded == result_without_graded


class TestMetricRegistryIntegration:
    """Test MetricRegistry integration with IR metrics."""

    def test_all_ir_metrics_registered(self):
        """Test that all IR metrics are registered.

        **Why this test is important:**
          - Ensures metrics are discoverable
          - Validates auto-registration works

        **What it tests:**
          - All four IR metrics are in registry
        """
        all_metrics = MetricRegistry.all_metrics()

        assert "precision@k" in all_metrics
        assert "recall@k" in all_metrics
        assert "map" in all_metrics
        assert "mrr" in all_metrics

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
        map_cls = MetricRegistry.get("map")
        mrr_cls = MetricRegistry.get("mrr")

        assert precision_cls is not None
        assert recall_cls is not None
        assert map_cls is not None
        assert mrr_cls is not None

        assert precision_cls.__name__ == "PrecisionAtK"
        assert recall_cls.__name__ == "RecallAtK"
        assert map_cls.__name__ == "MeanAveragePrecision"
        assert mrr_cls.__name__ == "MRR"

        # Verify all can be instantiated
        p = precision_cls(k=5)
        r = recall_cls(k=5)
        m = map_cls()
        mrr = mrr_cls()

        assert hasattr(p, "compute")
        assert hasattr(r, "compute")
        assert hasattr(m, "compute")
        assert hasattr(mrr, "compute")
