"""Unit tests for metrics builder."""

from __future__ import annotations

import pytest

from core.benchmark.config import BenchmarkConfig
from core.benchmark.metrics.builder import build_metrics_from_config
from core.benchmark.metrics.ir import MRR, NDCG, MeanAveragePrecision, PrecisionAtK, RecallAtK


class TestBuildMetricsFromConfig:
    """Tests for build_metrics_from_config."""

    def test_default_config_builds_all_metrics(self):
        """Default BenchmarkConfig produces all 5 metric types."""
        config = BenchmarkConfig()
        metrics = build_metrics_from_config(config)

        metric_types = {type(m) for m in metrics}
        assert PrecisionAtK in metric_types
        assert RecallAtK in metric_types
        assert NDCG in metric_types
        assert MeanAveragePrecision in metric_types
        assert MRR in metric_types

    def test_k_values_generate_k_dependent_metrics(self):
        """Each k value generates PrecisionAtK, RecallAtK, NDCG instances."""
        config = BenchmarkConfig(k_values=[5, 10, 20])
        metrics = build_metrics_from_config(config)

        precision_metrics = [m for m in metrics if isinstance(m, PrecisionAtK)]
        recall_metrics = [m for m in metrics if isinstance(m, RecallAtK)]
        ndcg_metrics = [m for m in metrics if isinstance(m, NDCG)]

        assert len(precision_metrics) == 3
        assert [m.k for m in precision_metrics] == [5, 10, 20]
        assert len(recall_metrics) == 3
        assert [m.k for m in recall_metrics] == [5, 10, 20]
        assert len(ndcg_metrics) == 3
        assert [m.k for m in ndcg_metrics] == [5, 10, 20]

    def test_single_k_value(self):
        """Single k value produces one of each k-dependent metric."""
        config = BenchmarkConfig(k_values=[10])
        metrics = build_metrics_from_config(config)

        precision_metrics = [m for m in metrics if isinstance(m, PrecisionAtK)]
        assert len(precision_metrics) == 1
        assert precision_metrics[0].k == 10

    def test_only_precision_metric(self):
        """Config with only precision@k produces only PrecisionAtK metrics."""
        config = BenchmarkConfig(metrics=["precision@k"], k_values=[5, 10])
        metrics = build_metrics_from_config(config)

        assert all(isinstance(m, PrecisionAtK) for m in metrics)
        assert len(metrics) == 2

    def test_only_map_metric(self):
        """Config with only MAP produces one MeanAveragePrecision."""
        config = BenchmarkConfig(metrics=["map"], k_values=[5, 10])
        metrics = build_metrics_from_config(config)

        assert len(metrics) == 1
        assert isinstance(metrics[0], MeanAveragePrecision)

    def test_only_mrr_metric(self):
        """Config with only MRR produces one MRR."""
        config = BenchmarkConfig(metrics=["mrr"], k_values=[5])
        metrics = build_metrics_from_config(config)

        assert len(metrics) == 1
        assert isinstance(metrics[0], MRR)

    def test_unknown_metric_is_skipped(self):
        """Unknown metric names are silently skipped."""
        config = BenchmarkConfig(metrics=["unknown_metric", "map"], k_values=[5])
        metrics = build_metrics_from_config(config)

        assert len(metrics) == 1
        assert isinstance(metrics[0], MeanAveragePrecision)

    def test_empty_metrics_list(self):
        """Empty metrics list returns empty result."""
        config = BenchmarkConfig(metrics=[], k_values=[5])
        metrics = build_metrics_from_config(config)
        assert metrics == []

    def test_metrics_are_callable(self):
        """All built metrics can compute scores."""
        config = BenchmarkConfig()
        metrics = build_metrics_from_config(config)

        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d3"}

        for metric in metrics:
            score = metric.compute(retrieved, relevant)
            assert isinstance(score, float)
