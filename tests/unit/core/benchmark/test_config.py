"""Unit tests for BenchmarkConfig."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.benchmark.config import BenchmarkConfig


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_defaults(self):
        """BenchmarkConfig has sensible defaults."""
        config = BenchmarkConfig()

        assert config.k_values == [50]
        assert "precision@k" in config.metrics
        assert "recall@k" in config.metrics
        assert "map" in config.metrics
        assert "ndcg" in config.metrics
        assert "mrr" in config.metrics
        assert config.warmup_queries == 5
        assert config.cooldown_seconds == 0.1
        assert config.concurrent_queries == 1
        assert config.output_format == "console"
        assert config.output_path is None
        assert config.providers == []

    def test_env_k_values(self, monkeypatch):
        """k_values can be set via BENCHMARK_K_VALUES env var."""
        monkeypatch.setenv("BENCHMARK_K_VALUES", "[5, 10]")
        config = BenchmarkConfig()
        assert config.k_values == [5, 10]

    def test_env_output_format(self, monkeypatch):
        """output_format can be set via BENCHMARK_OUTPUT_FORMAT env var."""
        monkeypatch.setenv("BENCHMARK_OUTPUT_FORMAT", "json")
        config = BenchmarkConfig()
        assert config.output_format == "json"

    def test_env_output_path(self, monkeypatch):
        """output_path can be set via BENCHMARK_OUTPUT_PATH env var."""
        monkeypatch.setenv("BENCHMARK_OUTPUT_PATH", "/tmp/results.json")
        config = BenchmarkConfig()
        assert config.output_path == Path("/tmp/results.json")

    def test_env_warmup_queries(self, monkeypatch):
        """warmup_queries can be set via BENCHMARK_WARMUP_QUERIES env var."""
        monkeypatch.setenv("BENCHMARK_WARMUP_QUERIES", "10")
        config = BenchmarkConfig()
        assert config.warmup_queries == 10

    def test_env_providers(self, monkeypatch):
        """providers can be set via BENCHMARK_PROVIDERS env var."""
        monkeypatch.setenv("BENCHMARK_PROVIDERS", '["qdrant", "weaviate"]')
        config = BenchmarkConfig()
        assert config.providers == ["qdrant", "weaviate"]

    def test_invalid_k_values_empty(self):
        """Empty k_values raises ValidationError."""
        with pytest.raises(ValidationError, match="too_short"):
            BenchmarkConfig(k_values=[])

    def test_invalid_k_values_negative(self):
        """Negative k_values raises ValidationError."""
        with pytest.raises(ValidationError, match="greater_than_equal"):
            BenchmarkConfig(k_values=[-1])

    def test_invalid_k_values_zero(self):
        """Zero k_values raises ValidationError."""
        with pytest.raises(ValidationError, match="greater_than_equal"):
            BenchmarkConfig(k_values=[0])

    def test_invalid_warmup_negative(self):
        """Negative warmup_queries raises ValidationError."""
        with pytest.raises(ValidationError, match="greater_than_equal"):
            BenchmarkConfig(warmup_queries=-1)

    def test_invalid_concurrent_zero(self):
        """Zero concurrent_queries raises ValidationError."""
        with pytest.raises(ValidationError, match="greater_than_equal"):
            BenchmarkConfig(concurrent_queries=0)

    def test_invalid_output_format(self):
        """Invalid output_format raises ValidationError."""
        with pytest.raises(ValidationError):
            BenchmarkConfig(output_format="xml")

    def test_output_format_both(self):
        """output_format can be 'both'."""
        config = BenchmarkConfig(output_format="both")
        assert config.output_format == "both"

    def test_custom_values(self):
        """BenchmarkConfig can be created with custom values."""
        config = BenchmarkConfig(
            k_values=[1, 3, 5],
            metrics=["precision@k"],
            warmup_queries=0,
            cooldown_seconds=0.5,
            concurrent_queries=4,
            output_format="json",
            output_path=Path("/tmp/out.json"),
            providers=["qdrant"],
        )
        assert config.k_values == [1, 3, 5]
        assert config.metrics == ["precision@k"]
        assert config.warmup_queries == 0
        assert config.cooldown_seconds == 0.5
        assert config.concurrent_queries == 4
        assert config.output_format == "json"
        assert config.output_path == Path("/tmp/out.json")
        assert config.providers == ["qdrant"]
