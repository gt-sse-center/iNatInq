"""Benchmark configuration using pydantic-settings.

Loads configuration from environment variables with the ``BENCHMARK_`` prefix.

Example:
    ```python
    from core.benchmark.config import BenchmarkConfig

    config = BenchmarkConfig()
    print(config.k_values)        # [5, 10, 20]
    print(config.output_format)   # "console"

    # Override with env vars:
    # BENCHMARK_K_VALUES='[5,10]'
    # BENCHMARK_OUTPUT_FORMAT='json'
    ```
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - required at runtime by Pydantic
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class BenchmarkConfig(BaseSettings):
    """Configuration for benchmark execution.

    Attributes:
        k_values: List of K values for top-K metrics (e.g., P@K, R@K, NDCG@K).
        metrics: List of metric names to compute.
        warmup_queries: Number of warmup queries before timing.
        cooldown_seconds: Seconds to wait between warmup and measurement.
        concurrent_queries: Number of concurrent queries (1 = sequential).
        output_format: Output format for results.
        output_path: Path for file-based output (JSON reporter).
        providers: List of provider names to benchmark.

    Environment Variables:
        All fields can be set via ``BENCHMARK_`` prefixed env vars.
        E.g., ``BENCHMARK_K_VALUES='[5,10]'``.
    """

    model_config = {"env_prefix": "BENCHMARK_"}

    k_values: list[int] = [5, 10, 20]
    metrics: list[str] = [
        "precision@k",
        "recall@k",
        "map",
        "ndcg",
        "mrr",
    ]
    warmup_queries: int = 5
    cooldown_seconds: float = 0.1
    concurrent_queries: int = 1
    output_format: Literal["console", "json", "both"] = "console"
    output_path: Path | None = None
    providers: list[str] = []

    @field_validator("k_values")
    @classmethod
    def k_values_must_be_positive(cls, v: list[int]) -> list[int]:
        """Validate that all K values are positive integers."""
        if not v:
            msg = "k_values must not be empty"
            raise ValueError(msg)
        for k in v:
            if k < 1:
                msg = f"k_values must be positive, got {k}"
                raise ValueError(msg)
        return v

    @field_validator("warmup_queries")
    @classmethod
    def warmup_must_be_non_negative(cls, v: int) -> int:
        """Validate that warmup_queries is non-negative."""
        if v < 0:
            msg = f"warmup_queries must be non-negative, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("concurrent_queries")
    @classmethod
    def concurrent_must_be_positive(cls, v: int) -> int:
        """Validate that concurrent_queries is at least 1."""
        if v < 1:
            msg = f"concurrent_queries must be at least 1, got {v}"
            raise ValueError(msg)
        return v
