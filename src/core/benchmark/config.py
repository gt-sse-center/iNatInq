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
from typing import Annotated, Literal

from pydantic import Field
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

    k_values: Annotated[list[Annotated[int, Field(ge=1)]], Field(min_length=1)] = [50]
    metrics: list[str] = [
        "precision@k",
        "recall@k",
        "map",
        "ndcg",
        "mrr",
    ]
    warmup_queries: Annotated[int, Field(ge=0)] = 5
    cooldown_seconds: float = 0.1
    concurrent_queries: Annotated[int, Field(ge=1)] = 1
    output_format: Literal["console", "json", "both"] = "console"
    output_path: Path | None = None
    providers: list[str] = []
