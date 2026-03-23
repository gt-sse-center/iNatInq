"""CLI for the benchmark framework using Typer.

Provides commands for running benchmarks, comparing providers,
listing metrics, and validating datasets.

Example:
    ```bash
    uv run python -m core.benchmark.cli validate benchmarks/sample/sample-gold.json
    uv run python -m core.benchmark.cli metrics
    uv run python -m core.benchmark.cli compare --dataset gold.json --provider qdrant
    uv run python -m core.benchmark.cli run --dataset gold.json --provider qdrant
    ```
"""

from __future__ import annotations

import asyncio
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from core.benchmark.datasets.json_dataset import JSONDataset
from core.benchmark.metrics.base import MetricRegistry

app = typer.Typer(help="Benchmark framework for evaluating vector database providers.")


class OutputFormat(StrEnum):
    """Output format for benchmark results."""

    console = "console"
    json = "json"
    both = "both"


@app.command()
def validate(
    dataset_path: Annotated[Path, typer.Argument(help="Path to dataset JSON file.")],
) -> None:
    """Validate a benchmark dataset JSON file.

    Loads the dataset and checks that it conforms to the expected schema.
    """
    try:
        dataset = JSONDataset.from_file(dataset_path)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as e:
        typer.echo(f"Validation error: {e}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Dataset '{dataset.name}' is valid.")
    typer.echo(f"  Modality: {dataset.modality}")
    typer.echo(f"  Queries:  {len(dataset)}")


@app.command()
def metrics() -> None:
    """List all available benchmark metrics."""
    import core.benchmark.metrics.ir  # noqa: F401

    all_metrics = MetricRegistry.all_metrics()
    if not all_metrics:
        typer.echo("No metrics registered.")
        return

    typer.echo("Available metrics:")
    for name in all_metrics:
        metric_cls = MetricRegistry.get(name)
        description = getattr(metric_cls, "description", "") if metric_cls else ""
        typer.echo(f"  {name}: {description}")


def _build_reporters(
    output_format: OutputFormat,
    output_path: Path | None,
) -> list:
    """Build reporter instances from CLI options.

    Args:
        output_format: Desired output format.
        output_path: File path for JSON output.

    Returns:
        List of Reporter instances.
    """
    from core.benchmark.reporters.console import ConsoleReporter
    from core.benchmark.reporters.json_reporter import JSONReporter

    reporters: list = []

    if output_format in (OutputFormat.console, OutputFormat.both):
        reporters.append(ConsoleReporter(stream=sys.stdout))

    if output_format in (OutputFormat.json, OutputFormat.both):
        path = output_path or Path("benchmark-results.json")
        reporters.append(JSONReporter(output_path=path))

    return reporters


@app.command()
def compare(
    dataset_path: Annotated[Path | None, typer.Option("--dataset", help="Path to dataset JSON file.")] = None,
    providers: Annotated[
        list[str] | None, typer.Option("--provider", help="Provider name(s) to benchmark.")
    ] = None,
    collection: Annotated[
        str | None, typer.Option("--collection", help="Qdrant collection name override.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of results per query.")] = 10,
    warmup: Annotated[int, typer.Option("--warmup", help="Number of warmup queries.")] = 5,
    output_path: Annotated[Path | None, typer.Option("--output", help="Output file path.")] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.console,
) -> None:
    """Compare multiple providers on a benchmark dataset.

    Resolves each provider via ``resolve_search_pipeline``, runs benchmarks,
    and reports results.
    """
    if not dataset_path:
        typer.echo("Error: --dataset is required.", err=True)
        raise typer.Exit(code=1)

    if not providers:
        typer.echo("Error: at least one --provider is required.", err=True)
        raise typer.Exit(code=1)

    from core.benchmark.config import BenchmarkConfig
    from core.benchmark.metrics.builder import build_metrics_from_config
    from core.benchmark.provider_factory import resolve_search_pipeline
    from core.benchmark.runner.base import BenchmarkResult  # noqa: TC001
    from core.benchmark.runner.default import DefaultBenchmarkRunner

    try:
        dataset = JSONDataset.from_file(dataset_path)
    except Exception as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1) from None

    config = BenchmarkConfig()
    active_metrics = build_metrics_from_config(config)
    reporters = _build_reporters(output_format, output_path)

    async def _run_compare() -> dict[str, BenchmarkResult]:
        results: dict[str, BenchmarkResult] = {}
        for provider_name in providers:  # type: ignore[union-attr]
            try:
                provider, pipeline = resolve_search_pipeline(provider_name, collection=collection)
            except (ValueError, Exception) as e:
                typer.echo(f"Error resolving provider '{provider_name}': {e}", err=True)
                raise typer.Exit(code=1) from None

            runner = DefaultBenchmarkRunner(
                search_pipeline=pipeline,
                collection=collection,
            )
            result = await runner.run(
                provider=provider,
                dataset=dataset,
                metrics=active_metrics,
                limit=limit,
                warmup_queries=warmup,
            )
            results[provider_name] = result

        for reporter in reporters:
            await reporter.report(results)

        return results

    asyncio.run(_run_compare())

    typer.echo(f"Benchmark complete for providers: {', '.join(providers)}")


@app.command()
def run(
    dataset_path: Annotated[Path | None, typer.Option("--dataset", help="Path to dataset JSON file.")] = None,
    provider_name: Annotated[
        str | None, typer.Option("--provider", help="Provider name to benchmark.")
    ] = None,
    collection: Annotated[
        str | None, typer.Option("--collection", help="Qdrant collection name override.")
    ] = None,
    quantization_profile: Annotated[
        str | None,
        typer.Option(
            "--quantization-profile",
            help="Quantization profile (none, scalar, scalar-rescore, binary-rescore).",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of results per query.")] = 10,
    warmup: Annotated[int, typer.Option("--warmup", help="Number of warmup queries.")] = 5,
    output_path: Annotated[Path | None, typer.Option("--output", help="Output file path.")] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.console,
) -> None:
    """Run a benchmark against a single provider.

    Resolves the provider via ``resolve_search_pipeline``, runs the benchmark,
    and reports results.
    """
    if not dataset_path:
        typer.echo("Error: --dataset is required.", err=True)
        raise typer.Exit(code=1)

    if not provider_name:
        typer.echo("Error: --provider is required.", err=True)
        raise typer.Exit(code=1)

    from core.benchmark.config import BenchmarkConfig
    from core.benchmark.metrics.builder import build_metrics_from_config
    from core.benchmark.provider_factory import resolve_search_pipeline
    from core.benchmark.runner.default import DefaultBenchmarkRunner

    try:
        dataset = JSONDataset.from_file(dataset_path)
    except Exception as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1) from None

    try:
        provider, pipeline = resolve_search_pipeline(
            provider_name,
            collection=collection,
            quantization_profile=quantization_profile,
        )
    except (ValueError, Exception) as e:
        typer.echo(f"Error resolving provider '{provider_name}': {e}", err=True)
        raise typer.Exit(code=1) from None

    config = BenchmarkConfig()
    active_metrics = build_metrics_from_config(config)
    reporters = _build_reporters(output_format, output_path)

    async def _run_benchmark() -> None:
        runner = DefaultBenchmarkRunner(
            search_pipeline=pipeline,
            collection=collection,
        )
        result = await runner.run(
            provider=provider,
            dataset=dataset,
            metrics=active_metrics,
            limit=limit,
            warmup_queries=warmup,
        )

        label = f"{provider_name}:{quantization_profile}" if quantization_profile else provider_name  # type: ignore[arg-type]
        results = {label: result}
        for reporter in reporters:
            await reporter.report(results)

    asyncio.run(_run_benchmark())

    typer.echo(f"Benchmark complete for provider: {provider_name}")


@app.command()
def quantization(
    dataset_path: Annotated[Path | None, typer.Option("--dataset", help="Path to dataset JSON file.")] = None,
    collections: Annotated[
        list[str] | None,
        typer.Option("--collection", help="Collection(s) in profile order: float32, scalar, binary."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of results per query.")] = 50,
    warmup: Annotated[int, typer.Option("--warmup", help="Number of warmup queries.")] = 5,
    output_path: Annotated[Path | None, typer.Option("--output", help="Output file path.")] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.both,
) -> None:
    r"""Run a 4-profile quantization benchmark comparing float32, scalar, scalar+rescore, and binary+rescore.

    Requires three Qdrant collections (float32 baseline, scalar-quantized,
    binary-quantized) with identical point sets.

    Example::

        cd src/
        uv run python -m core.benchmark.cli quantization \
            --dataset ../benchmarks/inquire/inquire-val-bench20k.json \
            --collection bench-siglip-float32 \
            --collection bench-siglip-scalar \
            --collection bench-siglip-binary \
            --limit 50 --warmup 5 \
            --output ../benchmarks/results/quantization-benchmark.json
    """
    if not dataset_path:
        typer.echo("Error: --dataset is required.", err=True)
        raise typer.Exit(code=1)

    if not collections or len(collections) != 3:
        typer.echo(
            "Error: exactly 3 --collection values required (float32, scalar, binary).",
            err=True,
        )
        raise typer.Exit(code=1)

    from core.benchmark.config import BenchmarkConfig
    from core.benchmark.metrics.builder import build_metrics_from_config
    from core.benchmark.provider_factory import resolve_search_pipeline
    from core.benchmark.runner.base import BenchmarkResult  # noqa: TC001
    from core.benchmark.runner.default import DefaultBenchmarkRunner

    float32_col, scalar_col, binary_col = collections

    profiles: list[tuple[str, str, str | None]] = [
        ("float32-baseline", float32_col, None),
        ("scalar-int8", scalar_col, None),
        ("scalar-int8-rescore", scalar_col, "scalar-rescore"),
        ("binary-oversample-rescore", binary_col, "binary-rescore"),
    ]

    try:
        dataset = JSONDataset.from_file(dataset_path)
    except Exception as e:
        typer.echo(f"Error loading dataset: {e}", err=True)
        raise typer.Exit(code=1) from None

    config = BenchmarkConfig(k_values=[limit])
    active_metrics = build_metrics_from_config(config)
    reporters = _build_reporters(output_format, output_path)

    async def _run_quantization() -> dict[str, BenchmarkResult]:
        results: dict[str, BenchmarkResult] = {}

        for profile_name, col, q_profile in profiles:
            typer.echo(f"Running profile '{profile_name}' on collection '{col}'...")
            try:
                provider, pipeline = resolve_search_pipeline(
                    "qdrant",
                    collection=col,
                    quantization_profile=q_profile,
                )
            except (ValueError, Exception) as e:
                typer.echo(f"Error resolving profile '{profile_name}': {e}", err=True)
                raise typer.Exit(code=1) from None

            runner = DefaultBenchmarkRunner(
                search_pipeline=pipeline,
                collection=col,
            )
            result = await runner.run(
                provider=provider,
                dataset=dataset,
                metrics=active_metrics,
                limit=limit,
                warmup_queries=warmup,
            )
            results[profile_name] = result

        for reporter in reporters:
            await reporter.report(results)

        return results

    asyncio.run(_run_quantization())

    typer.echo("Quantization benchmark complete.")


if __name__ == "__main__":
    app()
