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

import json
from enum import StrEnum
from pathlib import Path  # noqa: TC003 - required at runtime for typer get_type_hints()
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


@app.command()
def compare(
    dataset_path: Annotated[Path | None, typer.Option("--dataset", help="Path to dataset JSON file.")] = None,
    providers: Annotated[
        list[str] | None, typer.Option("--provider", help="Provider name(s) to benchmark.")
    ] = None,
    output_path: Annotated[Path | None, typer.Option("--output", help="Output file path.")] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="Output format.")
    ] = OutputFormat.console,
) -> None:
    """Compare multiple providers on a benchmark dataset.

    Provider resolution is not yet implemented. This command currently
    validates arguments and reports configuration.
    """
    if not dataset_path:
        typer.echo("Error: --dataset is required.", err=True)
        raise typer.Exit(code=1)

    if not providers:
        typer.echo("Error: at least one --provider is required.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Dataset:  {dataset_path}")
    typer.echo(f"Providers: {', '.join(providers)}")
    typer.echo(f"Format:   {output_format.value}")
    if output_path:
        typer.echo(f"Output:   {output_path}")
    typer.echo("Note: Provider resolution not yet implemented.")


@app.command()
def run(
    dataset_path: Annotated[Path | None, typer.Option("--dataset", help="Path to dataset JSON file.")] = None,
    provider_name: Annotated[
        str | None, typer.Option("--provider", help="Provider name to benchmark.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of results per query.")] = 10,
    warmup: Annotated[int, typer.Option("--warmup", help="Number of warmup queries.")] = 5,
) -> None:
    """Run a benchmark against a single provider.

    Provider resolution is not yet implemented. This command currently
    validates arguments and reports configuration.
    """
    if not dataset_path:
        typer.echo("Error: --dataset is required.", err=True)
        raise typer.Exit(code=1)

    if not provider_name:
        typer.echo("Error: --provider is required.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Dataset:  {dataset_path}")
    typer.echo(f"Provider: {provider_name}")
    typer.echo(f"Limit:    {limit}")
    typer.echo(f"Warmup:   {warmup}")
    typer.echo("Note: Provider resolution not yet implemented.")


if __name__ == "__main__":
    app()
