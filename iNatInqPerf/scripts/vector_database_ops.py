#!/bin/env/python
"""Script to create the specified vector database."""

from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset, load_from_disk
from loguru import logger

from inatinqperf.benchmark.benchmark import Benchmarker
from inatinqperf.container import container_context

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def build(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="The configuration file to use for creating the benchmark.",
        ),
    ],
    base_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="The base path relative to which various artifacts are saved.",
        ),
    ] = Path(__file__).parent.parent,
    dataset_path: Annotated[
        Path | None, typer.Option(exists=True, help="The path to the local Hugging Face dataset directory")
    ] = None,
    dataset_id: Annotated[
        str | None, typer.Option(help="The name/id of the dataset on HuggingFace.co")
    ] = None,
    dataset_revision: Annotated[str, typer.Option(help="The dataset revision on Hugging Face")] = "main",
) -> None:
    """Create the vector database."""

    benchmarker = Benchmarker(config_file, base_path=base_path)

    with container_context(benchmarker.cfg, auto_stop=False):
        # Build specified vector database

        if dataset_path:
            dataset = load_from_disk(dataset_path=dataset_path)
        elif dataset_id:
            dataset = load_dataset(dataset_id, revision=dataset_revision, split="train")
        else:
            logger.error("No dataset provided, exiting...")
            return

        if "id" not in dataset.column_names:
            dataset = dataset.rename_column("photo_id", "id")

        benchmarker.build(dataset)


@app.command()
def search(
    config_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="The configuration file to use for creating the benchmark.",
        ),
    ],
    base_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="The base path relative to which various artifacts are saved.",
        ),
    ] = Path(__file__).parent.parent,
    baseline_results_filepath: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="The (optional) path to the baseline results.",
        ),
    ] = None,
) -> None:
    """Run search performance benchmark on the vector database."""

    benchmarker = Benchmarker(config_file, base_path=base_path)
    benchmarker.search(
        benchmarker.get_vector_db(),
        baseline_results_path=baseline_results_filepath,
    )


if __name__ == "__main__":
    app()
