#!/bin/env/python
"""Script to create the specified vector database."""

from pathlib import Path
from typing import Annotated

import typer
from datasets import load_dataset

from inatinqperf.benchmark.benchmark import Benchmarker
from inatinqperf.container import container_context

app = typer.Typer(pretty_exceptions_enable=False)


@app.command("build")
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
    dataset_name: Annotated[
        str, typer.Option(help="The name of the dataset to ")
    ] = "gt-csse/inat-open-data-embeddings",
    dataset_revision: Annotated[str, typer.Option(help="The dataset revision on Hugging Face")] = "main",
) -> None:
    """Create the vector database."""

    benchmarker = Benchmarker(config_file, base_path=base_path)

    with container_context(benchmarker.cfg, auto_stop=False):
        # Build specified vector database
        dataset = load_dataset(dataset_name, revision=dataset_revision)
        dataset = dataset.rename_column("photo_id", "id")
        benchmarker.build(dataset)


if __name__ == "__main__":
    app()
