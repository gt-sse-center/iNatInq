#!/bin/env/python
"""Script to create the specified vector database."""

from pathlib import Path
from typing import Annotated

import typer

from inatinqperf.benchmark.benchmark import Benchmarker

app = typer.Typer(pretty_exceptions_enable=False)


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
