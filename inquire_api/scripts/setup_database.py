"""Script to upload a HuggingFace dataset to a vector database."""

from pathlib import Path
from typing import Annotated

import typer
from datasets import load_from_disk
from loguru import logger

from inquire_api.container import Container
from inquire_api.vector_db import VectorDatabaseAdaptor

app = typer.Typer(pretty_exceptions_enable=False)


@app.command("main")
def main(dataset_path: Annotated[Path, typer.Argument(help="The dataset path")]):
    """Main runner."""
    # Check if dataset already exists
    if dataset_path.exists():
        dataset = load_from_disk(dataset_path=dataset_path)
    else:
        raise ValueError("Cannot load dataset")

    logger.info("Starting container")
    Container()

    vector_db = VectorDatabaseAdaptor(dataset_path=dataset_path, collection_name="inatinq")
    vector_db.initialize_collection(dataset, batch_size=1024)


if __name__ == "__main__":
    app()
