"""Helper script to load embeddings from HuggingFace dataset to the postgres database with the pgvector extension."""

from typing import Annotated

import typer
from datasets import Dataset, load_dataset
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session
from tqdm import tqdm

from inat_toolkit.database import Photo

app = typer.Typer(pretty_exceptions_enable=False)


def insert_into_db(hf_dataset: Dataset, session: Session):
    """Insert embeddings into the Postgres database based on photo ID."""

    dataset_size = len(hf_dataset)
    for d in tqdm(hf_dataset, total=dataset_size):
        valid = d["status"]
        if not valid:
            continue

        photo_id = d["photo_id"]
        embedding = d["embedding"]

        statement = update(Photo).where(Photo.photo_id == photo_id).values(embedding=embedding)

        session.execute(statement)
        session.commit()


@app.command("main")
def main(
    dataset_name: Annotated[str, typer.Argument(help="Dataset on HuggingFace")],
    dataset_revision: Annotated[str, typer.Argument(help="The dataset revision on HuggingFace")],
):
    """Main runner."""
    hf_dataset = load_dataset(dataset_name, revision=dataset_revision, split="train")

    connection_string = "postgresql+psycopg://localhost:5432/inaturalist-open-data"

    with Session(create_engine(connection_string)) as session:
        insert_into_db(hf_dataset, session=session)


if __name__ == "__main__":
    app()
