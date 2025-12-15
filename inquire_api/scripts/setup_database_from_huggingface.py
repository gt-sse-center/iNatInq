"""Script to upload a HuggingFace dataset + Postgres data to a vector database."""

from typing import Annotated

import typer
from datasets import load_dataset
from inat_toolkit.database import Observation, Photo, Taxon
from inat_toolkit.image import get_url
from loguru import logger
from qdrant_client import models
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from inquire_api.vector_db import HuggingFaceDataset, VectorDatabaseAdaptor

app = typer.Typer(pretty_exceptions_enable=False)


def query_inaturalist_database(photo_id: int, db_host: str, db_port: str):
    """Query the inaturalist postgres database for the row which has the specified photo id."""
    engine = create_engine(f"postgresql+psycopg://{db_host}:{db_port}/inaturalist-open-data")
    with Session(engine) as session:
        stmt = (
            select(  # For getting the image
                Photo.photo_id,
                Photo.extension,
                Observation.observer_id,
                # For position filters
                Observation.latitude,
                Observation.longitude,
                Observation.positional_accuracy,
                # For date filtering
                Observation.observed_on,
                # Get the species type info
                Taxon.taxon_id,
                Taxon.name.label("taxon"),
            )
            .join(Observation, Photo.observation_uuid == Observation.observation_uuid)
            # this should be a left join since taxon_id may be empty
            .join(Taxon, Observation.taxon_id == Taxon.taxon_id, isouter=True)
            .where(Photo.photo_id == photo_id)
        )
        result = session.execute(stmt)

    # Close the connection by disposing of the engine
    engine.dispose()

    return result


def initialize_collection(
    vectordb: VectorDatabaseAdaptor,
    dataset: HuggingFaceDataset,
    db_host: str,
    db_port: str,
) -> None:
    """Create a dataset collection and upload data to it."""
    logger.info(f"Creating collection {vectordb.collection_name}")

    dim = dataset.info.features["embedding"].length

    vectors_config = models.VectorParams(
        size=dim,
        distance=vectordb.translate_metric(vectordb.metric),
        on_disk=True,  # save to disk immediately
    )
    index_params = models.HnswConfigDiff(
        m=0,  # disable indexing until dataset upload is complete
        ef_construct=vectordb.ef,
        max_indexing_threads=0,
        on_disk=True,  # Store index on disk
    )

    if not vectordb.client.collection_exists(collection_name=vectordb.collection_name):
        vectordb.client.create_collection(
            collection_name=vectordb.collection_name,
            vectors_config=vectors_config,
            hnsw_config=index_params,
            shard_number=4,  # reasonable default as per qdrant docs
        )

    dataset_size = dataset.info.splits["train"].num_examples
    # We iterate one by one since each photo_id can have multiple entries for different observations
    for d in tqdm(dataset, total=dataset_size):
        valid = d["status"]
        if not valid:
            continue

        photo_id = d["photo_id"]
        vector = d["embedding"]

        ids = []
        vectors = []
        metadata = []
        for row in query_inaturalist_database(photo_id, db_host, db_port):
            m = {
                "img_url": get_url(int(row.photo_id), str(row.extension)),
                "file_name": f"photos/{row.photo_id}/medium.{row.extension}",
                "location": vectordb.get_geo_coordinate(row.latitude, row.longitude),
                "positional_accuracy": row.positional_accuracy,
                "observed_on": vectordb.get_rfc339_date(row.observed_on),
                "species": row.taxon,
            }
            metadata.append(m)

            # Add the photo id and embedding vector equal number of times
            ids.append(photo_id)
            vectors.append(vector)

        vectordb.client.upsert(
            collection_name=vectordb.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=metadata,
            ),
        )

    # Set the indexing params
    vectordb.client.update_collection(
        collection_name=vectordb.collection_name,
        hnsw_config=models.HnswConfigDiff(m=vectordb.m),
    )

    # Log the number of point uploaded
    num_points_in_db = vectordb.client.count(
        collection_name=vectordb.collection_name,
        exact=True,
    ).count
    logger.info(f"Number of points in vector database: {num_points_in_db}")

    logger.info("Waiting for indexing to complete")
    vectordb.wait_for_index_ready(vectordb.collection_name)
    logger.info("Indexing complete!")


@app.command("main")
def main(
    dataset_name: Annotated[str, typer.Argument(help="Dataset on HuggingFace")],
    revision: Annotated[str, typer.Argument(help="The dataset revision on HuggingFace")],
    db_host: Annotated[
        str, typer.Option(help="The hostname for where the Postgres DB server is running")
    ] = "localhost",
    db_port: Annotated[str, typer.Option(help="The port where the Postgres DB server is accessed")] = "5432",
):
    """Main runner."""

    # Check if dataset already exists
    logger.info("Loading dataset from HuggingFace")
    dataset = load_dataset(dataset_name, streaming=True, split="train", revision=revision)

    vector_db = VectorDatabaseAdaptor(collection_name=f"inatinq-{revision}", port="7333", grpc_port="7334")
    initialize_collection(vector_db, dataset, db_host, db_port)


if __name__ == "__main__":
    app()
