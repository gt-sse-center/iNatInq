"""Script to generate a HuggingFace dataset of the iNaturalist Open Dataset with embeddings."""

from datasets import Dataset, Features, List, Split, Value
from loguru import logger
from sqlalchemy import Result, create_engine, select
from sqlalchemy.orm import Session
from tqdm import tqdm

from inat_toolkit.database import Observation, Photo, Taxon
from inat_toolkit.embed import ImageEmbedder
from inat_toolkit.image import download_image, get_url


def create_huggingface_dataset(query_results: Result, num_data_points: int) -> Dataset:
    """Create the HuggingFace dataset."""
    image_embedder = ImageEmbedder(model_id="openai/clip-vit-base-patch16")

    features = Features(
        {
            # `id` column to be of type int64
            "id": Value("int64"),
            "img_url": Value("string"),
            "file_name": Value("string"),
            # `img_embedding` column is of type datasets.List[float32]
            "img_embedding": List(feature=Value("float32"), length=image_embedder.embedding_dim),
            "latitude": Value("float32"),
            "longitude": Value("float32"),
            "positional_accuracy": Value("float32"),
            "observed_on": Value("date32"),
            "taxon": Value("string"),
        },
    )

    # We can't use Dataset.from_generator since HF tries to serialize the generator, and that fails due to the type of `query_results`.
    # Hence we do it by first initializing an empty dataset and then using the generator to incrementally add to it.
    dataset = Dataset.from_dict(
        mapping={
            "id": [],
            "img_url": [],
            "file_name": [],
            "img_embedding": [],
            "latitude": [],
            "longitude": [],
            "positional_accuracy": [],
            "observed_on": [],
            "taxon": [],
        },
        features=features,
        split=Split.TRAIN,
    )

    for row in tqdm(query_results, total=num_data_points):
        img_url = get_url(int(row.photo_id), str(row.extension))

        d = {
            "id": row.photo_id,
            "img_url": img_url,
            "file_name": f"phots/{row.photo_id}/medium.{row.extension}",
            "img_embedding": image_embedder(download_image(img_url)).squeeze(),
            "latitude": row.latitude,
            "longitude": row.longitude,
            "positional_accuracy": row.positional_accuracy,
            "observed_on": row.observed_on,
            "taxon": row.taxon,
        }

        dataset = dataset.add_item(d)

    return dataset


def main():
    """Main runner."""

    num_data_points = 5_000  # _000
    with Session(create_engine("postgresql+psycopg://localhost:5432/inaturalist-open-data")) as session:
        stmt = (
            select(
                # For getting the image
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
            .join(
                Taxon, Observation.taxon_id == Taxon.taxon_id, isouter=True
            )  # this should be a left join since taxon_id may be empty
            .limit(num_data_points)
        )

        results = session.execute(stmt)

        logger.info(f"Database query returned succesfully with {num_data_points} results")

        hf_dataset = create_huggingface_dataset(results, num_data_points)

        dataset_path = f"data/inaturalist_open_dataset_{num_data_points}"
        logger.info(f"Saving to disk at {dataset_path}")
        hf_dataset.save_to_disk(dataset_path)

        # logger.info("Logging into 🤗")
        # hf_login(token=access_token)
        # logger.info(f"Pushing to 🤗 dataset {hub}")
        # dataset.push_to_hub(hub)


if __name__ == "__main__":
    main()
