"""Script convert the INQUIRE demo dataset to a dataset with embeddings of size 512.

1. Loads the original webapp dataset.
2. Converts it to embeddings of size 512.
3. Creates a new dataset.
"""

from pathlib import Path

import PIL
import requests
from datasets import Dataset, Features, List, Value, load_from_disk
from loguru import logger
from PIL import Image
from tqdm import tqdm

from inquire_api.embedding import ImageEmbedder


def download_image(url: str):
    """Download the image at `url`."""
    resp = requests.get(url, stream=True, timeout=10)
    return Image.open(resp.raw)


def generate_dataset(original_dataset: Dataset, embedder: ImageEmbedder):
    """Function to generate the new dataset from the original one."""
    dataset_dict = {
        "id": [],
        "img_url": [],
        "file_name": [],
        "img_embedding": [],
    }

    for d in tqdm(original_dataset, total=len(original_dataset)):
        try:
            img = download_image(d["img_url"])

        except PIL.UnidentifiedImageError:
            logger.error(f"Image at {d['img_url']} not found")
            continue

        img_embedding = embedder(img)
        dataset_dict["img_embedding"].append(img_embedding.squeeze().tolist())

        photo_id = int(d["img_url"].split("/")[-2])
        dataset_dict["id"].append(photo_id)
        dataset_dict["img_url"].append(d["img_url"])
        dataset_dict["file_name"].append(d["file_name"])

    logger.info(f"Embedded {len(dataset_dict['id'])} images.")

    dataset = Dataset.from_dict(
        dataset_dict,
        features=Features(
            {
                # `id` column to be of type int64
                "id": Value("int64"),
                "img_url": Value("string"),
                "file_name": Value("string"),
                # `img_embedding` column is of type datasets.List[float32]
                "img_embedding": List(feature=Value("float32"), length=embedder.embedding_dim),
            },
        ),
    )

    return dataset


def main():
    """Main runner."""
    # Check if dataset already exists
    dataset_path = Path("data/inat_clip-vit-base-patch16")

    logger.info("Loading from disk")
    original_dataset = load_from_disk("data/inat_tiny_url_filtered_siglip_vit_so400_14_384")

    logger.info("Create embedder")
    embedder = ImageEmbedder(model_id="openai/clip-vit-base-patch16")

    dataset = generate_dataset(original_dataset, embedder)
    dataset.save_to_disk(dataset_path)


if __name__ == "__main__":
    main()
