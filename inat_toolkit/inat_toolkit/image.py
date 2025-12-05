"""Module with image utilities."""

import numpy as np
import requests
from loguru import logger
from PIL import Image

from inat_toolkit.embed import ImageEmbedder


@logger.catch
def download_image(url: str, timeout: int = 10) -> Image.Image:
    """Download the image at `url`."""

    resp = requests.get(url, stream=True, timeout=timeout)
    return Image.open(resp.raw)


def get_url(
    photo_id: list[object],
    extension: str = "jpeg",
    hostname: str = "inaturalist-open-data.s3.amazonaws.com",
    image_size: str = "medium",
) -> str:
    """Get the iNat image url from the CSV row."""
    return f"https://{hostname}/photos/{photo_id}/{image_size}.{extension}"


def get_image_embedding(
    csv_row: list[object],
    model_id: str = "openai/clip-vit-base-patch16",
    download_timeout: int = 10,
) -> np.ndarray | None:
    """Function to read the CSV row, download the image and generate the embedding.

    `download_timeout` specifies how long (in seconds) before the image download times out.

    Returns a vector of size (embedding_dim,) or None if the image is not found.

    The CSV should look like this:
    photo_uuid                           | photo_id | observation_uuid                     | observer_id | extension | license  | width | height | position
    8d6b2534-d30a-47a8-bc1c-986a21817997 | 21213    | 7ae155fc-f49e-4e4f-91c5-51e31e805478 | 516         | jpg       | CC-BY-NC | 584   | 389    | 0
    """

    photo_id = csv_row[1]
    extension = csv_row[4]
    url = get_url(photo_id=photo_id, extension=extension)

    embedder = ImageEmbedder(model_id=model_id)

    # try downloading the image
    img = download_image(url, timeout=download_timeout)
    if img is None:
        return None

    img_embedding = embedder(img)
    return img_embedding.squeeze()
