"""Unit tests for the image module."""

import PIL
import pytest

from inat_toolkit import image


@pytest.fixture(name="csv_row")
def csv_row_fixture():
    return [
        "8d6b2534-d30a-47a8-bc1c-986a21817997",
        21213,
        "7ae155fc-f49e-4e4f-91c5-51e31e805478",
        516,
        "jpg",
        "CC-BY-NC",
        584,
        389,
        0,
    ]


def test_get_url(csv_row):
    photo_id = csv_row[1]
    extension = csv_row[4]
    url = image.get_url(photo_id=photo_id, extension=extension)
    assert url == "https://inaturalist-open-data.s3.amazonaws.com/photos/21213/medium.jpg"


def test_download_image(csv_row):
    photo_id = csv_row[1]
    extension = csv_row[4]
    url = image.get_url(photo_id=photo_id, extension=extension)
    img = image.download_image(url)
    width, height = img.size

    # regression
    assert width == 500
    assert height == 333


def test_download_invalid_image():
    url = image.get_url(photo_id=17, extension="jpg")
    with pytest.raises(PIL.UnidentifiedImageError):
        image.download_image(url)


def test_get_image_embedding(csv_row):
    embedding = image.get_image_embedding(csv_row)
    assert embedding.shape == (512,)
