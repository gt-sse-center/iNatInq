"""Unit tests for clients.inaturalist_open_data module."""

from __future__ import annotations

import gzip
import io
from unittest.mock import MagicMock

import pybreaker
import pytest
import requests

from clients.inaturalist_open_data import (
    INaturalistOpenDataClient,
    INaturalistPhotoRecord,
    SUPPORTED_IMAGE_SIZES,
)
from core.exceptions import UpstreamError


def _make_metadata_response(content: bytes) -> MagicMock:
    """Create mock streaming response for metadata downloads."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.raw = io.BytesIO(content)
    response.close = MagicMock()
    return response


class TestINaturalistOpenDataClientInit:
    """Test suite for INaturalistOpenDataClient initialization."""

    def test_creates_client_with_defaults(self) -> None:
        client = INaturalistOpenDataClient()

        assert client.photo_base_url == "https://inaturalist-open-data.s3.amazonaws.com/photos"
        assert client.timeout_s == 120
        assert client._session is not None
        assert isinstance(client._breaker, pybreaker.CircuitBreaker)
        assert client._breaker.name == "inaturalist_open_data"


class TestINaturalistOpenDataClientPhotoUrl:
    """Test suite for photo URL construction."""

    def test_build_photo_url(self) -> None:
        client = INaturalistOpenDataClient()

        url = client.build_photo_url(photo_id="12345", extension="jpg", size="medium")

        assert url == "https://inaturalist-open-data.s3.amazonaws.com/photos/12345/medium.jpg"

    def test_build_photo_url_normalizes_extension_and_size(self) -> None:
        client = INaturalistOpenDataClient()

        url = client.build_photo_url(photo_id="12345", extension=".JPG", size="LARGE")

        assert url == "https://inaturalist-open-data.s3.amazonaws.com/photos/12345/large.jpg"

    def test_build_photo_url_raises_for_unsupported_size(self) -> None:
        client = INaturalistOpenDataClient()

        with pytest.raises(ValueError, match="Unsupported image size"):
            client.build_photo_url(photo_id="12345", extension="jpg", size="thumbnail")

        assert "medium" in SUPPORTED_IMAGE_SIZES

    def test_build_photo_url_raises_for_empty_photo_id(self) -> None:
        client = INaturalistOpenDataClient()

        with pytest.raises(ValueError, match="photo_id cannot be empty"):
            client.build_photo_url(photo_id=" ", extension="jpg")


class TestINaturalistOpenDataClientMetadataParsing:
    """Test suite for metadata parsing behavior."""

    def test_read_photo_records_parses_tsv_metadata(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        payload = (
            "photo_id\textension\tlicense\n"
            "111\tjpg\tcc-by\n"
            "222\tjpeg\tcc0\n"
        ).encode("utf-8")
        mock_session.get.return_value = _make_metadata_response(payload)

        records = client.read_photo_records(
            metadata_url="https://example.com/photos.tsv",
            size="small",
        )

        assert len(records) == 2
        assert records[0].photo_id == "111"
        assert records[0].extension == "jpg"
        assert records[0].photo_url.endswith("/111/small.jpg")
        assert records[1].photo_url.endswith("/222/small.jpeg")

        mock_session.get.assert_called_once_with(
            "https://example.com/photos.tsv",
            timeout=120,
            stream=True,
        )

    def test_read_photo_records_parses_comma_delimited_metadata(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        payload = "photo_id,extension\n333,png\n".encode("utf-8")
        mock_session.get.return_value = _make_metadata_response(payload)

        records = client.read_photo_records(metadata_url="https://example.com/photos.csv")

        assert len(records) == 1
        assert records[0].photo_url.endswith("/333/medium.png")

    def test_read_photo_records_parses_gzip_metadata(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        compressed = gzip.compress("photo_id\textension\n444\tgif\n".encode("utf-8"))
        mock_session.get.return_value = _make_metadata_response(compressed)

        records = client.read_photo_records(metadata_url="https://example.com/photos.csv.gz")

        assert len(records) == 1
        assert records[0].photo_id == "444"
        assert records[0].photo_url.endswith("/444/medium.gif")

    def test_read_photo_records_skips_rows_missing_required_fields(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        payload = (
            "photo_id\textension\n"
            "555\tjpg\n"
            "\tjpg\n"
            "666\t\n"
            "777\twebp\n"
        ).encode("utf-8")
        mock_session.get.return_value = _make_metadata_response(payload)

        records = client.read_photo_records(metadata_url="https://example.com/photos.tsv")

        assert [record.photo_id for record in records] == ["555", "777"]

    def test_read_photo_records_respects_max_rows(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        payload = (
            "photo_id\textension\n"
            "101\tjpg\n"
            "102\tjpg\n"
            "103\tjpg\n"
        ).encode("utf-8")
        mock_session.get.return_value = _make_metadata_response(payload)

        records = client.read_photo_records(metadata_url="https://example.com/photos.tsv", max_rows=2)

        assert len(records) == 2
        assert records[0].photo_id == "101"
        assert records[1].photo_id == "102"

    def test_iter_photo_records_yields_typed_records(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        payload = "photo_id\textension\n999\tjpg\n".encode("utf-8")
        mock_session.get.return_value = _make_metadata_response(payload)

        records = list(client.iter_photo_records(metadata_url="https://example.com/photos.tsv"))

        assert len(records) == 1
        assert isinstance(records[0], INaturalistPhotoRecord)

    def test_read_photo_records_raises_upstream_error_on_metadata_request_error(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)
        mock_session.get.side_effect = requests.RequestException("boom")

        with pytest.raises(UpstreamError, match="Failed to fetch iNaturalist metadata"):
            client.read_photo_records(metadata_url="https://example.com/photos.tsv")


class TestINaturalistOpenDataClientDownloadImage:
    """Test suite for image download behavior."""

    def test_download_image_returns_bytes(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.content = b"image-bytes"
        mock_session.get.return_value = response

        image = client.download_image("https://example.com/photos/1/medium.jpg")

        assert image == b"image-bytes"
        mock_session.get.assert_called_once_with(
            "https://example.com/photos/1/medium.jpg",
            timeout=120,
        )

    def test_download_image_raises_upstream_error_on_request_failure(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)
        mock_session.get.side_effect = requests.RequestException("timeout")

        with pytest.raises(UpstreamError, match="Failed to download iNaturalist image"):
            client.download_image("https://example.com/photos/1/medium.jpg")

    def test_download_image_raises_upstream_error_on_empty_content(self) -> None:
        client = INaturalistOpenDataClient()
        mock_session = MagicMock(spec=requests.Session)
        client.set_session(mock_session)

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.content = b""
        mock_session.get.return_value = response

        with pytest.raises(UpstreamError, match="Downloaded empty image content"):
            client.download_image("https://example.com/photos/1/medium.jpg")
