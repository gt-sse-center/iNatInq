"""Client for reading iNaturalist open-data metadata and downloading photos.

This client provides:
- Streaming CSV/TSV metadata parsing (including .gz inputs)
- Photo URL construction from metadata fields (photo_id + extension)
- Image byte download with retry session and circuit-breaker protection
"""

from __future__ import annotations

import csv
import gzip
import io
import itertools
from urllib.parse import urlparse
from typing import TYPE_CHECKING

import attrs
import requests

from core.exceptions import UpstreamError
from foundation.circuit_breaker import with_circuit_breaker
from foundation.http import create_retry_session

from .mixins import CircuitBreakerMixin, LoggerMixin

if TYPE_CHECKING:
    from collections.abc import Iterator

SUPPORTED_IMAGE_SIZES = frozenset({"square", "small", "medium", "large", "original"})
SUPPORTED_METADATA_DATASETS = frozenset(
    {"observations", "photos", "taxa", "taxon_names", "identifications", "observers"}
)
INAT_OPEN_DATA_DOCS_URL = "https://github.com/inaturalist/inaturalist-open-data/tree/documentation"


@attrs.define(frozen=True, slots=True)
class INaturalistPhotoRecord:
    """Parsed iNaturalist photo metadata row."""

    photo_id: str
    extension: str
    photo_url: str
    source_row: dict[str, str]


@attrs.define(frozen=False, slots=True)
class INaturalistOpenDataClient(CircuitBreakerMixin, LoggerMixin):
    """Client for iNaturalist open-data photo metadata and image downloads."""

    photo_base_url: str = attrs.field(default="https://inaturalist-open-data.s3.amazonaws.com/photos")
    metadata_base_url: str = attrs.field(default="https://inaturalist-open-data.s3.amazonaws.com")
    timeout_s: int = attrs.field(default=120)
    circuit_breaker_failure_threshold: int = attrs.field(default=5)
    circuit_breaker_recovery_timeout_s: int = attrs.field(default=30)

    _session: requests.Session | None = attrs.field(init=False, default=None)

    def _circuit_breaker_config(self) -> tuple[str, int, int]:
        """Return circuit breaker configuration for iNaturalist open data."""
        return (
            "inaturalist_open_data",
            self.circuit_breaker_failure_threshold,
            self.circuit_breaker_recovery_timeout_s,
        )

    def __attrs_post_init__(self) -> None:
        """Initialize HTTP session and circuit breaker."""
        if self._session is None:
            self._session = create_retry_session()
        self._init_circuit_breaker()

    @property
    def session(self) -> requests.Session:
        """Get requests session, creating one when necessary."""
        if self._session is None:
            self._session = create_retry_session()
        assert self._session is not None
        return self._session

    def set_session(self, session: requests.Session) -> None:
        """Set custom HTTP session for testing or pooling control."""
        if self._session is not None:
            self._session.close()
        self._session = session

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def build_photo_url(self, *, photo_id: str, extension: str, size: str = "medium") -> str:
        """Build a photo URL from iNaturalist metadata fields.

        URL pattern:
            https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{extension}
        """
        resolved_size = size.strip().lower()
        if resolved_size not in SUPPORTED_IMAGE_SIZES:
            msg = f"Unsupported image size '{size}'. Supported: {sorted(SUPPORTED_IMAGE_SIZES)}"
            raise ValueError(msg)

        resolved_photo_id = photo_id.strip()
        if not resolved_photo_id:
            raise ValueError("photo_id cannot be empty")

        resolved_extension = extension.strip().lower().lstrip(".")
        if not resolved_extension:
            raise ValueError("extension cannot be empty")

        return f"{self.photo_base_url}/{resolved_photo_id}/{resolved_size}.{resolved_extension}"

    def build_metadata_url(self, *, dataset: str = "photos", compressed: bool = True) -> str:
        """Build metadata file URL from dataset name.

        Example:
            https://inaturalist-open-data.s3.amazonaws.com/photos.csv.gz
        """
        resolved_dataset = dataset.strip().lower()
        if resolved_dataset not in SUPPORTED_METADATA_DATASETS:
            msg = (
                f"Unsupported dataset '{dataset}'. Supported: {sorted(SUPPORTED_METADATA_DATASETS)}"
            )
            raise ValueError(msg)

        extension = "csv.gz" if compressed else "csv"
        return f"{self.metadata_base_url.rstrip('/')}/{resolved_dataset}.{extension}"

    def build_metadata_s3_uri(self, *, dataset: str = "photos", compressed: bool = True) -> str:
        """Build metadata S3 URI from dataset name.

        Example:
            s3://inaturalist-open-data/photos.csv.gz
        """
        resolved_dataset = dataset.strip().lower()
        if resolved_dataset not in SUPPORTED_METADATA_DATASETS:
            msg = (
                f"Unsupported dataset '{dataset}'. Supported: {sorted(SUPPORTED_METADATA_DATASETS)}"
            )
            raise ValueError(msg)

        extension = "csv.gz" if compressed else "csv"
        return f"s3://inaturalist-open-data/{resolved_dataset}.{extension}"

    def latest_metadata_archive_url(self) -> str:
        """Return URL for the latest metadata tar.gz archive."""
        return f"{self.metadata_base_url.rstrip('/')}/metadata/inaturalist-open-data-latest.tar.gz"

    @staticmethod
    def _resolve_metadata_source_url(metadata_url: str) -> str:
        """Resolve metadata source URL to an HTTP(S) streamable endpoint.

        Accepts:
          - https://... / http://...
          - s3://bucket/key (converted to https://bucket.s3.amazonaws.com/key)
        """
        resolved = metadata_url.strip()
        if not resolved:
            raise ValueError("metadata_url cannot be empty")

        if resolved.startswith("s3://"):
            parsed = urlparse(resolved)
            bucket = parsed.netloc.strip()
            key = parsed.path.lstrip("/")
            if not bucket or not key:
                raise ValueError(f"Invalid S3 metadata URI: {metadata_url}")
            return f"https://{bucket}.s3.amazonaws.com/{key}"

        return resolved

    @with_circuit_breaker("inaturalist_open_data")
    def download_image(self, photo_url: str) -> bytes:
        """Download image bytes from a photo URL."""
        try:
            response = self.session.get(photo_url, timeout=self.timeout_s)
            response.raise_for_status()
            image_bytes = response.content
        except requests.RequestException as e:
            msg = f"Failed to download iNaturalist image from {photo_url}: {e}"
            raise UpstreamError(msg) from e

        if not image_bytes:
            raise UpstreamError(f"Downloaded empty image content from {photo_url}")
        return image_bytes

    @with_circuit_breaker("inaturalist_open_data")
    def _fetch_metadata_response(self, metadata_url: str) -> requests.Response:
        """Fetch metadata file response in streaming mode."""
        resolved_url = self._resolve_metadata_source_url(metadata_url)
        try:
            response = self.session.get(resolved_url, timeout=self.timeout_s, stream=True)
            response.raise_for_status()
            return response
        except (requests.RequestException, ValueError) as e:
            msg = f"Failed to fetch iNaturalist metadata from {metadata_url}: {e}"
            raise UpstreamError(msg) from e

    @staticmethod
    def _detect_delimiter(header_line: str) -> str:
        """Detect delimiter from header line."""
        try:
            dialect = csv.Sniffer().sniff(header_line, delimiters=",\t")
            return dialect.delimiter
        except csv.Error:
            return "\t"

    @staticmethod
    def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
        """Return first non-empty value for candidate keys."""
        for key in keys:
            value = row.get(key, "")
            if value:
                resolved = value.strip()
                if resolved:
                    return resolved
        return None

    def _row_to_photo_record(
        self,
        row: dict[str, str],
        *,
        size: str,
    ) -> INaturalistPhotoRecord | None:
        """Convert CSV row to photo record; return None when required fields are missing."""
        photo_id = self._first_non_empty(row, ("photo_id", "id"))
        extension = self._first_non_empty(row, ("extension", "photo_extension", "ext"))
        if not photo_id or not extension:
            return None

        try:
            photo_url = self.build_photo_url(photo_id=photo_id, extension=extension, size=size)
        except ValueError:
            return None

        normalized_row = {str(k): (v if isinstance(v, str) else "") for k, v in row.items() if k}
        return INaturalistPhotoRecord(
            photo_id=photo_id,
            extension=extension.lstrip(".").lower(),
            photo_url=photo_url,
            source_row=normalized_row,
        )

    def iter_photo_records(
        self,
        *,
        metadata_url: str,
        size: str = "medium",
        delimiter: str | None = None,
        max_rows: int | None = None,
    ) -> Iterator[INaturalistPhotoRecord]:
        """Iterate parsed photo records from an iNaturalist metadata file URL.

        Supports `.csv`, `.tsv`, and gz-compressed variants.
        """
        response = self._fetch_metadata_response(metadata_url)
        text_stream: io.TextIOWrapper | None = None
        try:
            binary_stream: io.BufferedIOBase | io.BytesIO
            if metadata_url.lower().endswith(".gz"):
                binary_stream = gzip.GzipFile(fileobj=response.raw)
            else:
                binary_stream = response.raw

            text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8", newline="")
            header_line = text_stream.readline()
            if not header_line:
                return

            resolved_delimiter = delimiter or self._detect_delimiter(header_line)
            line_iter = itertools.chain([header_line], text_stream)
            reader = csv.DictReader(line_iter, delimiter=resolved_delimiter)

            emitted = 0
            for row in reader:
                if not row:
                    continue

                normalized = {str(k): (v if isinstance(v, str) else "") for k, v in row.items() if k}
                record = self._row_to_photo_record(normalized, size=size)
                if record is None:
                    continue

                yield record
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    break
        finally:
            if text_stream is not None:
                text_stream.close()
            response.close()

    def read_photo_records(
        self,
        *,
        metadata_url: str,
        size: str = "medium",
        delimiter: str | None = None,
        max_rows: int | None = None,
    ) -> list[INaturalistPhotoRecord]:
        """Read all photo records from metadata URL into a list."""
        return list(
            self.iter_photo_records(
                metadata_url=metadata_url,
                size=size,
                delimiter=delimiter,
                max_rows=max_rows,
            )
        )
