"""Fixtures for benchmark integration tests.

Provides path constants, dataset fixtures, CLI runner, and MinIO
image-loading for the INQUIRE benchmark test suite.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from clients.s3 import S3ClientWrapper
from core.benchmark.datasets.json_dataset import JSONDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

INQUIRE_FIXTURES_DIR = _PROJECT_ROOT / "syntheticdata" / "data" / "inquire" / "query_127"
INQUIRE_VAL_PATH = _PROJECT_ROOT / "benchmarks" / "inquire" / "inquire-val.json"
SAMPLE_GOLD_PATH = _PROJECT_ROOT / "benchmarks" / "sample" / "sample-gold.json"

INQUIRE_BUCKET = "inquire-train-data"

# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def inquire_images_dir() -> Path:
    """Validate that INQUIRE fixture images exist on disk.

    Skips the entire session if the fixture directory is missing or empty.
    """
    if not INQUIRE_FIXTURES_DIR.is_dir():
        pytest.skip(f"INQUIRE fixtures not found: {INQUIRE_FIXTURES_DIR}")
    jpgs = list(INQUIRE_FIXTURES_DIR.glob("*.jpg"))
    if not jpgs:
        pytest.skip(f"No .jpg files in {INQUIRE_FIXTURES_DIR}")
    return INQUIRE_FIXTURES_DIR


@pytest.fixture(scope="session")
def inquire_val_dataset() -> JSONDataset:
    """Load the INQUIRE validation dataset from benchmarks/."""
    if not INQUIRE_VAL_PATH.exists():
        pytest.skip(f"inquire-val.json not found: {INQUIRE_VAL_PATH}")
    return JSONDataset.from_file(INQUIRE_VAL_PATH)


@pytest.fixture(scope="session")
def sample_gold_dataset() -> JSONDataset:
    """Load the sample gold-standard dataset from benchmarks/."""
    if not SAMPLE_GOLD_PATH.exists():
        pytest.skip(f"sample-gold.json not found: {SAMPLE_GOLD_PATH}")
    return JSONDataset.from_file(SAMPLE_GOLD_PATH)


# ---------------------------------------------------------------------------
# MinIO image-loading fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def inquire_bucket(
    minio_client: S3ClientWrapper,
    inquire_images_dir: Path,
) -> str:
    """Create the ``inquire-train-data`` bucket and upload all 52 fixture images.

    Uses the shared ``minio_client`` from ``tests/integration/clients/conftest.py``.
    Returns the bucket name for downstream tests.
    """
    minio_client.ensure_bucket(INQUIRE_BUCKET)

    jpgs = sorted(inquire_images_dir.glob("*.jpg"))
    logger.info("Uploading %d fixture images to bucket %s", len(jpgs), INQUIRE_BUCKET)

    for jpg in jpgs:
        image_id = jpg.stem  # e.g. "4217699"
        minio_client.put_object(
            bucket=INQUIRE_BUCKET,
            key=image_id,
            body=jpg.read_bytes(),
        )

    logger.info("Finished uploading %d images", len(jpgs))
    return INQUIRE_BUCKET


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner():
    """Create a Typer CLI test runner."""
    from typer.testing import CliRunner

    return CliRunner()
