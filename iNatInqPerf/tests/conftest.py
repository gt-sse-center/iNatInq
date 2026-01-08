"""Pytest configuration for shared test setup."""

import os
import sys
from functools import partialmethod
from pathlib import Path

import pytest
from loguru import logger
from tqdm import tqdm
import yaml

# Disable tqdm bars in tests
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@pytest.fixture(name="source_dir")
def source_dir_fixture():
    """A fixture for the source directory."""
    # Add the source directory to the fake filesystem so everything can download correctly.
    source_dir = Path(__file__).parent.parent
    return source_dir


@pytest.fixture(name="fixtures_dir")
def fixtures_dir_fixture(source_dir):
    return source_dir / "tests" / "fixtures"


@pytest.fixture(name="config_yaml")
def config_yaml_fixture(fixtures_dir):
    """The config as a yaml file within a fake source directory."""
    config_file = fixtures_dir / "inquire_test.yaml"

    return config_file


@pytest.fixture(name="benchmark_yaml")
def benchmark_config_fixture(config_yaml: Path):
    with config_yaml.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(name="qdrant_yaml")
def qdrant_config_fixture(fixtures_dir):
    config_yaml = fixtures_dir / "inquire_qdrant.yaml"
    with config_yaml.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(name="weaviate_yaml")
def weaviate_config_fixture(fixtures_dir):
    config_yaml = fixtures_dir / "inquire_weaviate.yaml"
    with config_yaml.open() as f:
        return yaml.safe_load(f)


# Set logging level to CRITICAL so it doesn't show
# in test output but is still captured for testing.
logger.remove()
logger.add(sys.stderr, level="CRITICAL")
