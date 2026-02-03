"""Shared fixtures for ingestion tests."""

from unittest.mock import MagicMock, patch

import pytest

from config import RayJobConfig


@pytest.fixture
def ray_job_config() -> RayJobConfig:
    """Create a RayJobConfig for testing."""
    return RayJobConfig(
        ray_address="ray://localhost:10001",
        ray_namespace="test-namespace",
        runtime_env={"pip": ["requests"]},
    )


@pytest.fixture
def mock_ray():
    """Provide the mocked ray module."""
    import sys

    return sys.modules["ray"]
