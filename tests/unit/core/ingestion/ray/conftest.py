"""Shared fixtures for Ray ingestion tests."""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def reset_ray_mock():
    """Reset ray mock state before each test."""
    ray_mock = sys.modules["ray"]
    ray_mock.reset_mock()
    # Explicitly clear side_effect which takes precedence over return_value
    ray_mock.wait.side_effect = None
    ray_mock.get.side_effect = None
    ray_mock.init.side_effect = None
    ray_mock.shutdown.side_effect = None
    ray_mock.is_initialized.side_effect = None
    yield


@pytest.fixture
def mock_ray():
    """Provide the mocked ray module."""
    return sys.modules["ray"]
