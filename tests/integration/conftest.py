"""Shared integration fixtures.

Expose client/service fixtures to all integration tests and ensure
the real Ray module is used (not the unit-test mock).
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

from tests.integration.clients.conftest import *  # noqa: F403


def _ensure_real_ray() -> None:
    ray_mod = sys.modules.get("ray")
    if not isinstance(ray_mod, MagicMock):
        return

    for name in list(sys.modules.keys()):
        if name == "ray" or name.startswith("ray."):
            del sys.modules[name]

    importlib.invalidate_caches()
    importlib.import_module("ray")

    for name in list(sys.modules.keys()):
        if name == "core.ingestion" or name.startswith("core.ingestion."):
            del sys.modules[name]


@pytest.fixture(autouse=True, scope="session")
def _integration_real_ray() -> None:
    _ensure_real_ray()
