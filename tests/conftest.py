"""Shared pytest configuration and fixtures.

This module provides global fixtures and configuration that are available
to all tests in the test suite.

Fixtures defined here are automatically available to all tests without explicit
import statements. Keep fixtures small, composable, and focused on
seup/teardown.  Do NOT put business logic in fixtures.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

_REAL_RAY: ModuleType | None = None
_MOCK_RAY: MagicMock | None = None


def _get_mock_ray() -> MagicMock:
    global _MOCK_RAY
    if _MOCK_RAY is None:
        from tests.unit import conftest as unit_conftest

        _MOCK_RAY = unit_conftest.mock_ray
    return _MOCK_RAY


def _get_real_ray() -> ModuleType:
    global _REAL_RAY
    if _REAL_RAY is None:
        for name in list(sys.modules.keys()):
            if name == "ray" or name.startswith("ray."):
                del sys.modules[name]
        importlib.invalidate_caches()
        _REAL_RAY = importlib.import_module("ray")
    return _REAL_RAY


def _patch_ray_references(ray_module: ModuleType | MagicMock) -> None:
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not (name.startswith("core.") or name.startswith("tests.")):
            continue
        if not hasattr(module, "ray"):
            continue
        try:
            current = getattr(module, "ray")
            if isinstance(current, ModuleType) and current.__name__.startswith(f"{name}."):
                # Avoid clobbering package submodules like core.ingestion.ray
                continue
            setattr(module, "ray", ray_module)
        except Exception:
            continue


def pytest_runtest_setup(item) -> None:
    path = str(item.fspath)
    if "/tests/integration/" in path:
        real_ray = _get_real_ray()
        sys.modules["ray"] = real_ray
        _patch_ray_references(real_ray)
    elif "/tests/unit/" in path:
        mock_ray = _get_mock_ray()
        sys.modules["ray"] = mock_ray
        sys.modules["ray.job_submission"] = mock_ray.job_submission
        _patch_ray_references(mock_ray)
