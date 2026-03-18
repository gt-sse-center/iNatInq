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


def _ensure_ray_submodule_mocks() -> None:
    """Restore ray submodule mocks that _get_real_ray() may have cleared.

    _get_real_ray() deletes ALL ray.* entries from sys.modules when loading
    the real ray package.  Unit tests for the Databricks entrypoint need
    ray.util.spark to be mocked so the source module can be imported.
    """
    if "ray.util.spark" not in sys.modules:
        mock_spark = MagicMock()
        mock_spark.setup_ray_cluster = MagicMock(return_value=MagicMock())
        mock_spark.MAX_NUM_WORKER_NODES = 10
        sys.modules["ray.util.spark"] = mock_spark

    # If the databricks entrypoint was previously imported with the real
    # (and possibly broken) ray, clear the cached module so it can be
    # re-imported cleanly with the mocks.
    mod_name = "core.ingestion.databricks.process_s3_to_vector_dbs"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        if not isinstance(getattr(mod, "ray", None), MagicMock):
            del sys.modules[mod_name]


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
        _ensure_ray_submodule_mocks()
        _patch_ray_references(mock_ray)


def pytest_sessionfinish(session, exitstatus) -> None:
    """Restore real ray in sys.modules so ray's atexit handler can run cleanly."""
    try:
        # Ensure we have a real ray module loaded for atexit shutdown.
        real_ray = _get_real_ray()
    except Exception:
        # If real ray truly cannot be imported, remove mocks so the handler
        # either sees no ray module or handles the missing module gracefully.
        sys.modules.pop("ray", None)
        sys.modules.pop("ray.job_submission", None)
        return

    # Put the real ray back so its atexit handler sees a coherent module.
    sys.modules["ray"] = real_ray
    # Drop any mock-only submodules we may have registered.
    sys.modules.pop("ray.job_submission", None)
