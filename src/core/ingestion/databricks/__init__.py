"""Databricks ingestion entrypoints.

This package contains Databricks-specific wrappers for running ingestion
pipelines in a Databricks Job context.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = frozenset(
    {
        "process_s3_to_vector_dbs",
        "process_s3_images",
        "process_inat_images",
        "run_ingest",
        "run_ingest_image",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily expose Databricks submodules for patching/import convenience."""
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


__all__ = [
    "process_inat_images",
    "process_s3_images",
    "process_s3_to_vector_dbs",
    "run_ingest",
    "run_ingest_image",
]
