"""Shared Databricks runtime helpers.

This module centralizes:
- Databricks `python_params` (KEY=VALUE) parsing
- Runtime bootstrap (sys.path + logging setup) for entrypoint wrappers
"""

from __future__ import annotations

import os
import sys
from logging.config import dictConfig
from pathlib import Path


def apply_python_params(args: list[str]) -> None:
    """Apply Databricks python_params (KEY=VALUE) to the environment."""
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        if not key or not key.isupper() or not key.replace("_", "").isalnum():
            continue
        os.environ[key] = value


def bootstrap_runtime(params: list[str], *, entrypoint_file: str) -> None:
    """Apply params and configure import path/logging for Databricks wrappers."""
    apply_python_params(params)

    repo_src_env = os.getenv("INATINQ_SRC_DIR")
    repo_src = Path(repo_src_env) if repo_src_env else Path(entrypoint_file).resolve().parents[3]

    if repo_src.exists():
        sys.path.insert(0, str(repo_src.resolve()))

    # Deferred import: ensure env overrides are applied first.
    from foundation.logger import LOGGING_CONFIG

    dictConfig(LOGGING_CONFIG)

