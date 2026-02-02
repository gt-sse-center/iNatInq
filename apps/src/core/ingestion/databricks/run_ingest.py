"""Databricks entrypoint for Ray ingestion.

This wrapper converts Databricks python_params (KEY=VALUE) into environment
variables before invoking the Ray ingestion entrypoint.
"""

import os
import sys
from logging.config import dictConfig
from pathlib import Path


def _load_params(params: list[str]) -> None:
    """Load KEY=VALUE params into os.environ."""
    for item in params:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        os.environ[key] = value


def _bootstrap_runtime(params: list[str]) -> None:
    """Apply params before resolving repo paths and configuring logging."""
    _load_params(params)

    repo_src_env = os.getenv("INATINQ_SRC_DIR")
    # Default to the local repo's src directory based on this file's location.
    repo_src = Path(repo_src_env) if repo_src_env else Path(__file__).resolve().parents[3]

    if repo_src.exists():
        sys.path.insert(0, str(repo_src.resolve()))

    # Deferred import: we must load python_params first so LOGGING_CONFIG can
    # read env-driven settings without being imported too early.
    from foundation.logger import LOGGING_CONFIG

    dictConfig(LOGGING_CONFIG)


if __name__ == "__main__":
    _bootstrap_runtime(sys.argv[1:])
    # Deferred import: only valid after sys.path is set from INATINQ_SRC_DIR.
    from core.ingestion.databricks.process_s3_to_vector_dbs import main

    main()
