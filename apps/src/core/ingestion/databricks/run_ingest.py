"""Databricks entrypoint for Ray ingestion.

This wrapper converts Databricks python_params (KEY=VALUE) into environment
variables before invoking the Ray ingestion entrypoint.
"""

import os
import sys
from logging.config import dictConfig
from pathlib import Path

repo_src_env = os.getenv("INATINQ_SRC_DIR")
if repo_src_env:
    repo_src = Path(repo_src_env)
else:
    # Default to the local repo's src directory based on this file's location.
    repo_src = Path(__file__).resolve().parents[3]

if repo_src.exists():
    sys.path.insert(0, str(repo_src.resolve()))

from core.ingestion.databricks.process_s3_to_qdrant import main  # noqa: E402
from foundation.logger import LOGGING_CONFIG  # noqa: E402

dictConfig(LOGGING_CONFIG)


def _load_params(params: list[str]) -> None:
    """Load KEY=VALUE params into os.environ."""
    for item in params:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        os.environ[key] = value


if __name__ == "__main__":
    _load_params(sys.argv[1:])
    main()
