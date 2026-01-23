"""Databricks entrypoint for Ray ingestion.

This wrapper converts Databricks python_params (KEY=VALUE) into environment
variables before invoking the Ray ingestion entrypoint.
"""

import os
import sys

from core.ingestion.databricks.process_s3_to_qdrant import main


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
