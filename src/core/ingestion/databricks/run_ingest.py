"""Databricks entrypoint for Ray ingestion.

This wrapper converts Databricks python_params (KEY=VALUE) into environment
variables before invoking the Ray ingestion entrypoint.
"""

import os
import sys

try:
    from core.ingestion.databricks.runtime import apply_python_params, bootstrap_runtime
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from runtime import apply_python_params, bootstrap_runtime


def _load_params(params: list[str]) -> None:
    """Backward-compatible alias for Databricks python_params loading."""
    apply_python_params(params)


if __name__ == "__main__":
    bootstrap_runtime(sys.argv[1:], entrypoint_file=__file__)
    # Deferred import: only valid after sys.path is set from INATINQ_SRC_DIR.
    from core.ingestion.databricks.process_s3_to_vector_dbs import main

    main()
