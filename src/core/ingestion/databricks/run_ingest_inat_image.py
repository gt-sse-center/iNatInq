"""Databricks entrypoint for iNaturalist image ingestion.

This wrapper converts Databricks python_params (KEY=VALUE) into environment
variables before invoking the iNaturalist image ingestion entrypoint.
"""

from __future__ import annotations

import os
import sys

try:
    from core.ingestion.databricks.runtime import bootstrap_runtime
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from runtime import bootstrap_runtime


def _entrypoint_file() -> str:
    """Resolve entrypoint path for script and notebook exec contexts."""
    explicit_path = os.getenv("DATABRICKS_ENTRYPOINT_FILE")
    if explicit_path:
        return explicit_path
    return globals().get("__file__") or (sys.argv[0] if sys.argv else "run_ingest_inat_image.py")


if __name__ == "__main__":
    bootstrap_runtime(sys.argv[1:], entrypoint_file=_entrypoint_file())
    # Deferred import: only valid after sys.path is set from INATINQ_SRC_DIR.
    from core.ingestion.databricks.process_inat_images import main

    main()
