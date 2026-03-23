"""Databricks entrypoint for Bronze-backed incremental Ray image ingestion."""

from __future__ import annotations

import sys

try:
    from core.ingestion.databricks.runtime import bootstrap_runtime
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from runtime import bootstrap_runtime


def _entrypoint_file() -> str:
    """Resolve entrypoint path for script and notebook exec contexts."""
    return globals().get("__file__") or (sys.argv[0] if sys.argv else "run_ingest_image_from_bronze.py")


if __name__ == "__main__":
    bootstrap_runtime(sys.argv[1:], entrypoint_file=_entrypoint_file())
    # Deferred import: only valid after sys.path is set from INATINQ_SRC_DIR.
    from core.ingestion.databricks.process_s3_images_from_bronze import main

    main()
