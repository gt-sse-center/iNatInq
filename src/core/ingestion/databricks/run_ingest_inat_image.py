"""Databricks entrypoint for iNaturalist image ingestion.

This wrapper converts Databricks python_params (KEY=VALUE) into environment
variables before invoking the iNaturalist image ingestion entrypoint.
"""

from __future__ import annotations

import sys

try:
    from core.ingestion.databricks.runtime import bootstrap_runtime
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from runtime import bootstrap_runtime


if __name__ == "__main__":
    bootstrap_runtime(sys.argv[1:], entrypoint_file=__file__)
    # Deferred import: only valid after sys.path is set from INATINQ_SRC_DIR.
    from core.ingestion.databricks.process_inat_images import main

    main()
