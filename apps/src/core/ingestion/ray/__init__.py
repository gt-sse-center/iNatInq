"""Ray entrypoints for ML pipeline processing.

This package contains entrypoints for running ingestion jobs on local Ray clusters.
The actual @ray.remote task functions live in core.ingestion.tasks.

Note: Imports are kept minimal to avoid triggering dictConfig at import time.
Use explicit imports from submodules when needed.
"""

__all__ = ["process_s3_object_ray", "run_ray_job"]


def __getattr__(name: str):
    """Lazy import to avoid triggering dictConfig at import time.

    This prevents process_s3_to_vector_dbs.py from being imported (which calls
    dictConfig) until the symbol is actually used.
    """
    if name == "run_ray_job":
        from .process_s3_to_vector_dbs import main as run_ray_job

        return run_ray_job
    if name == "process_s3_object_ray":
        from core.ingestion.tasks.text_processing import process_s3_object_ray

        return process_s3_object_ray
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
