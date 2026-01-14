"""Spark job implementations for ML pipeline processing.

Note: Imports are kept minimal to avoid triggering dictConfig at import time.
Use explicit imports from submodules when needed.
"""

__all__ = ["create_spark_session", "load_local_to_s3", "process_s3_to_qdrant"]


def __getattr__(name: str):
    """Lazy import to avoid triggering dictConfig at import time.

    This prevents process_s3_to_qdrant.py and load_local_to_s3.py from being
    imported (which call dictConfig) until the symbol is actually used.
    """
    if name == "process_s3_to_qdrant":
        from .process_s3_to_qdrant import main as process_s3_to_qdrant

        return process_s3_to_qdrant
    if name == "load_local_to_s3":
        from .load_local_to_s3 import main as load_local_to_s3

        return load_local_to_s3
    if name == "create_spark_session":
        from .spark_config import create_spark_session

        return create_spark_session
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
