"""Shared helpers for building ingestion job parameters.

This module centralizes the construction of env-style parameters used by both
Ray Jobs and Databricks Jobs. It ensures parameter parity across submission
paths and avoids drift between service implementations.

Design goals:
    - **Single source of truth** for ingestion env parameters
    - **Provider-agnostic**: supports Qdrant and Weaviate config passthrough
    - **Minimal coupling**: callers can extend with extra env keys when needed
    - **Explicit typing**: returns a simple dict of string values
"""

import os
from collections.abc import Iterable

from config import EmbeddingConfig


_VECTOR_ENV_KEYS = (
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
    "WEAVIATE_GRPC_HOST",
)


def build_ingestion_env(
    *,
    namespace: str,
    s3_endpoint: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_bucket: str,
    s3_prefix: str,
    embedding_config: EmbeddingConfig,
    collection: str,
    extra_env_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Build env-style params for ingestion jobs (Ray/Databricks).

    This function returns a dictionary of environment variables that configure
    the ingestion pipeline entrypoint. It includes:
      - Namespace and S3 connection details
      - Vector DB collection name
      - Embedding provider settings
      - Optional vector DB credentials from the current process environment

    Args:
        namespace: Kubernetes namespace (used for service discovery).
        s3_endpoint: S3/MinIO endpoint URL.
        s3_access_key_id: S3 access key.
        s3_secret_access_key: S3 secret key.
        s3_bucket: S3 bucket name.
        s3_prefix: S3 prefix to filter objects.
        embedding_config: Embedding provider configuration.
        collection: Vector DB collection name.
        extra_env_keys: Optional iterable of env var names to pass through
            from the current process if set.

    Returns:
        Dictionary of environment variables suitable for Ray runtime_env or
        Databricks python_params conversion.
    """
    env_vars = {
        "K8S_NAMESPACE": namespace,
        "S3_PREFIX": s3_prefix,
        "S3_ENDPOINT": s3_endpoint,
        "S3_ACCESS_KEY_ID": s3_access_key_id,
        "S3_SECRET_ACCESS_KEY": s3_secret_access_key,
        "S3_BUCKET": s3_bucket,
        "VECTOR_DB_COLLECTION": collection,
        "EMBEDDING_PROVIDER_TYPE": embedding_config.provider_type,
    }

    if embedding_config.vector_size is not None:
        env_vars["EMBEDDING_VECTOR_SIZE"] = str(embedding_config.vector_size)
    if embedding_config.ollama_url:
        env_vars["OLLAMA_BASE_URL"] = embedding_config.ollama_url
    if embedding_config.ollama_model:
        env_vars["OLLAMA_MODEL"] = embedding_config.ollama_model

    for key in _VECTOR_ENV_KEYS:
        value = os.getenv(key)
        if value:
            env_vars[key] = value

    if extra_env_keys:
        for key in extra_env_keys:
            value = os.getenv(key)
            if value:
                env_vars[key] = value

    return env_vars


def add_ray_tuning_env(env_vars: dict[str, str]) -> None:
    """Add RAY_* tuning env vars from the current process.

    This is used for Databricks job submission to ensure Ray tuning parameters
    (e.g., worker counts, batch sizes) are passed through consistently.

    Args:
        env_vars: Mutable env var dict to be updated in-place.
    """
    for key, value in os.environ.items():
        if key.startswith("RAY_") and value:
            env_vars[key] = value
