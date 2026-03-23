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
from core.ingestion.shared.env_keys import (
    AUTOLOADER_OPTIONAL_ENV_KEYS as _AUTOLOADER_OPTIONAL_ENV_KEYS,
    AUTOLOADER_REQUIRED_ENV_KEYS as _AUTOLOADER_REQUIRED_ENV_KEYS,
    CDC_PROGRESS_ENV_KEYS as _CDC_PROGRESS_ENV_KEYS,
    DLQ_ENV_KEYS as _DLQ_ENV_KEYS,
    IMAGE_OPTIONAL_ENV_KEYS as _IMAGE_OPTIONAL_ENV_KEYS,
    INAT_IMAGE_ENV_KEYS as _INAT_IMAGE_ENV_KEYS,
    OLLAMA_TUNING_ENV_KEYS as _OLLAMA_TIMEOUT_ENV_KEYS,
    S3_CONNECTION_ENV_KEYS as _S3_CONNECTION_ENV_KEYS,
    S3_TUNING_ENV_KEYS as _S3_TUNING_ENV_KEYS,
    VECTOR_ENV_KEYS as _VECTOR_ENV_KEYS,
    VECTOR_TIMEOUT_ENV_KEYS as _VECTOR_TIMEOUT_ENV_KEYS,
)


def _passthrough_env_vars(
    env_vars: dict[str, str],
    *key_groups: Iterable[str],
    overwrite: bool = True,
) -> None:
    """Copy selected environment variables into env_vars.

    Args:
        env_vars: Target mapping to update in-place.
        *key_groups: One or more iterables of environment variable names.
        overwrite: Whether to overwrite existing keys in env_vars.
    """
    for key_group in key_groups:
        for key in key_group:
            value = os.getenv(key)
            if not value:
                continue
            if overwrite or key not in env_vars:
                env_vars[key] = value


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


def build_image_ingestion_env(
    *,
    namespace: str,
    s3_endpoint: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_bucket: str,
    s3_prefix: str,
    embedding_config: EmbeddingConfig,
    collection: str,
    pull_from_dlq: bool,
    extra_env_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Build env-style params for image ingestion jobs (Ray/Databricks).

    Args:
        namespace: Kubernetes namespace (used for service discovery).
        s3_endpoint: S3/MinIO endpoint URL.
        s3_access_key_id: S3 access key.
        s3_secret_access_key: S3 secret key.
        s3_bucket: S3 bucket name.
        s3_prefix: S3 prefix to filter objects.
        embedding_config: Image embedding provider configuration.
        collection: Vector DB base collection name.
        pull_from_dlq: Whether the pipeline should process dead letter queue entries
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
        "EMBEDDING_PROVIDER": embedding_config.provider_type,
        "CLIP_TIMEOUT": str(embedding_config.clip_timeout),
        "CLIP_CIRCUIT_BREAKER_THRESHOLD": str(embedding_config.clip_circuit_breaker_threshold),
        "CLIP_CIRCUIT_BREAKER_TIMEOUT": str(embedding_config.clip_circuit_breaker_timeout),
        "CLIP_MAX_BATCH_SIZE": str(embedding_config.clip_max_batch_size),
        "IMAGE_BATCH_SIZE": str(embedding_config.image_batch_size),
        "IMAGE_MAX_SIZE_MB": str(embedding_config.image_max_size_mb),
        "IMAGE_TARGET_SIZE": str(embedding_config.image_target_size),
        "PULL_FROM_DLQ": str(pull_from_dlq),
    }

    if embedding_config.clip_url:
        env_vars["CLIP_URL"] = embedding_config.clip_url
    if embedding_config.clip_model:
        env_vars["CLIP_MODEL"] = embedding_config.clip_model
    if embedding_config.clip_vector_size is not None:
        env_vars["CLIP_VECTOR_SIZE"] = str(embedding_config.clip_vector_size)
    if embedding_config.infinity_url:
        env_vars["INFINITY_URL"] = embedding_config.infinity_url
    if embedding_config.infinity_model:
        env_vars["INFINITY_MODEL"] = embedding_config.infinity_model
    if embedding_config.infinity_vector_size is not None:
        env_vars["INFINITY_VECTOR_SIZE"] = str(embedding_config.infinity_vector_size)
    env_vars["INFINITY_TIMEOUT"] = str(embedding_config.infinity_timeout)

    _passthrough_env_vars(
        env_vars,
        _VECTOR_ENV_KEYS,
        _VECTOR_TIMEOUT_ENV_KEYS,
        _S3_TUNING_ENV_KEYS,
        _OLLAMA_TIMEOUT_ENV_KEYS,
        _INAT_IMAGE_ENV_KEYS,
        _DLQ_ENV_KEYS,
    )
    _passthrough_env_vars(env_vars, _IMAGE_OPTIONAL_ENV_KEYS, overwrite=False)

    if extra_env_keys:
        _passthrough_env_vars(env_vars, extra_env_keys)

    return env_vars


def build_inat_image_ingestion_env(
    *,
    namespace: str,
    embedding_config: EmbeddingConfig,
    collection: str,
    extra_env_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Build env-style params for iNaturalist image ingestion jobs.

    Unlike S3 image ingestion, this does not require MinIO/S3 connection
    parameters. Metadata and image source behavior is driven by INAT_* vars.
    """
    env_vars = {
        "K8S_NAMESPACE": namespace,
        "VECTOR_DB_COLLECTION": collection,
        "EMBEDDING_PROVIDER": embedding_config.provider_type,
        "CLIP_TIMEOUT": str(embedding_config.clip_timeout),
        "CLIP_CIRCUIT_BREAKER_THRESHOLD": str(embedding_config.clip_circuit_breaker_threshold),
        "CLIP_CIRCUIT_BREAKER_TIMEOUT": str(embedding_config.clip_circuit_breaker_timeout),
        "CLIP_MAX_BATCH_SIZE": str(embedding_config.clip_max_batch_size),
        "IMAGE_BATCH_SIZE": str(embedding_config.image_batch_size),
        "IMAGE_MAX_SIZE_MB": str(embedding_config.image_max_size_mb),
        "IMAGE_TARGET_SIZE": str(embedding_config.image_target_size),
    }

    if embedding_config.clip_url:
        env_vars["CLIP_URL"] = embedding_config.clip_url
    if embedding_config.clip_model:
        env_vars["CLIP_MODEL"] = embedding_config.clip_model
    if embedding_config.clip_vector_size is not None:
        env_vars["CLIP_VECTOR_SIZE"] = str(embedding_config.clip_vector_size)
    if embedding_config.infinity_url:
        env_vars["INFINITY_URL"] = embedding_config.infinity_url
    if embedding_config.infinity_model:
        env_vars["INFINITY_MODEL"] = embedding_config.infinity_model
    if embedding_config.infinity_vector_size is not None:
        env_vars["INFINITY_VECTOR_SIZE"] = str(embedding_config.infinity_vector_size)
    env_vars["INFINITY_TIMEOUT"] = str(embedding_config.infinity_timeout)

    _passthrough_env_vars(
        env_vars,
        _VECTOR_ENV_KEYS,
        _VECTOR_TIMEOUT_ENV_KEYS,
        _OLLAMA_TIMEOUT_ENV_KEYS,
        _INAT_IMAGE_ENV_KEYS,
    )
    _passthrough_env_vars(env_vars, _IMAGE_OPTIONAL_ENV_KEYS, overwrite=False)

    if extra_env_keys:
        _passthrough_env_vars(env_vars, extra_env_keys)

    return env_vars


def build_s3_autoloader_env(
    *,
    namespace: str,
    extra_env_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Build env-style params for S3 Auto Loader ingestion jobs.

    Required Auto Loader keys are copied from the current process environment.
    """
    env_vars = {"K8S_NAMESPACE": namespace}
    _passthrough_env_vars(
        env_vars,
        _AUTOLOADER_REQUIRED_ENV_KEYS,
        _AUTOLOADER_OPTIONAL_ENV_KEYS,
        _S3_CONNECTION_ENV_KEYS,
        _S3_TUNING_ENV_KEYS,
    )

    # Normalize copied values; trim whitespace and drop empty values.
    for key in (
        *_AUTOLOADER_REQUIRED_ENV_KEYS,
        *_AUTOLOADER_OPTIONAL_ENV_KEYS,
        *_S3_CONNECTION_ENV_KEYS,
        *_S3_TUNING_ENV_KEYS,
    ):
        value = env_vars.get(key)
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            env_vars[key] = normalized
        else:
            env_vars.pop(key, None)

    missing = [key for key in _AUTOLOADER_REQUIRED_ENV_KEYS if key not in env_vars]
    if missing:
        raise ValueError(f"Missing required Auto Loader config: {', '.join(missing)}")

    minio_group = {
        "S3_ENDPOINT": env_vars.get("S3_ENDPOINT"),
        "S3_ACCESS_KEY_ID": env_vars.get("S3_ACCESS_KEY_ID"),
        "S3_SECRET_ACCESS_KEY": env_vars.get("S3_SECRET_ACCESS_KEY"),
    }
    if any(minio_group.values()) and not all(minio_group.values()):
        missing = [key for key, value in minio_group.items() if not value]
        raise ValueError(f"Missing required MinIO S3 config for Auto Loader: {', '.join(missing)}")

    if extra_env_keys:
        _passthrough_env_vars(env_vars, extra_env_keys)
    return env_vars


def build_s3_bronze_image_ingestion_env(
    *,
    namespace: str,
    s3_endpoint: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_bucket: str,
    embedding_config: EmbeddingConfig,
    collection: str,
    extra_env_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Build env-style params for Bronze-backed incremental image ingestion."""
    env_vars = build_image_ingestion_env(
        namespace=namespace,
        s3_endpoint=s3_endpoint,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        s3_bucket=s3_bucket,
        s3_prefix=(os.getenv("S3_PREFIX") or "").strip(),
        embedding_config=embedding_config,
        collection=collection,
        pull_from_dlq=False,
        extra_env_keys=None,
    )
    _passthrough_env_vars(
        env_vars,
        ("AUTOLOADER_BRONZE_TABLE",),
        _CDC_PROGRESS_ENV_KEYS,
    )

    if "AUTOLOADER_BRONZE_TABLE" not in env_vars:
        raise ValueError("Missing required Bronze CDC config: AUTOLOADER_BRONZE_TABLE")

    if extra_env_keys:
        _passthrough_env_vars(env_vars, extra_env_keys)
    return env_vars
