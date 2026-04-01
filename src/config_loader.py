"""YAML configuration loader with environment layering.

Loads YAML config files and applies them as environment variable defaults,
bridging the YAML config system with the existing env-var-based config.

Layering order (later overrides earlier):
    hardcoded defaults < config.yaml < environments/{ENV}.yaml < secrets.yaml < env vars

Environment variables always win - YAML values are only applied when the
corresponding env var is not already set.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("pipeline.config_loader")

# Sentinel to track whether initialization has run
_initialized = False

# Default config directory relative to repo root
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


# =============================================================================
# Deep Merge
# =============================================================================


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins).

    Args:
        base: Base dictionary.
        overlay: Dictionary to merge on top. Values in overlay take precedence.

    Returns:
        New merged dictionary. Neither input is mutated.
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# YAML Loading
# =============================================================================


def _resolve_env_substitutions(value: str) -> str:
    """Resolve ${VAR_NAME} patterns in a string from os.environ.

    Unresolved variables (not set in environment) are left as-is.

    Args:
        value: String potentially containing ${VAR_NAME} patterns.

    Returns:
        String with resolved substitutions.
    """

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r"\$\{([^}]+)}", _replace, value)


def _resolve_all_substitutions(obj: Any) -> Any:
    """Recursively resolve ${VAR} substitutions in all string values."""
    if isinstance(obj, str):
        return _resolve_env_substitutions(obj)
    if isinstance(obj, dict):
        return {k: _resolve_all_substitutions(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_all_substitutions(item) for item in obj]
    return obj


def load_yaml_config(
    env: str | None = None,
    config_dir: Path | None = None,
) -> dict:
    """Load and merge YAML configuration files.

    Loads config.yaml as the base, optionally merges an environment overlay
    (environments/{env}.yaml) and secrets.yaml if they exist.

    Args:
        env: Environment name (e.g. "dev", "staging", "prod").
            Read from ENV or PIPELINE_ENV if not provided.
        config_dir: Path to the configs directory. Defaults to
            the repo's configs/ directory.

    Returns:
        Merged configuration dictionary.
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR

    # Load base config
    base_path = config_dir / "config.yaml"
    if not base_path.exists():
        logger.warning("Base config not found: %s", base_path)
        return {}

    with open(base_path) as f:
        config = yaml.safe_load(f) or {}

    # Merge environment overlay
    if env:
        env_path = config_dir / "environments" / f"{env}.yaml"
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f) or {}
            config = deep_merge(config, env_config)
            logger.debug("Merged environment overlay: %s", env_path)
        else:
            logger.debug("No environment overlay found: %s", env_path)

    # Merge secrets
    secrets_path = config_dir / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path) as f:
            secrets_config = yaml.safe_load(f) or {}
        config = deep_merge(config, secrets_config)
        logger.debug("Merged secrets: %s", secrets_path)

    # Resolve ${VAR} substitutions
    return _resolve_all_substitutions(config)


# =============================================================================
# YAML-to-Env-Var Mapping
# =============================================================================

YAML_TO_ENV_MAP: dict[str, str] = {
    # Storage
    "storage.endpoint": "S3_ENDPOINT",
    "storage.bucket": "S3_BUCKET",
    "storage.region": "S3_REGION",
    "storage.use_ssl": "S3_USE_SSL",
    "storage.path_style": "S3_PATH_STYLE",
    "storage.timeout_seconds": "S3_TIMEOUT",
    "storage.retry.max_attempts": "S3_MAX_RETRIES",
    "storage.retry.min_wait_seconds": "S3_RETRY_MIN_WAIT",
    "storage.retry.max_wait_seconds": "S3_RETRY_MAX_WAIT",
    "storage.circuit_breaker.failure_threshold": "S3_CIRCUIT_BREAKER_THRESHOLD",
    "storage.circuit_breaker.recovery_timeout": "S3_CIRCUIT_BREAKER_TIMEOUT",
    # Vector databases
    "vector_databases.search_provider": "VECTOR_DB_PROVIDER",
    "vector_databases.collection": "VECTOR_DB_COLLECTION",
    "vector_databases.qdrant.url": "QDRANT_URL",
    "vector_databases.qdrant.api_key": "QDRANT_API_KEY",
    "vector_databases.qdrant.timeout_seconds": "QDRANT_TIMEOUT",
    "vector_databases.qdrant.circuit_breaker.failure_threshold": "QDRANT_CIRCUIT_BREAKER_THRESHOLD",
    "vector_databases.qdrant.circuit_breaker.recovery_timeout": "QDRANT_CIRCUIT_BREAKER_TIMEOUT",
    # Embeddings
    "embeddings.provider": "EMBEDDING_PROVIDER",
    "embeddings.vector_size": "EMBEDDING_VECTOR_SIZE",
    "embeddings.openai.model": "OPENAI_MODEL",
    "embeddings.huggingface.model": "HUGGINGFACE_MODEL",
    "embeddings.huggingface.device": "HUGGINGFACE_DEVICE",
    "embeddings.sagemaker.endpoint": "SAGEMAKER_ENDPOINT",
    "embeddings.sagemaker.region": "SAGEMAKER_REGION",
    # Image embeddings
    "image_embeddings.model": "CLIP_MODEL",
    "image_embeddings.vector_size": "CLIP_VECTOR_SIZE",
    "image_embeddings.url": "CLIP_URL",
    "image_embeddings.timeout_seconds": "CLIP_TIMEOUT",
    "image_embeddings.max_batch_size": "CLIP_MAX_BATCH_SIZE",
    "image_embeddings.preprocessing.target_size": "IMAGE_TARGET_SIZE",
    "image_embeddings.preprocessing.max_size_mb": "IMAGE_MAX_SIZE_MB",
    "image_embeddings.circuit_breaker.failure_threshold": "CLIP_CIRCUIT_BREAKER_THRESHOLD",
    "image_embeddings.circuit_breaker.recovery_timeout": "CLIP_CIRCUIT_BREAKER_TIMEOUT",
    # Processing
    "processing.workers": "RAY_NUM_WORKERS",
    "processing.batch_size": "RAY_S3_BATCH_SIZE",
    "processing.image_batch_size": "RAY_IMAGE_BATCH_SIZE",
    "processing.embed_batch_size": "RAY_EMBED_BATCH_MAX",
    "processing.upsert_batch_size": "RAY_BATCH_UPSERT_SIZE",
    "processing.checkpoint.enabled": "RAY_CHECKPOINT_ENABLED",
    "processing.checkpoint.directory": "RAY_CHECKPOINT_DIR",
    # Ray
    "ray.address": "RAY_ADDRESS",
    "ray.namespace": "RAY_NAMESPACE",
    "ray.cluster.num_workers": "RAY_NUM_WORKERS",
    "ray.cluster.worker_cpus": "RAY_WORKER_CPUS",
    "ray.cluster.worker_memory_bytes": "RAY_WORKER_MEMORY",
    "ray.cluster.head_cpus": "RAY_HEAD_CPUS",
    "ray.cluster.head_memory_bytes": "RAY_HEAD_MEMORY",
    "ray.tasks.num_cpus": "RAY_TASK_NUM_CPUS",
    "ray.tasks.max_retries": "RAY_TASK_MAX_RETRIES",
    "ray.tasks.pipeline_concurrency": "RAY_PIPELINE_CONCURRENCY",
    "ray.tasks.progress_log_interval": "RAY_PROGRESS_LOG_INTERVAL",
    "ray.rate_limits.embedding_max_concurrency": "RAY_EMBEDDING_MAX_CONCURRENCY",
    "ray.rate_limits.embedding_requests_per_second": "RAY_EMBEDDING_REQUESTS_PER_SECOND",
    "ray.timeouts.embedding_seconds": "RAY_EMBEDDING_TIMEOUT",
    "ray.timeouts.upsert_seconds": "RAY_UPSERT_TIMEOUT",
    "ray.timeouts.wait_seconds": "RAY_WAIT_TIMEOUT",
    "ray.retry.max_attempts": "RAY_RETRY_MAX_ATTEMPTS",
    "ray.retry.min_wait_seconds": "RAY_RETRY_MIN_WAIT",
    "ray.retry.max_wait_seconds": "RAY_RETRY_MAX_WAIT",
    "ray.circuit_breaker.failure_threshold": "RAY_CIRCUIT_BREAKER_THRESHOLD",
    "ray.circuit_breaker.recovery_timeout": "RAY_CIRCUIT_BREAKER_TIMEOUT",
    # Databricks
    "databricks.host": "DATABRICKS_HOST",
    "databricks.job_id": "DATABRICKS_JOB_ID",
    "databricks.task_type": "DATABRICKS_TASK_TYPE",
    "databricks.workspace_path": "DATABRICKS_WORKSPACE_PATH",
    # Semantic Cache
    "semantic_cache.enabled": "SEMANTIC_CACHE_ENABLED",
    "semantic_cache.similarity_threshold": "SEMANTIC_CACHE_SIMILARITY_THRESHOLD",
    "semantic_cache.max_entries": "SEMANTIC_CACHE_MAX_ENTRIES",
    "semantic_cache.invalidation_interval": "SEMANTIC_CACHE_INVALIDATION_INTERVAL",
    "semantic_cache.timeout": "SEMANTIC_CACHE_TIMEOUT",
    # Environment
    "environment.mode": "PIPELINE_ENV",
}


# =============================================================================
# Apply YAML Defaults
# =============================================================================


def _get_nested(config: dict, dotted_path: str) -> Any:
    """Extract a value from a nested dict using a dotted path.

    Args:
        config: Nested dictionary.
        dotted_path: Dot-separated key path (e.g. "storage.bucket").

    Returns:
        The value at the path, or None if any key is missing.
    """
    current = config
    for key in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _coerce_to_str(value: Any) -> str | None:
    """Convert a YAML value to a string suitable for os.environ.

    Args:
        value: Value from YAML config.

    Returns:
        String representation, or None if the value should be skipped.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def apply_yaml_defaults(config: dict) -> None:
    """Apply YAML config values as env var defaults.

    For each mapping in YAML_TO_ENV_MAP, extracts the value from the merged
    config and sets the corresponding env var only if it is not already set.

    Args:
        config: Merged YAML configuration dictionary.
    """
    for yaml_path, env_key in YAML_TO_ENV_MAP.items():
        value = _get_nested(config, yaml_path)
        str_value = _coerce_to_str(value)
        if str_value is None:
            continue
        # Empty strings from YAML should not override env var defaults
        if str_value == "":
            continue
        if env_key not in os.environ:
            os.environ[env_key] = str_value


# =============================================================================
# Public Entry Point
# =============================================================================


def initialize_config(config_dir: Path | None = None) -> None:
    """Load YAML configs and apply as env var defaults.

    Safe to call multiple times; only runs once per process.

    Args:
        config_dir: Optional path to the configs directory.
    """
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    env = os.environ.get("ENV") or os.environ.get("PIPELINE_ENV")
    config = load_yaml_config(env=env, config_dir=config_dir)
    apply_yaml_defaults(config)

    _initialized = True
    logger.info(
        "YAML config initialized",
        extra={"env": env or "(base only)", "config_dir": str(config_dir or _DEFAULT_CONFIG_DIR)},
    )
