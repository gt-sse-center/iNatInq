"""Configuration management for the dev pipeline orchestrator.

This module provides configuration system for the pipeline service using
Pydantic Settings. All settings are loaded from environment variables
with sensible dev defaults.

## Configuration Sources

Configuration is read from environment variables at process startup. The
`get_settings()` function uses `@lru_cache` to ensure settings are
loaded once per process (containers have static env vars, so this is
safe and efficient).

## Environment Variables

The following environment variables are supported (all optional with
defaults):

**Ollama (Embeddings Service)**
- `OLLAMA_BASE_URL`: Base URL for Ollama API
  (default: `http://ollama.ml-system:11434`)
- `OLLAMA_MODEL`: Default embedding model name
  (default: `nomic-embed-text`)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: `60`)
- `OLLAMA_CIRCUIT_BREAKER_THRESHOLD`: Failures before circuit opens
  (default: `5`)
- `OLLAMA_CIRCUIT_BREAKER_TIMEOUT`: Circuit recovery timeout in seconds
  (default: `30`)
- `OLLAMA_BATCH_TIMEOUT_MULTIPLIER`: Multiplier for batch timeout
  (default: `1.0`)
- `OLLAMA_MAX_BATCH_SIZE`: Maximum texts per batch request (default: `12`)

**Qdrant (Vector Database)**
- `QDRANT_URL`: Qdrant service URL. Auto-detected based on environment:
  - In-cluster: `http://qdrant.{namespace}:6333` (default)
  - Local: `http://localhost:6333` (default)
- `QDRANT_COLLECTION`: Default collection name for storing vectors
  (default: `documents`)
- `QDRANT_API_KEY`: Optional API key for Qdrant Cloud authentication
- `QDRANT_TIMEOUT`: Request timeout in seconds (default: `300`)
- `QDRANT_CIRCUIT_BREAKER_THRESHOLD`: Failures before circuit opens
  (default: `3`)
- `QDRANT_CIRCUIT_BREAKER_TIMEOUT`: Circuit recovery timeout in seconds
  (default: `60`)
- `QDRANT_PREFER_GRPC`: Whether to prefer gRPC over HTTP
  (default: `false`)
- `QDRANT_DISABLE_INDEXING_DURING_INGEST`: When true, the ingestion driver
  disables Qdrant indexing for the target collection before processing and
  re-enables it after all batches complete (default: `false`)

**S3/MinIO (Object Storage)**
- `S3_ENDPOINT`: S3-compatible service endpoint. Auto-detected based
  on environment:
  - In-cluster: `http://minio.{namespace}:9000` (default)
  - Local: `http://localhost:9000` (default)
- `S3_ACCESS_KEY_ID`: S3 access key (default: `minioadmin`)
- `S3_SECRET_ACCESS_KEY`: S3 secret key (default: `minioadmin`)
- `S3_BUCKET`: Default bucket name for pipeline data
  (default: `pipeline`)
- `S3_REGION`: AWS region name (default: `us-east-1`)
- `S3_USE_SSL`: Whether to use SSL/TLS (default: `false`)
- `S3_PATH_STYLE`: Whether to use path-style addressing
  (default: `true`)
- `S3_TIMEOUT`: Request timeout in seconds (default: `30`)
- `S3_MAX_RETRIES`: Maximum retry attempts (default: `3`)
- `S3_RETRY_MIN_WAIT`: Minimum retry backoff in seconds (default: `1.0`)
- `S3_RETRY_MAX_WAIT`: Maximum retry backoff in seconds (default: `10.0`)
- `S3_CIRCUIT_BREAKER_THRESHOLD`: Failures before circuit opens
  (default: `5`)
- `S3_CIRCUIT_BREAKER_TIMEOUT`: Circuit recovery timeout in seconds
  (default: `120`)

**Environment Detection**
- `PIPELINE_ENV`: Explicit environment override (`cluster` or `local`).
  If not set, automatically detected via Kubernetes service account
  token or `KUBERNETES_SERVICE_HOST`.

**Spark/Kubernetes**
- `SPARK_MASTER_URL`: Spark master URL
  (default: `spark://spark-master:7077` or `local[*]` for local mode)
- `SPARK_EXECUTOR_MEMORY`: Memory per executor (default: `2g`)
- `SPARK_EXECUTOR_CORES`: Cores per executor (default: `2`)
- `SPARK_DRIVER_MEMORY`: Driver memory (default: `1g`)
- `SPARK_DEFAULT_PARALLELISM`: Default RDD partitions (default: `200`)
- `SPARK_SHUFFLE_PARTITIONS`: Shuffle partitions (default: `200`)
- `SPARK_NETWORK_TIMEOUT`: Network timeout (default: `600s`)
- `SPARK_HEARTBEAT_INTERVAL`: Executor heartbeat interval
  (default: `60s`)
- `SPARK_CHECKPOINT_DIR`: Checkpoint directory
  (default: `/tmp/spark-checkpoints` or S3 URI)
- `SPARK_CHECKPOINT_ENABLED`: Enable checkpointing (default: `true`)
- `SPARK_CHECKPOINT_SAVE_INTERVAL`: Save checkpoint every N items
  (default: `10`)
- `SPARK_CHECKPOINT_SAVE_INTERVAL_SECONDS`: Save checkpoint every N
  seconds (default: `30.0`)
- `SPARK_PARTITION_TARGET_SIZE`: Target keys per partition
  (default: `100`)
- `SPARK_MAX_PARTITIONS`: Maximum partitions (default: `200`)
- `SPARK_BATCH_UPSERT_SIZE`: Qdrant batch upsert size (default: `200`)
- `SPARK_EMBED_BATCH_SIZE`: Embedding batch size (default: `8`)
- `SPARK_MAX_CONCURRENT_PER_PARTITION`: Max concurrent ops per
  partition (default: `20`)
- `SPARK_MAX_CONCURRENT_BATCH_UPSERTS`: Max concurrent batch upsert
  operations per partition (default: `5`)
- `SPARK_RETRY_MAX_ATTEMPTS`: Max retry attempts (default: `3`)
- `SPARK_RETRY_WAIT_MIN`: Min retry wait seconds (default: `2.0`)
- `SPARK_RETRY_WAIT_MAX`: Max retry wait seconds (default: `10.0`)
- `SPARK_RETRY_MULTIPLIER`: Retry exponential backoff multiplier
  (default: `1.0`)

**Ray Job Configuration**
- `RAY_ADDRESS`: Ray cluster address (auto-detected in K8s if
  `K8S_NAMESPACE` is set)
- `RAY_NUM_WORKERS`: Number of Ray worker processes (default: `0`)
- `RAY_WORKER_CPUS`: CPUs per worker (default: `1.0`)
- `RAY_WORKER_MEMORY`: Memory per worker in bytes
  (default: `500000000` = 500MB)
- `RAY_HEAD_CPUS`: CPUs for head node (default: `1.0`)
- `RAY_HEAD_MEMORY`: Memory for head node in bytes
  (default: `200000000` = 200MB)
- `RAY_NAMESPACE`: Ray namespace for job isolation
  (default: `ml-pipeline`)
- `RAY_OLLAMA_MAX_CONCURRENCY`: Maximum concurrent Ollama requests per
  worker (default: `10`)
- `RAY_OLLAMA_RPS`: Rate limit for Ollama requests per second
  (default: `5`)
- `RAY_EMBED_BATCH_MIN`: Minimum batch size for embeddings
  (default: `1`)
- `RAY_EMBED_BATCH_MAX`: Maximum batch size for embeddings
  (default: `8`)
- `RAY_BATCH_UPSERT_SIZE`: Batch size for vector DB upserts
  (default: `200`)
- `RAY_S3_BATCH_SIZE`: Number of S3 keys per Ray task (default: `50`)
- `RAY_CHECKPOINT_DIR`: Checkpoint directory
  (default: `/tmp/ray-checkpoints` or S3 URI)
- `RAY_CHECKPOINT_ENABLED`: Enable checkpointing (default: `true`)
- `RAY_TASK_NUM_CPUS`: CPUs requested per Ray task (default: `1`)
- `RAY_TASK_MAX_RETRIES`: Max retries for failed tasks (default: `3`)
- `RAY_PIPELINE_CONCURRENCY`: Max concurrent async ops in task
  (default: `10`)
- `RAY_WAIT_TIMEOUT`: ray.wait() timeout in seconds (default: `1.0`)
- `RAY_WAIT_BATCH_SIZE`: Results per ray.wait() call (default: `10`)
- `RAY_PROGRESS_LOG_INTERVAL`: Log progress every N keys
  (default: `1000`)
- `S3_PREFIX`: S3 key prefix for pipeline jobs (default: `""`)
- `IMAGE_MAX_ITEMS`: Optional cap on images to process
  (default: no limit)
- `IMAGE_PAGE_SIZE`: Keys per S3 API page (default: `1000`)
- `RAY_CIRCUIT_BREAKER_THRESHOLD`: Failures to open breaker
  (default: `5`)
- `RAY_CIRCUIT_BREAKER_TIMEOUT`: Recovery timeout in seconds
  (default: `30`)
- `RAY_EMBEDDING_TIMEOUT`: Embedding request timeout (default: `120`)
- `RAY_UPSERT_TIMEOUT`: Vector DB upsert timeout (default: `60`)
- `RAY_RETRY_MAX_ATTEMPTS`: Max retry attempts (default: `3`)
- `RAY_RETRY_MIN_WAIT`: Min retry wait in seconds (default: `1.0`)
- `RAY_RETRY_MAX_WAIT`: Max retry wait in seconds (default: `10.0`)

**Databricks Job Configuration**
- `DATABRICKS_HOST`: Databricks workspace host (e.g., `https://dbc.cloud`)
- `DATABRICKS_TOKEN`: Databricks access token
- `DATABRICKS_JOB_ID`: Databricks job ID (integer)
- `DATABRICKS_INAT_JOB_ID`: Optional dedicated Databricks iNaturalist image job ID
- `DATABRICKS_TASK_TYPE`: Task parameter style (`python` only, default: `python`)
- `DATABRICKS_WORKSPACE_PATH`: Optional workspace path (if used)

**Embedding Provider Configuration**
- `EMBEDDING_PROVIDER`: Provider type - `ollama`, `openai`,
  `huggingface`, or `sagemaker` (default: `ollama`)
- `EMBEDDING_VECTOR_SIZE`: Expected vector dimension (optional,
  auto-detected if not set)
- `OLLAMA_BASE_URL`: Ollama service URL (default: auto-detected based
  on environment)
- `OLLAMA_MODEL`: Ollama model name (default: `nomic-embed-text`)
- `OPENAI_API_KEY`: OpenAI API key (required if
  `EMBEDDING_PROVIDER=openai`)
- `OPENAI_MODEL`: OpenAI model name
  (default: `text-embedding-ada-002`)
- `HUGGINGFACE_MODEL`: HuggingFace model name (required if
  `EMBEDDING_PROVIDER=huggingface`)
- `HUGGINGFACE_DEVICE`: Device for HuggingFace models - `cpu` or
  `cuda` (default: `cpu`)
- `SAGEMAKER_ENDPOINT`: SageMaker endpoint name (required if
  `EMBEDDING_PROVIDER=sagemaker`)
- `SAGEMAKER_REGION`: AWS region for SageMaker (default: `us-east-1`)

**Image Embedding Provider Configuration**
- `IMAGE_EMBEDDING_PROVIDER`: Provider type - `clip` or `llava`
  (default: `clip`)
- `CLIP_URL`: CLIP/Ollama service URL (default: auto-detected based on
  environment and backend)
- `CLIP_MODEL`: Model name for image embedding (default: `ViT-B/32`
  for clip backend, `llava` for ollama backend)
- `CLIP_BACKEND`: API backend type - `ollama`, `clip`, or `hosted_clip`
  (default: `ollama`)
- `CLIP_API_KEY`: Optional API key for hosted CLIP endpoints
- `CLIP_TIMEOUT`: Request timeout in seconds (default: `120`)
- `CLIP_CIRCUIT_BREAKER_THRESHOLD`: Failures before circuit opens
  (default: `5`)
- `CLIP_CIRCUIT_BREAKER_TIMEOUT`: Circuit recovery timeout in seconds
  (default: `30`)
- `CLIP_MAX_BATCH_SIZE`: Maximum images per batch API request
  (default: `8`)
- `CLIP_VECTOR_SIZE`: Override auto-detected vector dimension
  (optional)
- `IMAGE_BATCH_SIZE`: Images per processing batch in Ray/pipeline
  (default: `10`, smaller than text due to memory)
- `IMAGE_MAX_SIZE_MB`: Maximum allowed image size in megabytes
  (default: `10.0`)
- `IMAGE_TARGET_SIZE`: Target dimension for image resizing before
  embedding (default: `224`, standard CLIP input size)

**Vector Database Provider Configuration**
- `VECTOR_DB_PROVIDER`: Provider type - `qdrant` or `weaviate`
  (default: `qdrant`)
- `VECTOR_DB_COLLECTION`: Collection name (default: `documents`)
- `QDRANT_URL`: Qdrant service URL (backward compatible,
  auto-detected if not set)
- `WEAVIATE_URL`: Weaviate service URL (required if
  `VECTOR_DB_PROVIDER=weaviate`)
- `WEAVIATE_API_KEY`: Weaviate API key (optional, for authenticated
  instances)

**Kubernetes**
- `K8S_NAMESPACE`: Kubernetes namespace for ML components
  (default: `ml-system`)

## Usage

```python
from config import get_settings
from clients import create_s3_client, create_vector_db_client

# Get settings
settings = get_settings()

# Create clients using factory functions
embedding_client = create_embedding_client()
vector_db_client = create_vector_db_client()
s3_client = create_s3_client()

# Or use config directly
from config import EmbeddingConfig, VectorDBConfig, MinIOConfig
embedding_client = create_embedding_client(config=settings.embedding)
vector_db_client = create_vector_db_client(config=settings.vector_db)
s3_client = create_s3_client(config=settings.minio)
```

## Design Notes

This module uses Pydantic Settings for configuration management, providing:
- Type coercion and validation
- Environment variable parsing
- Nested configuration structures
- Multiple configuration sources (env vars)
"""

import os
import logging
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


def _parse_optional_positive_int(name: str) -> int | None:
    """Parse an optional positive integer from an environment variable.

    Returns None if the variable is unset, empty, not a valid integer,
    or not positive.

    Args:
        name: Environment variable name.

    Returns:
        Parsed positive integer, or None.
    """
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except ValueError:
        return None


def _is_in_cluster() -> bool:
    """Detect if running inside a Kubernetes cluster.

    Checks for:
    1. Kubernetes service account token (most reliable)
    2. KUBERNETES_SERVICE_HOST environment variable
    3. Explicit PIPELINE_ENV=cluster environment variable

    Returns:
        True if running in-cluster, False otherwise.
    """
    # Check for explicit override
    env_override = os.getenv("PIPELINE_ENV")
    if env_override:
        return env_override.lower() == "cluster"

    # Check for service account token (most reliable)
    if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
        return True

    # Check for Kubernetes service host
    return bool(os.getenv("KUBERNETES_SERVICE_HOST"))


def resolve_vector_db_provider(provider: str | None = None, default: str = "qdrant") -> str:
    """Normalize vector DB provider and apply a fallback default.

    Args:
        provider: Optional provider override. If None, reads VECTOR_DB_PROVIDER
            from the environment.
        default: Provider to use when no value is configured.

    Returns:
        Normalized provider string.
    """
    raw_value = provider if provider is not None else os.getenv("VECTOR_DB_PROVIDER")
    normalized = (raw_value or "").strip().lower()
    return normalized or default


def resolve_vector_db_targets(
    provider: str | None = None, *, logger: logging.Logger | None = None
) -> tuple[bool, bool]:
    """Resolve which vector DB targets should be enabled.

    Args:
        provider: Optional provider override. If None, reads
            ``VECTOR_DB_PROVIDER`` from the environment.
        logger: Optional logger used for invalid provider warnings.

    Returns:
        Tuple of ``(use_qdrant, use_weaviate)``.
    """
    normalized_provider = resolve_vector_db_provider(provider=provider, default="both")
    target_mapping: dict[str, tuple[bool, bool]] = {
        "both": (True, True),
        "qdrant": (True, False),
        "weaviate": (False, True),
    }
    targets = target_mapping.get(normalized_provider)
    if targets is not None:
        return targets

    if logger is not None:
        logger.warning(
            "Invalid VECTOR_DB_PROVIDER; defaulting to both",
            extra={"vector_db_provider": normalized_provider},
        )
    return target_mapping["both"]


class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider.

    This configuration class supports multiple embedding providers and can be
    extended to add new providers without breaking existing code.

    Attributes:
        provider_type: Type of embedding provider. Must be one of:
            "ollama", "openai", "huggingface", or "sagemaker".
        vector_size: Expected vector dimension. If None, will be
            auto-detected from the first embedding or provider default.
        ollama_url: Ollama service URL. Required if
            provider_type="ollama". Auto-detected based on environment if
            not set.
        ollama_model: Ollama model name. Required if
            provider_type="ollama". Default: "nomic-embed-text".
        ollama_timeout: Ollama request timeout in seconds. Default: 60.
        ollama_circuit_breaker_threshold: Failures before circuit opens.
            Default: 5.
        ollama_circuit_breaker_timeout: Circuit recovery timeout in seconds.
            Default: 30.
        ollama_batch_timeout_multiplier: Multiplier for batch timeout.
            Default: 1.0.
        ollama_max_batch_size: Maximum texts per batch request. Default: 12.
        openai_api_key: OpenAI API key. Required if
            provider_type="openai".
        openai_model: OpenAI model name. Required if
            provider_type="openai". Default: "text-embedding-ada-002".
        huggingface_model: HuggingFace model name. Required if
            provider_type="huggingface".
        huggingface_device: Device for HuggingFace models. Must be
            "cpu" or "cuda". Default: "cpu".
        sagemaker_endpoint: SageMaker endpoint name. Required if
            provider_type="sagemaker".
        sagemaker_region: AWS region for SageMaker endpoint.
            Default: "us-east-1".
    """

    provider_type: Literal["ollama", "openai", "huggingface", "sagemaker"]
    vector_size: int | None = None

    # Ollama settings
    ollama_url: str | None = None
    ollama_model: str | None = None
    ollama_timeout: int = 60
    ollama_circuit_breaker_threshold: int = 5
    ollama_circuit_breaker_timeout: int = 30
    ollama_batch_timeout_multiplier: float = 1.0
    ollama_max_batch_size: int = 12

    # OpenAI settings
    openai_api_key: str | None = None
    openai_model: str | None = None

    # HuggingFace settings
    huggingface_model: str | None = None
    huggingface_device: str | None = None

    # SageMaker settings
    sagemaker_endpoint: str | None = None
    sagemaker_region: str | None = None

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> "EmbeddingConfig":
        """Create EmbeddingConfig from environment variables.

        Supports:
        - EMBEDDING_PROVIDER: Provider type (ollama, openai, etc.)
        - EMBEDDING_VECTOR_SIZE: Expected vector dimension (optional)
        - OLLAMA_BASE_URL, OLLAMA_MODEL: Ollama config (backward compatible)
        - OPENAI_API_KEY, OPENAI_MODEL: OpenAI config
        - HUGGINGFACE_MODEL: HuggingFace model name
        - SAGEMAKER_ENDPOINT, SAGEMAKER_REGION: SageMaker config

        Args:
            namespace: Kubernetes namespace for service discovery.

        Returns:
            Configured EmbeddingConfig instance.
        """
        # Determine provider type
        provider_type = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

        # Validate provider type
        valid_providers = ("ollama", "openai", "huggingface", "sagemaker")
        if provider_type not in valid_providers:
            msg = f"Invalid EMBEDDING_PROVIDER: {provider_type}. Must be one of: {valid_providers}"
            raise ValueError(msg)

        in_cluster = _is_in_cluster()

        # Build config based on provider type
        # Parse vector size once (can be None)
        vector_size_str = os.getenv("EMBEDDING_VECTOR_SIZE")
        vector_size: int | None = None
        if vector_size_str:
            vector_size = int(vector_size_str)

        if provider_type == "ollama":
            # Default URL based on environment
            default_url = f"http://ollama.{namespace}:11434" if in_cluster else "http://localhost:11434"
            ollama_url_val = os.getenv("OLLAMA_BASE_URL") or default_url
            ollama_model_val = os.getenv("OLLAMA_MODEL") or "nomic-embed-text"
            return cls(
                provider_type="ollama",
                vector_size=vector_size,
                ollama_url=ollama_url_val,
                ollama_model=ollama_model_val,
                ollama_timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
                ollama_circuit_breaker_threshold=int(os.getenv("OLLAMA_CIRCUIT_BREAKER_THRESHOLD", "5")),
                ollama_circuit_breaker_timeout=int(os.getenv("OLLAMA_CIRCUIT_BREAKER_TIMEOUT", "30")),
                ollama_batch_timeout_multiplier=float(os.getenv("OLLAMA_BATCH_TIMEOUT_MULTIPLIER", "1.0")),
                ollama_max_batch_size=int(os.getenv("OLLAMA_MAX_BATCH_SIZE", "12")),
            )

        if provider_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            openai_model_val = os.getenv("OPENAI_MODEL") or "text-embedding-ada-002"
            return cls(
                provider_type="openai",
                vector_size=vector_size,
                openai_api_key=api_key,
                openai_model=openai_model_val,
            )

        if provider_type == "huggingface":
            model = os.getenv("HUGGINGFACE_MODEL")
            if not model:
                raise ValueError("HUGGINGFACE_MODEL is required for HuggingFace provider")
            huggingface_device_val = os.getenv("HUGGINGFACE_DEVICE") or "cpu"
            return cls(
                provider_type="huggingface",
                vector_size=vector_size,
                huggingface_model=model,
                huggingface_device=huggingface_device_val,
            )

        if provider_type == "sagemaker":
            endpoint = os.getenv("SAGEMAKER_ENDPOINT")
            if not endpoint:
                raise ValueError("SAGEMAKER_ENDPOINT is required for SageMaker provider")
            sagemaker_region_val = os.getenv("SAGEMAKER_REGION") or "us-east-1"
            return cls(
                provider_type="sagemaker",
                vector_size=vector_size,
                sagemaker_endpoint=endpoint,
                sagemaker_region=sagemaker_region_val,
            )

        # This should be unreachable due to validation above, but needed
        # for type checking
        msg = f"Unsupported provider type: {provider_type}"
        raise ValueError(msg)


class ImageEmbeddingConfig(BaseModel):
    """Configuration for image embedding provider (CLIP, LLaVA, etc.).

    This configuration class supports multi-modal embedding providers that can
    generate embeddings from image data.

    Attributes:
        provider_type: Type of image embedding provider. Supports "clip" or "llava".
            Default: "clip".
        clip_url: CLIP/Ollama service URL. Required if provider_type="clip".
            Auto-detected based on environment if not set.
        clip_model: Model name for image embedding (e.g., "llava", "ViT-B/32").
            Default: "llava".
        clip_backend: API backend type. One of "ollama", "clip", or "hosted_clip".
            Default: "ollama".
            - "ollama": Uses Ollama's /api/embeddings endpoint with LLaVA
            - "clip": Uses ai4all/clip's /embed/image endpoint
            - "hosted_clip": Uses hosted CLIP endpoints (Azure ML-style /score)
        clip_timeout: Request timeout in seconds. Default: 120 (higher for images).
        clip_circuit_breaker_threshold: Failures before circuit opens. Default: 5.
        clip_circuit_breaker_timeout: Circuit recovery timeout in seconds. Default: 30.
        clip_max_batch_size: Maximum images per batch API request. Default: 8.
        clip_vector_size: Override auto-detected vector size. Default: None.
        image_batch_size: Number of images per processing batch (Ray/pipeline).
            Smaller than text batches due to higher memory per image. Default: 10.
        image_max_size_mb: Maximum allowed image size in megabytes. Images larger
            than this will be rejected. Default: 10 MB.
        image_target_size: Target dimension for image resizing before embedding.
            Images are resized to (target_size, target_size) for CLIP models.
            Default: 224 (standard CLIP input size).
    """

    provider_type: Literal["clip", "llava"] = "clip"

    # CLIP settings
    clip_url: str | None = None
    clip_model: str | None = None
    clip_backend: Literal["ollama", "clip", "hosted_clip"] = "ollama"
    clip_api_key: str | None = None
    clip_timeout: int = 120
    clip_circuit_breaker_threshold: int = 5
    clip_circuit_breaker_timeout: int = 30
    clip_max_batch_size: int = 8
    clip_vector_size: int | None = None

    # Image processing settings
    image_batch_size: int = 10
    image_max_size_mb: float = 10.0
    image_target_size: int = 224

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> "ImageEmbeddingConfig":
        """Create ImageEmbeddingConfig from environment variables.

        Supports:
        - IMAGE_EMBEDDING_PROVIDER: Provider type ("clip" or "llava", default: "clip")
        - CLIP_URL or OLLAMA_BASE_URL: Service URL
        - CLIP_MODEL: Model name (default depends on backend)
        - CLIP_BACKEND: API backend type ("ollama", "clip", or "hosted_clip", default: "ollama")
        - CLIP_API_KEY: Optional API key for authenticated CLIP/Ollama endpoints
        - CLIP_TIMEOUT: Request timeout in seconds
        - CLIP_CIRCUIT_BREAKER_THRESHOLD: Failures before circuit opens
        - CLIP_CIRCUIT_BREAKER_TIMEOUT: Circuit recovery timeout
        - CLIP_MAX_BATCH_SIZE: Maximum images per batch API request
        - CLIP_VECTOR_SIZE: Override vector dimension
        - IMAGE_BATCH_SIZE: Images per processing batch (Ray/pipeline)
        - IMAGE_MAX_SIZE_MB: Maximum allowed image size in MB
        - IMAGE_TARGET_SIZE: Target dimension for image resizing

        Args:
            namespace: Kubernetes namespace for service discovery.

        Returns:
            Configured ImageEmbeddingConfig instance.
        """
        in_cluster = _is_in_cluster()

        # Get provider type (clip or llava)
        provider_type = os.getenv("IMAGE_EMBEDDING_PROVIDER", "clip").lower()
        if provider_type not in ("clip", "llava"):
            provider_type = "clip"

        # Get backend type (ollama, clip, or hosted_clip)
        clip_backend = os.getenv("CLIP_BACKEND", "ollama").lower()
        if clip_backend not in ("ollama", "clip", "hosted_clip"):
            clip_backend = "ollama"

        # Resolve URL based on backend
        if clip_backend in ("clip", "hosted_clip"):
            default_url = f"http://clip.{namespace}:8000" if in_cluster else "http://localhost:8000"
            default_model = "ViT-B/32"
        else:
            default_url = f"http://ollama.{namespace}:11434" if in_cluster else "http://localhost:11434"
            default_model = "llava"

        clip_url = os.getenv("CLIP_URL") or os.getenv("OLLAMA_BASE_URL", default_url)
        clip_model = os.getenv("CLIP_MODEL", default_model)

        # Parse optional vector size
        vector_size_str = os.getenv("CLIP_VECTOR_SIZE")
        clip_vector_size: int | None = None
        if vector_size_str:
            clip_vector_size = int(vector_size_str)

        return cls(
            provider_type=provider_type,  # type: ignore[arg-type]
            clip_url=clip_url,
            clip_model=clip_model,
            clip_backend=clip_backend,  # type: ignore[arg-type]
            clip_api_key=os.getenv("CLIP_API_KEY"),
            clip_timeout=int(os.getenv("CLIP_TIMEOUT", "120")),
            clip_circuit_breaker_threshold=int(os.getenv("CLIP_CIRCUIT_BREAKER_THRESHOLD", "5")),
            clip_circuit_breaker_timeout=int(os.getenv("CLIP_CIRCUIT_BREAKER_TIMEOUT", "30")),
            clip_max_batch_size=int(os.getenv("CLIP_MAX_BATCH_SIZE", "8")),
            clip_vector_size=clip_vector_size,
            image_batch_size=int(os.getenv("IMAGE_BATCH_SIZE", "10")),
            image_max_size_mb=float(os.getenv("IMAGE_MAX_SIZE_MB", "10.0")),
            image_target_size=int(os.getenv("IMAGE_TARGET_SIZE", "224")),
        )


class MinIOConfig(BaseModel):
    """Configuration for MinIO/S3-compatible object storage.

    Attributes:
        endpoint_url: S3 service endpoint URL. Automatically resolved
            based on environment (in-cluster vs local).
        access_key_id: S3 access key for authentication.
            Default: "minioadmin".
        secret_access_key: S3 secret key for authentication.
            Default: "minioadmin".
        bucket: Default bucket name for operations. Default: "pipeline".
        region: AWS region name. Default: "us-east-1".
        use_ssl: Whether to use SSL/TLS. Default: False (MinIO
            typically uses HTTP).
        path_style: Whether to use path-style addressing. Default: True
            (required for MinIO compatibility).
        timeout: Request timeout in seconds. Default: 30.
        max_retries: Maximum retry attempts for transient errors. Default: 3.
        retry_min_wait: Minimum wait between retries in seconds. Default: 1.0.
        retry_max_wait: Maximum wait between retries in seconds. Default: 10.0.
        circuit_breaker_threshold: Failures before circuit breaker opens.
            Default: 5.
        circuit_breaker_timeout: Seconds before circuit breaker recovery.
            Default: 120.
    """

    # Connection settings
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    region: str = "us-east-1"
    use_ssl: bool = False
    path_style: bool = True

    # Timeout and retry settings
    timeout: int = 30
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 120

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> "MinIOConfig":
        """Create MinIOConfig from environment variables.

        Automatically resolves endpoint URL:
        - In-cluster: Uses Kubernetes service discovery
        - Local: Uses localhost with port-forward assumption

        Args:
            namespace: Kubernetes namespace (default: ml-system).

        Returns:
            Configured MinIOConfig instance.
        """
        # Detect environment
        in_cluster = _is_in_cluster()

        # Resolve endpoint
        default_endpoint = f"http://minio.{namespace}:9000" if in_cluster else "http://localhost:9000"
        endpoint = os.getenv("S3_ENDPOINT", default_endpoint)

        return cls(
            # Connection settings
            endpoint_url=endpoint,
            access_key_id=os.getenv("S3_ACCESS_KEY_ID", "minioadmin"),
            secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin"),
            bucket=os.getenv("S3_BUCKET", "pipeline"),
            region=os.getenv("S3_REGION", "us-east-1"),
            use_ssl=os.getenv("S3_USE_SSL", "false").lower() == "true",
            path_style=os.getenv("S3_PATH_STYLE", "true").lower() == "true",
            # Timeout and retry settings
            timeout=int(os.getenv("S3_TIMEOUT", "30")),
            max_retries=int(os.getenv("S3_MAX_RETRIES", "3")),
            retry_min_wait=float(os.getenv("S3_RETRY_MIN_WAIT", "1.0")),
            retry_max_wait=float(os.getenv("S3_RETRY_MAX_WAIT", "10.0")),
            # Circuit breaker settings
            circuit_breaker_threshold=int(os.getenv("S3_CIRCUIT_BREAKER_THRESHOLD", "5")),
            circuit_breaker_timeout=int(os.getenv("S3_CIRCUIT_BREAKER_TIMEOUT", "120")),
        )


class VectorDBConfig(BaseModel):
    """Configuration for vector database provider.

    This configuration class supports Qdrant and Weaviate vector databases.

    Attributes:
        provider_type: Type of vector database provider. Must be one of:
            "qdrant" or "weaviate".
        collection: Default collection name to use for storing and
            querying vectors.
        ingestion_targets: Set of vector DBs to index during ingestion.
            Defaults to both ("qdrant", "weaviate"). At least one required.
        qdrant_url: Qdrant service URL. Required if
            provider_type="qdrant". Auto-detected based on environment
            if not set.
        qdrant_api_key: Qdrant API key. Optional, for authenticated instances.
        qdrant_timeout: Qdrant request timeout in seconds. Default: 300.
        qdrant_circuit_breaker_threshold: Failures before circuit opens.
            Default: 3.
        qdrant_circuit_breaker_timeout: Circuit recovery timeout in seconds.
            Default: 60.
        weaviate_url: Weaviate service URL. Required if
            provider_type="weaviate". Auto-detected based on environment
            if not set.
        weaviate_api_key: Weaviate API key. Optional, for authenticated
            instances.
        weaviate_timeout: Weaviate request timeout in seconds. Default: 300.
        weaviate_circuit_breaker_threshold: Failures before circuit opens.
            Default: 3.
        weaviate_circuit_breaker_timeout: Circuit recovery timeout in seconds.
            Default: 60.
        qdrant_disable_indexing_during_ingest: If True, disable Qdrant indexing
            for the target collection during ingestion and re-enable it after
            all batches have completed. Default: False.
    """

    provider_type: Literal["qdrant", "weaviate"]
    collection: str
    ingestion_targets: frozenset[str] = frozenset({"qdrant", "weaviate"})

    # Qdrant settings
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout: int = 300
    qdrant_circuit_breaker_threshold: int = 3
    qdrant_circuit_breaker_timeout: int = 60
    qdrant_disable_indexing_during_ingest: bool = False

    # Weaviate settings
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None
    weaviate_grpc_host: str | None = None
    weaviate_grpc_port: int | None = None  # Custom gRPC port for testcontainers
    weaviate_timeout: int = 300
    weaviate_circuit_breaker_threshold: int = 3
    weaviate_circuit_breaker_timeout: int = 60

    model_config = SettingsConfigDict(frozen=True)

    @staticmethod
    def parse_targets_from_env() -> frozenset[str]:
        """Parse VECTOR_DB_TARGETS from environment.

        Reads a comma-separated list of target database names. Each entry
        must be one of {"qdrant", "weaviate"}. At least one is required.

        Returns:
            Frozenset of validated target names.

        Raises:
            ValueError: If targets are empty or contain invalid names.
        """
        raw = os.getenv("VECTOR_DB_TARGETS")
        if not raw:
            return frozenset({"qdrant", "weaviate"})

        valid = {"qdrant", "weaviate"}
        targets = frozenset(t.strip().lower() for t in raw.split(",") if t.strip())

        if not targets:
            raise ValueError("VECTOR_DB_TARGETS must contain at least one target")

        invalid = targets - valid
        if invalid:
            msg = f"Invalid VECTOR_DB_TARGETS: {', '.join(sorted(invalid))}. Must be subset of: {valid}"
            raise ValueError(msg)

        return targets

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> "VectorDBConfig":
        """Create VectorDBConfig from environment variables.

        Supports:
        - VECTOR_DB_PROVIDER: Provider type (qdrant, weaviate, etc.)
        - VECTOR_DB_COLLECTION: Collection name (default: "documents")
        - VECTOR_DB_TARGETS: Comma-separated ingestion targets (default: "qdrant,weaviate")
        - QDRANT_URL: Qdrant service URL (backward compatible)
        - WEAVIATE_URL: Weaviate service URL
        - WEAVIATE_API_KEY: Weaviate API key (optional)

        Args:
            namespace: Kubernetes namespace for service discovery.

        Returns:
            Configured VectorDBConfig instance.
        """
        # Determine provider type
        provider_type = resolve_vector_db_provider()

        # Validate provider type
        valid_providers = ("qdrant", "weaviate")
        if provider_type not in valid_providers:
            msg = f"Invalid VECTOR_DB_PROVIDER: {provider_type}. Must be one of: {valid_providers}"
            raise ValueError(msg)

        in_cluster = _is_in_cluster()
        ingestion_targets = cls.parse_targets_from_env()

        # Build config based on provider type
        collection = os.getenv("VECTOR_DB_COLLECTION", "documents")

        if provider_type == "qdrant":
            # Default URL based on environment
            default_url = f"http://qdrant.{namespace}:6333" if in_cluster else "http://localhost:6333"
            return cls(
                provider_type="qdrant",
                collection=collection,
                ingestion_targets=ingestion_targets,
                qdrant_url=os.getenv("QDRANT_URL", default_url),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                qdrant_timeout=int(os.getenv("QDRANT_TIMEOUT", "300")),
                qdrant_circuit_breaker_threshold=int(os.getenv("QDRANT_CIRCUIT_BREAKER_THRESHOLD", "3")),
                qdrant_circuit_breaker_timeout=int(os.getenv("QDRANT_CIRCUIT_BREAKER_TIMEOUT", "60")),
                qdrant_disable_indexing_during_ingest=os.getenv(
                    "QDRANT_DISABLE_INDEXING_DURING_INGEST", "false"
                ).lower()
                == "true",
            )

        # Weaviate is the only other valid option
        default_url = f"http://weaviate.{namespace}:8080" if in_cluster else "http://localhost:8080"
        return cls(
            provider_type="weaviate",
            collection=collection,
            ingestion_targets=ingestion_targets,
            weaviate_url=os.getenv("WEAVIATE_URL", default_url),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            weaviate_grpc_host=os.getenv("WEAVIATE_GRPC_HOST"),
            weaviate_timeout=int(os.getenv("WEAVIATE_TIMEOUT", "300")),
            weaviate_circuit_breaker_threshold=int(os.getenv("WEAVIATE_CIRCUIT_BREAKER_THRESHOLD", "3")),
            weaviate_circuit_breaker_timeout=int(os.getenv("WEAVIATE_CIRCUIT_BREAKER_TIMEOUT", "60")),
        )

    @classmethod
    def from_env_for_provider(cls, provider_type: str, namespace: str = "ml-system") -> "VectorDBConfig":
        """Create VectorDBConfig for a specific provider type.

        Unlike `from_env()` which reads VECTOR_DB_PROVIDER from the environment,
        this method accepts the provider type as a parameter. Useful when routes
        need to target a specific provider regardless of the default configuration.

        Args:
            provider_type: Provider type ("qdrant" or "weaviate").
            namespace: Kubernetes namespace for service discovery.

        Returns:
            Configured VectorDBConfig for the specified provider.

        Raises:
            ValueError: If provider type is invalid or required config is missing.
        """
        valid_providers = ("qdrant", "weaviate")
        if provider_type not in valid_providers:
            msg = f"Invalid provider type: {provider_type}. Must be one of: {valid_providers}"
            raise ValueError(msg)

        in_cluster = _is_in_cluster()
        collection = os.getenv("VECTOR_DB_COLLECTION", "documents")
        ingestion_targets = cls.parse_targets_from_env()

        if provider_type == "qdrant":
            default_url = f"http://qdrant.{namespace}:6333" if in_cluster else "http://localhost:6333"
            return cls(
                provider_type="qdrant",
                collection=collection,
                ingestion_targets=ingestion_targets,
                qdrant_url=os.getenv("QDRANT_URL", default_url),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            )

        # Weaviate is the only other valid option
        default_url = f"http://weaviate.{namespace}:8080" if in_cluster else "http://localhost:8080"
        return cls(
            provider_type="weaviate",
            collection=collection,
            ingestion_targets=ingestion_targets,
            weaviate_url=os.getenv("WEAVIATE_URL", default_url),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            weaviate_grpc_host=os.getenv("WEAVIATE_GRPC_HOST"),
            weaviate_timeout=int(os.getenv("WEAVIATE_TIMEOUT", "300")),
            weaviate_circuit_breaker_threshold=int(os.getenv("WEAVIATE_CIRCUIT_BREAKER_THRESHOLD", "3")),
            weaviate_circuit_breaker_timeout=int(os.getenv("WEAVIATE_CIRCUIT_BREAKER_TIMEOUT", "60")),
        )


class RayJobConfig(BaseModel):
    """Configuration for Ray job execution.

    All settings are loaded from environment variables with sensible defaults.
    This allows fine-tuning Ray job performance without code changes.

    Throughput Tuning Notes:
        - RAY_TASK_NUM_CPUS=0.5: Embedding tasks are I/O-bound (waiting on
          Ollama). Requesting 0.5 CPU allows Ray to schedule 2x more tasks
          per worker node.
        - RAY_WAIT_BATCH_SIZE=50: For large jobs (1000+ keys), reduces
          driver-side scheduling overhead.
        - RAY_OLLAMA_REQUESTS_PER_SECOND: With batch embedding API, each
          request embeds embed_batch_max texts. Consider increasing from 5
          to match desired throughput (5 RPS * 8 texts/request = 40
          embeddings/sec).
        - RAY_S3_BATCH_SIZE: Can increase from 50 to 100+ now that S3 fetch
          is parallel. More work per task = less Ray scheduling overhead.

    Attributes:
        num_workers: Number of Ray worker processes. Default: 0
            (auto-scale based on cluster resources when using external
            Ray cluster).
        worker_cpus: Number of CPU cores per worker. Default: 1.0.
        worker_memory: Memory per worker in bytes.
            Default: 500000000 (500MB).
        head_cpus: Number of CPU cores for head node. Default: 1.0.
        head_memory: Memory for head node in bytes.
            Default: 200000000 (200MB).
        ray_namespace: Ray namespace for job isolation.
            Default: "ml-pipeline".
        ray_address: Ray cluster address. If set, connects to external
            cluster. Auto-detected in K8s if K8S_NAMESPACE is set.
            Default: None.
        dashboard_address: Ray dashboard HTTP address for job submission.
            Used by JobSubmissionClient. Auto-detected based on environment
            if not explicitly set.
        runtime_env: Runtime environment configuration (packages, env
            vars). Default: empty dict.
        ollama_max_concurrency: Maximum concurrent Ollama requests per
            worker. Default: 10.
        ollama_requests_per_second: Rate limit for Ollama requests per
            second. Default: 5.
        embed_batch_min: Minimum batch size for embedding generation.
            Default: 1.
        embed_batch_max: Maximum batch size for embedding generation.
            Default: 8.
        batch_upsert_size: Batch size for vector database upserts.
            Default: 200.
        checkpoint_dir: Checkpoint directory path (local path or S3
            URI). Default: "/tmp/ray-checkpoints".
        checkpoint_enabled: Whether to enable checkpointing for job
            recovery. Default: True.
        s3_batch_size: Number of S3 keys per Ray task. Larger batches reduce
            overhead but increase memory per task. Default: 50.
        image_batch_size: Number of image S3 keys per Ray task (image pipeline).
            Smaller than text batches due to higher memory per image. Default: 20.
        image_embed_batch_size: Batch size for CLIP image embedding per API call.
            Smaller than text embed batches. Default: 4.
        s3_prefix: S3 key prefix for pipeline jobs. Default: "".
        image_max_items: Optional cap on images to process. None means no
            limit. Default: None.
        image_page_size: Keys per S3 API page for image listing.
            Default: 1000.
        task_num_cpus: CPUs requested per Ray task. Affects scheduling.
            Default: 1.
        task_max_retries: Maximum retries for failed Ray tasks.
            Default: 3.
        pipeline_concurrency: Max concurrent async operations (embed/upsert)
            within a task. Default: 10.
        wait_timeout: Timeout in seconds for ray.wait() progress checks.
            Default: 1.0.
        wait_batch_size: Number of results to fetch per ray.wait() call.
            Default: 10.
        progress_log_interval: Log progress every N completed keys.
            Default: 1000.
        circuit_breaker_threshold: Failures before circuit breaker opens.
            Default: 5.
        circuit_breaker_timeout: Seconds before circuit breaker recovery.
            Default: 30.
        embedding_timeout: Timeout in seconds for embedding requests.
            Default: 120.
        upsert_timeout: Timeout in seconds for vector DB upserts.
            Default: 60.
        retry_max_attempts: Max retry attempts for transient failures.
            Default: 3.
        retry_min_wait: Minimum wait between retries in seconds.
            Default: 1.0.
        retry_max_wait: Maximum wait between retries in seconds.
            Default: 10.0.
    """

    # Cluster configuration
    num_workers: int = 4
    worker_cpus: float = 1.0
    worker_memory: int = 2_000_000_000  # 2GB
    head_cpus: float = 1.0
    head_memory: int = 1_000_000_000  # 1GB
    ray_namespace: str = "ml-pipeline"
    ray_address: str | None = None
    dashboard_address: str | None = None
    runtime_env: dict[str, Any] = Field(default_factory=dict)

    # Rate limiting and concurrency
    ollama_max_concurrency: int = 10
    ollama_requests_per_second: int = 5

    # Batch sizes
    embed_batch_min: int = 1
    embed_batch_max: int = 8
    batch_upsert_size: int = 200
    s3_batch_size: int = 50
    image_batch_size: int = 20
    image_embed_batch_size: int = 4

    # S3 image pipeline settings
    s3_prefix: str = ""
    image_max_items: int | None = None
    image_page_size: int = 1000

    # Checkpointing
    checkpoint_dir: str = "/tmp/ray-checkpoints"
    checkpoint_enabled: bool = True

    # Task configuration
    task_num_cpus: int = 1
    task_max_retries: int = 3
    pipeline_concurrency: int = 10

    # Progress monitoring
    wait_timeout: float = 1.0
    wait_batch_size: int = 10
    progress_log_interval: int = 1000

    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30

    # Timeouts
    embedding_timeout: int = 120
    upsert_timeout: int = 60

    # Retry configuration
    retry_max_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # In-flight batch limit (None = auto-calculate from cluster capacity)
    max_inflight_batches: int | None = None

    model_config = SettingsConfigDict(frozen=True)

    def effective_max_inflight_batches(self) -> int:
        """Compute effective max in-flight batches for streaming submission.

        If RAY_MAX_INFLIGHT_BATCHES is explicitly set, use that value.
        Otherwise, derive from cluster capacity:
        ``num_workers * (worker_cpus / task_num_cpus)``.

        Returns:
            Positive integer for max concurrent batch futures.
        """
        if self.max_inflight_batches is not None:
            return self.max_inflight_batches
        if self.num_workers <= 0:
            return 1
        tasks_per_worker = max(1, int(self.worker_cpus / self.task_num_cpus))
        return self.num_workers * tasks_per_worker

    @classmethod
    def from_env(cls, namespace: str | None = None) -> "RayJobConfig":
        """Create RayJobConfig from environment variables.

        Args:
            namespace: Kubernetes namespace for constructing default addresses.

        Returns:
            Configured RayJobConfig instance.

        Environment Variables:
            RAY_ADDRESS: Ray client address (e.g., ray://ray-head:20001).
            RAY_DASHBOARD_ADDRESS: Ray dashboard HTTP address for job submission
                (e.g., http://ray-head:8265). If not set, auto-detected from
                K8S_NAMESPACE or defaults to http://localhost:8265.
            K8S_NAMESPACE: Used for auto-detecting addresses when in Kubernetes.
        """
        # Get Ray address from env, default to external cluster in K8s
        ray_address = os.environ.get("RAY_ADDRESS")
        k8s_namespace = os.environ.get("K8S_NAMESPACE")

        if ray_address is None and k8s_namespace:
            # Auto-detect: if in K8s and RAY_ADDRESS not set, use external cluster
            ray_address = f"ray://ray-head.{k8s_namespace}.svc.cluster.local:10001"

        # Get dashboard address - this is what JobSubmissionClient needs
        dashboard_address = os.environ.get("RAY_DASHBOARD_ADDRESS")
        if dashboard_address is None:
            if k8s_namespace:
                # In Kubernetes, use service DNS
                dashboard_address = f"http://ray-head.{k8s_namespace}:8265"
            elif _is_in_cluster():
                # Fallback for in-cluster without K8S_NAMESPACE
                dashboard_address = "http://ray-head:8265"
            # Local development (Docker Compose uses service name)
            # Check if PIPELINE_ENV suggests we're in Docker Compose
            elif os.environ.get("PIPELINE_ENV") == "local":
                dashboard_address = "http://ray-head:8265"
            else:
                dashboard_address = "http://localhost:8265"

        return cls(
            # Cluster configuration
            num_workers=int(os.getenv("RAY_NUM_WORKERS", "0")),
            worker_cpus=float(os.getenv("RAY_WORKER_CPUS", "1.0")),
            worker_memory=int(os.getenv("RAY_WORKER_MEMORY", "500000000")),
            head_cpus=float(os.getenv("RAY_HEAD_CPUS", "1.0")),
            head_memory=int(os.getenv("RAY_HEAD_MEMORY", "200000000")),
            ray_namespace=os.getenv("RAY_NAMESPACE", "ml-pipeline"),
            ray_address=ray_address,
            dashboard_address=dashboard_address,
            runtime_env={},  # Can be extended to load from env
            # Rate limiting and concurrency
            ollama_max_concurrency=int(os.getenv("RAY_OLLAMA_MAX_CONCURRENCY", "10")),
            ollama_requests_per_second=int(os.getenv("RAY_OLLAMA_RPS", "5")),
            # Batch sizes
            embed_batch_min=int(os.getenv("RAY_EMBED_BATCH_MIN", "1")),
            embed_batch_max=int(os.getenv("RAY_EMBED_BATCH_MAX", "8")),
            batch_upsert_size=int(os.getenv("RAY_BATCH_UPSERT_SIZE", "200")),
            s3_batch_size=int(os.getenv("RAY_S3_BATCH_SIZE", "50")),
            image_batch_size=int(os.getenv("RAY_IMAGE_BATCH_SIZE", "20")),
            image_embed_batch_size=int(os.getenv("RAY_IMAGE_EMBED_BATCH_SIZE", "4")),
            # S3 image pipeline settings
            s3_prefix=os.getenv("S3_PREFIX", ""),
            image_max_items=_parse_optional_positive_int("IMAGE_MAX_ITEMS"),
            image_page_size=int(os.getenv("IMAGE_PAGE_SIZE", "1000")),
            # Checkpointing
            checkpoint_dir=os.getenv("RAY_CHECKPOINT_DIR", "/tmp/ray-checkpoints"),
            checkpoint_enabled=os.getenv("RAY_CHECKPOINT_ENABLED", "true").lower() == "true",
            # Task configuration
            task_num_cpus=int(os.getenv("RAY_TASK_NUM_CPUS", "1")),
            task_max_retries=int(os.getenv("RAY_TASK_MAX_RETRIES", "3")),
            pipeline_concurrency=int(os.getenv("RAY_PIPELINE_CONCURRENCY", "10")),
            # Progress monitoring
            wait_timeout=float(os.getenv("RAY_WAIT_TIMEOUT", "1.0")),
            wait_batch_size=int(os.getenv("RAY_WAIT_BATCH_SIZE", "10")),
            progress_log_interval=int(os.getenv("RAY_PROGRESS_LOG_INTERVAL", "1000")),
            # Circuit breaker
            circuit_breaker_threshold=int(os.getenv("RAY_CIRCUIT_BREAKER_THRESHOLD", "5")),
            circuit_breaker_timeout=int(os.getenv("RAY_CIRCUIT_BREAKER_TIMEOUT", "30")),
            # Timeouts
            embedding_timeout=int(os.getenv("RAY_EMBEDDING_TIMEOUT", "120")),
            upsert_timeout=int(os.getenv("RAY_UPSERT_TIMEOUT", "60")),
            # Retry configuration
            retry_max_attempts=int(os.getenv("RAY_RETRY_MAX_ATTEMPTS", "3")),
            retry_min_wait=float(os.getenv("RAY_RETRY_MIN_WAIT", "1.0")),
            retry_max_wait=float(os.getenv("RAY_RETRY_MAX_WAIT", "10.0")),
            # In-flight batch limit
            max_inflight_batches=_parse_optional_positive_int("RAY_MAX_INFLIGHT_BATCHES"),
        )


class INatConfig(BaseModel):
    """Configuration for iNaturalist image pipeline.

    Attributes:
        image_size: Image size variant for URL construction (e.g., "medium",
            "small", "original"). Default: "medium".
        max_rows: Maximum number of metadata rows to read. Required — no default.
        metadata_url: iNaturalist metadata URL. Empty string means auto-detect
            from client. Default: "".
        photo_base_url: Base URL for iNaturalist photo downloads.
            Default: "https://inaturalist-open-data.s3.amazonaws.com/photos".
        timeout_s: HTTP request timeout in seconds for image downloads.
            Default: 120.
        cb_failure_threshold: Circuit breaker failure threshold. Default: 5.
        cb_recovery_timeout_s: Circuit breaker recovery timeout in seconds.
            Default: 30.
        image_max_items: Optional cap on total images to process. None means
            no limit. Default: None.
    """

    image_size: str = "medium"
    max_rows: int
    metadata_url: str = ""
    photo_base_url: str = "https://inaturalist-open-data.s3.amazonaws.com/photos"
    timeout_s: int = 120
    cb_failure_threshold: int = 5
    cb_recovery_timeout_s: int = 30
    image_max_items: int | None = None

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls) -> "INatConfig":
        """Create INatConfig from environment variables.

        Environment Variables:
            INAT_IMAGE_SIZE: Image size variant (default: "medium").
            INAT_MAX_ROWS: Required positive integer — max metadata rows.
            INAT_METADATA_URL: Metadata URL (default: "" = auto-detect).
            INAT_PHOTO_BASE_URL: Photo base URL.
            INAT_TIMEOUT_S: Download timeout in seconds (default: 120).
            INAT_CB_FAILURE_THRESHOLD: Circuit breaker failures (default: 5).
            INAT_CB_RECOVERY_TIMEOUT_S: Circuit breaker recovery (default: 30).
            IMAGE_MAX_ITEMS: Optional cap on images to process.

        Returns:
            Configured INatConfig instance.

        Raises:
            RuntimeError: If INAT_MAX_ROWS is missing or not a positive integer.
        """
        raw_max_rows = os.getenv("INAT_MAX_ROWS", "").strip()
        if not raw_max_rows:
            raise RuntimeError("INAT_MAX_ROWS is required and must be a positive integer")
        max_rows = int(raw_max_rows)
        if max_rows <= 0:
            raise RuntimeError("INAT_MAX_ROWS must be a positive integer")
        return cls(
            image_size=(os.getenv("INAT_IMAGE_SIZE") or "medium").strip().lower(),
            max_rows=max_rows,
            metadata_url=(os.getenv("INAT_METADATA_URL") or "").strip(),
            photo_base_url=(
                os.getenv("INAT_PHOTO_BASE_URL") or "https://inaturalist-open-data.s3.amazonaws.com/photos"
            ).strip(),
            timeout_s=int((os.getenv("INAT_TIMEOUT_S") or "120").strip()),
            cb_failure_threshold=int((os.getenv("INAT_CB_FAILURE_THRESHOLD") or "5").strip()),
            cb_recovery_timeout_s=int((os.getenv("INAT_CB_RECOVERY_TIMEOUT_S") or "30").strip()),
            image_max_items=_parse_optional_positive_int("IMAGE_MAX_ITEMS"),
        )


class DatabricksRayJobConfig(BaseModel):
    """Configuration for Databricks job execution.

    Attributes:
        host: Databricks workspace host URL.
        token: Databricks access token.
        job_id: Databricks job ID.
        task_type: Parameter style for job tasks.
            Supported value: "python".
        workspace_path: Optional workspace path (used by notebook tasks).
    """

    host: str
    token: str
    job_id: int
    inat_job_id: int | None = None
    task_type: Literal["python"] = "python"
    workspace_path: str | None = None

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls) -> "DatabricksRayJobConfig":
        """Create DatabricksRayJobConfig from environment variables.

        Environment Variables:
            DATABRICKS_HOST: Databricks workspace host URL.
            DATABRICKS_TOKEN: Databricks access token.
            DATABRICKS_JOB_ID: Databricks job ID (integer).
            DATABRICKS_INAT_JOB_ID: Optional dedicated iNaturalist image job ID (integer).
            DATABRICKS_TASK_TYPE: Task parameter style ("python" only).
            DATABRICKS_WORKSPACE_PATH: Optional workspace path.
        """
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        job_id_raw = os.getenv("DATABRICKS_JOB_ID")
        inat_job_id_raw = os.getenv("DATABRICKS_INAT_JOB_ID")
        task_type = os.getenv("DATABRICKS_TASK_TYPE", "python").lower()
        workspace_path = os.getenv("DATABRICKS_WORKSPACE_PATH")

        missing = [
            name
            for name, value in (
                ("DATABRICKS_HOST", host),
                ("DATABRICKS_TOKEN", token),
                ("DATABRICKS_JOB_ID", job_id_raw),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required Databricks config: {', '.join(missing)}")

        try:
            job_id = int(job_id_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("DATABRICKS_JOB_ID must be an integer") from exc

        inat_job_id: int | None = None
        if inat_job_id_raw:
            try:
                inat_job_id = int(inat_job_id_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("DATABRICKS_INAT_JOB_ID must be an integer") from exc

        valid_task_types = ("python",)
        if task_type not in valid_task_types:
            msg = f"Invalid DATABRICKS_TASK_TYPE: {task_type}. Must be one of: {valid_task_types}"
            raise ValueError(msg)

        return cls(
            host=host,
            token=token,
            job_id=job_id,
            inat_job_id=inat_job_id,
            task_type=task_type,
            workspace_path=workspace_path,
        )


class Settings(BaseModel):
    """Immutable runtime configuration for the pipeline service.

    All fields are loaded from environment variables via `get_settings()`.
    The class is frozen to prevent accidental mutation after initialization.

    Attributes:
        embedding: Embedding provider configuration (provider-agnostic).
            Supports multiple providers: ollama, openai, huggingface,
            sagemaker.
        vector_db: Vector database provider configuration
            (provider-agnostic). Supports providers: qdrant, weaviate.
        minio: MinIO/S3 configuration for object storage. Contains
            endpoint URL, credentials, bucket name, and connection
            settings.
        k8s_namespace: Kubernetes namespace where ML components are
            deployed. Used for service discovery and resource naming.
    """

    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    minio: MinIOConfig
    k8s_namespace: str

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str | None = None) -> "Settings":
        """Create Settings from environment variables.

        Args:
            namespace: Kubernetes namespace. If None, uses K8S_NAMESPACE
                env var or defaults to "ml-system".

        Returns:
            Configured Settings instance.
        """
        ns = os.getenv("K8S_NAMESPACE", "ml-system") if namespace is None else namespace

        return cls(
            embedding=EmbeddingConfig.from_env(namespace=ns),
            vector_db=VectorDBConfig.from_env(namespace=ns),
            minio=MinIOConfig.from_env(namespace=ns),
            k8s_namespace=ns,
        )


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Load and return application settings (cached per process).

    This function reads all environment variables and constructs a
    `Settings` instance. The result is cached using `@lru_cache` to
    avoid re-reading env vars on every call.

    YAML configuration files are loaded first (if available) and applied
    as env var defaults before building Settings. Environment variables
    always take precedence over YAML values.

    Returns:
        A frozen `Settings` instance with all configuration values.

    Note:
        Settings are loaded once per process. In containerized
        environments, env vars are static for the lifetime of the
        container, so caching is safe and efficient. For local
        development with dynamic env changes, restart the service to
        pick up new values.
    """
    from config_loader import initialize_config

    initialize_config()
    return Settings.from_env()
