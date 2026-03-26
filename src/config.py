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
- `RAY_EMBEDDING_MAX_CONCURRENCY`: Maximum concurrent embedding requests per
  worker (default: `10`)
- `RAY_EMBEDDING_REQUESTS_PER_SECOND`: Rate limit for embedding requests per second
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
- `DATABRICKS_S3_AUTOLOADER_JOB_ID`: Optional dedicated Databricks S3 Auto Loader job ID
- `DATABRICKS_FROM_BRONZE_JOB_ID`: Optional dedicated Databricks Bronze CDC consumer job ID
- `DATABRICKS_TASK_TYPE`: Task parameter style (`python` only, default: `python`)
- `DATABRICKS_WORKSPACE_PATH`: Optional workspace path (if used)

**Embedding Provider Configuration**
- `EMBEDDING_PROVIDER`: Provider type — `clip`, `hosted_clip`, `infinity`,
  `huggingface`, or `sagemaker` (default: `clip`)
- `EMBEDDING_VECTOR_SIZE`: Expected vector dimension (optional,
  auto-detected if not set)
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
- `EMBEDDING_PROVIDER`: `clip` for local CLIP model, `hosted_clip` for hosted model
- `CLIP_URL`: CLIP service URL (default: auto-detected based on
  environment and backend)
- `CLIP_MODEL`: Model name for image embedding (default: `ViT-B/32`)
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
- `VECTOR_DB_PROVIDER`: Provider type - `qdrant`
  (default: `qdrant`)
- `VECTOR_DB_COLLECTION`: Collection name (default: `documents`)
- `QDRANT_URL`: Qdrant service URL (backward compatible,
  auto-detected if not set)

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
from enum import StrEnum
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


class ProviderType(StrEnum):
    """Specifies the underlying model provider."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SAGEMAKER = "sagemaker"
    LOCAL_CLIP = "clip"
    HOSTED_CLIP = "hosted_clip"
    INFINITY = "infinity"


class EmbeddingConfig(BaseModel):
    """Configuration for an embedding provider.

    This configuration class supports multiple embedding providers and can be
    extended to add new providers without breaking existing code.

    Attributes:
        provider_type: Type of embedding provider.
        vector_size: Expected vector dimension. If None, will be
            auto-detected from the first embedding or provider default.

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

        clip_url: CLIP service URL. Required if provider_type="clip".
            Auto-detected based on environment if not set.
        clip_model: Model name for image embedding (e.g., "ViT-B/32").
            Default: "ViT-B/32".
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

    provider_type: ProviderType
    vector_size: int | None = None

    # Image processing settings
    image_batch_size: int = 10
    image_max_size_mb: float = 10.0
    image_target_size: int = 224

    # OpenAI settings
    openai_api_key: str | None = None
    openai_model: str | None = None

    # HuggingFace settings
    huggingface_model: str | None = None
    huggingface_device: str | None = None

    # SageMaker settings
    sagemaker_endpoint: str | None = None
    sagemaker_region: str | None = None

    # CLIP settings
    clip_url: str | None = None
    clip_model: str | None = None
    clip_api_key: str | None = None
    clip_timeout: int = 120
    clip_circuit_breaker_threshold: int = 5
    clip_circuit_breaker_timeout: int = 30
    clip_max_batch_size: int = 8
    clip_vector_size: int | None = None

    # Infinity settings
    infinity_url: str | None = None
    infinity_model: str | None = None
    infinity_timeout: int = 120
    infinity_vector_size: int | None = None

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> "EmbeddingConfig":
        """Create EmbeddingConfig from environment variables.

        Supports:
        - EMBEDDING_PROVIDER: Provider type (clip, hosted_clip, openai, infinity, etc.)
        - EMBEDDING_VECTOR_SIZE: Expected vector dimension (optional)
        - OPENAI_API_KEY, OPENAI_MODEL: OpenAI config
        - HUGGINGFACE_MODEL: HuggingFace model name
        - SAGEMAKER_ENDPOINT, SAGEMAKER_REGION: SageMaker config
        - CLIP_URL: CLIP service URL
        - CLIP_MODEL: Model name (default depends on backend)
        - CLIP_API_KEY: Optional API key for authenticated CLIP endpoints
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
            Configured EmbeddingConfig instance.

        """

        # Ensure anyone using the old IMAGE_EMBEDDING_PROVIDER env variable is notified
        if os.getenv("IMAGE_EMBEDDING_PROVIDER") is not None:
            raise ValueError(
                "IMAGE_EMBEDDING_PROVIDER is no longer supported. Please use EMBEDDING_PROVIDER instead."
            )

        # Determine provider type
        provider_type_str = os.getenv("EMBEDDING_PROVIDER", "clip").lower()
        valid_providers = list(ProviderType._value2member_map_)
        if provider_type_str not in valid_providers:
            raise ValueError(
                f"Invalid EMBEDDING_PROVIDER: {provider_type_str}. Must be one of: {valid_providers}"
            )

        provider_type = ProviderType(provider_type_str)

        # Provider-agnostic settings
        vector_size_str = os.getenv("EMBEDDING_VECTOR_SIZE")
        vector_size: int | None = int(vector_size_str) if vector_size_str else None

        in_cluster = _is_in_cluster()

        image_batch_size = int(os.getenv("IMAGE_BATCH_SIZE", "10"))
        image_max_size_mb = float(os.getenv("IMAGE_MAX_SIZE_MB", "10.0"))
        image_target_size = int(os.getenv("IMAGE_TARGET_SIZE", "224"))

        # Initialize all provider-specific fields with sensible defaults/None
        # OpenAI
        openai_api_key: str | None = None
        openai_model: str | None = None

        # HuggingFace
        huggingface_model: str | None = None
        huggingface_device: str | None = None

        # SageMaker
        sagemaker_endpoint: str | None = None
        sagemaker_region: str | None = None

        # CLIP / image embeddings
        clip_url: str | None = None
        clip_model: str | None = None
        clip_api_key = os.getenv("CLIP_API_KEY")
        clip_timeout = int(os.getenv("CLIP_TIMEOUT", "120"))
        clip_circuit_breaker_threshold = int(os.getenv("CLIP_CIRCUIT_BREAKER_THRESHOLD", "5"))
        clip_circuit_breaker_timeout = int(os.getenv("CLIP_CIRCUIT_BREAKER_TIMEOUT", "30"))
        clip_max_batch_size = int(os.getenv("CLIP_MAX_BATCH_SIZE", "8"))
        clip_vector_size_str = os.getenv("CLIP_VECTOR_SIZE")
        clip_vector_size: int | None = int(clip_vector_size_str) if clip_vector_size_str else None

        # Infinity settings
        infinity_url: str | None = None
        infinity_model: str | None = None
        infinity_timeout = int(os.getenv("INFINITY_TIMEOUT", "120"))
        infinity_vector_size_str = os.getenv("INFINITY_VECTOR_SIZE")
        infinity_vector_size: int | None = int(infinity_vector_size_str) if infinity_vector_size_str else None

        # Provider-specific validation
        if provider_type is ProviderType.OPENAI:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            openai_model = os.getenv("OPENAI_MODEL") or "text-embedding-ada-002"

        elif provider_type is ProviderType.HUGGINGFACE:
            huggingface_model = os.getenv("HUGGINGFACE_MODEL")
            if not huggingface_model:
                raise ValueError("HUGGINGFACE_MODEL is required for HuggingFace provider")
            huggingface_device = os.getenv("HUGGINGFACE_DEVICE") or "cpu"

        elif provider_type is ProviderType.SAGEMAKER:
            sagemaker_endpoint = os.getenv("SAGEMAKER_ENDPOINT")
            if not sagemaker_endpoint:
                raise ValueError("SAGEMAKER_ENDPOINT is required for SageMaker provider")
            sagemaker_region = os.getenv("SAGEMAKER_REGION") or "us-east-1"

        elif provider_type in (ProviderType.LOCAL_CLIP, ProviderType.HOSTED_CLIP):
            default_url = f"http://clip.{namespace}:8000" if in_cluster else "http://localhost:8000"
            clip_url = os.getenv("CLIP_URL") or default_url
            default_model = "ViT-B/32"
            clip_model = os.getenv("CLIP_MODEL", default_model)

        elif provider_type is ProviderType.INFINITY:
            default_url = f"http://infinity.{namespace}:7997" if in_cluster else "http://localhost:7997"
            infinity_url = os.getenv("INFINITY_URL") or default_url
            infinity_model = os.getenv("INFINITY_MODEL") or "google/siglip-so400m-patch14-384"

        # Single construction with all resolved values
        return cls(
            provider_type=provider_type,
            vector_size=vector_size,
            # Image settings
            image_batch_size=image_batch_size,
            image_max_size_mb=image_max_size_mb,
            image_target_size=image_target_size,
            # OpenAI
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            # HuggingFace
            huggingface_model=huggingface_model,
            huggingface_device=huggingface_device,
            # SageMaker
            sagemaker_endpoint=sagemaker_endpoint,
            sagemaker_region=sagemaker_region,
            # CLIP
            clip_url=clip_url,
            clip_model=clip_model,
            clip_api_key=clip_api_key,
            clip_timeout=clip_timeout,
            clip_circuit_breaker_threshold=clip_circuit_breaker_threshold,
            clip_circuit_breaker_timeout=clip_circuit_breaker_timeout,
            clip_max_batch_size=clip_max_batch_size,
            clip_vector_size=clip_vector_size,
            # Infinity
            infinity_url=infinity_url,
            infinity_model=infinity_model,
            infinity_timeout=infinity_timeout,
            infinity_vector_size=infinity_vector_size,
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

    This configuration class supports Qdrant vector database.

    Attributes:
        provider_type: Type of vector database provider. Must be "qdrant".
        collection: Default collection name to use for storing and
            querying vectors.
        qdrant_url: Qdrant service URL. Required if
            provider_type="qdrant". Auto-detected based on environment
            if not set.
        qdrant_api_key: Qdrant API key. Optional, for authenticated instances.
        qdrant_timeout: Qdrant request timeout in seconds. Default: 300.
        qdrant_circuit_breaker_threshold: Failures before circuit opens.
            Default: 3.
        qdrant_circuit_breaker_timeout: Circuit recovery timeout in seconds.
            Default: 60.
    """

    provider_type: Literal["qdrant"]
    collection: str

    # Qdrant settings
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout: int = 300
    qdrant_circuit_breaker_threshold: int = 3
    qdrant_circuit_breaker_timeout: int = 60

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> "VectorDBConfig":
        """Create VectorDBConfig from environment variables.

        Supports:
        - VECTOR_DB_PROVIDER: Provider type (qdrant)
        - VECTOR_DB_COLLECTION: Collection name (default: "documents")
        - QDRANT_URL: Qdrant service URL (backward compatible)

        Args:
            namespace: Kubernetes namespace for service discovery.

        Returns:
            Configured VectorDBConfig instance.
        """
        # Determine provider type
        provider_type = resolve_vector_db_provider()

        # Validate provider type
        valid_providers = ("qdrant",)
        if provider_type not in valid_providers:
            msg = f"Invalid VECTOR_DB_PROVIDER: {provider_type}. Must be one of: {valid_providers}"
            raise ValueError(msg)

        in_cluster = _is_in_cluster()

        # Build config based on provider type
        collection = os.getenv("VECTOR_DB_COLLECTION", "documents")

        # Default URL based on environment
        default_url = f"http://qdrant.{namespace}:6333" if in_cluster else "http://localhost:6333"
        return cls(
            provider_type="qdrant",
            collection=collection,
            qdrant_url=os.getenv("QDRANT_URL", default_url),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_timeout=int(os.getenv("QDRANT_TIMEOUT", "300")),
            qdrant_circuit_breaker_threshold=int(os.getenv("QDRANT_CIRCUIT_BREAKER_THRESHOLD", "3")),
            qdrant_circuit_breaker_timeout=int(os.getenv("QDRANT_CIRCUIT_BREAKER_TIMEOUT", "60")),
        )

    @classmethod
    def from_env_for_provider(cls, provider_type: str, namespace: str = "ml-system") -> "VectorDBConfig":
        """Create VectorDBConfig for a specific provider type.

        Unlike `from_env()` which reads VECTOR_DB_PROVIDER from the environment,
        this method accepts the provider type as a parameter. Useful when routes
        need to target a specific provider regardless of the default configuration.

        Args:
            provider_type: Provider type ("qdrant").
            namespace: Kubernetes namespace for service discovery.

        Returns:
            Configured VectorDBConfig for the specified provider.

        Raises:
            ValueError: If provider type is invalid or required config is missing.
        """
        valid_providers = ("qdrant",)
        if provider_type not in valid_providers:
            msg = f"Invalid provider type: {provider_type}. Must be one of: {valid_providers}"
            raise ValueError(msg)

        in_cluster = _is_in_cluster()
        collection = os.getenv("VECTOR_DB_COLLECTION", "documents")

        default_url = f"http://qdrant.{namespace}:6333" if in_cluster else "http://localhost:6333"
        return cls(
            provider_type="qdrant",
            collection=collection,
            qdrant_url=os.getenv("QDRANT_URL", default_url),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        )


class RayJobConfig(BaseModel):
    """Configuration for Ray job execution.

    All settings are loaded from environment variables with sensible defaults.
    This allows fine-tuning Ray job performance without code changes.

    Throughput Tuning Notes:
        - RAY_TASK_NUM_CPUS=0.5: Embedding tasks are I/O-bound (waiting on
          embedding services). Requesting 0.5 CPU allows Ray to schedule 2x more tasks
          per worker node.
        - RAY_WAIT_BATCH_SIZE=50: For large jobs (1000+ keys), reduces
          driver-side scheduling overhead.
        - RAY_EMBEDDING_REQUESTS_PER_SECOND: With batch embedding API, each
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
        embedding_max_concurrency: Maximum concurrent embedding requests per
            worker. Default: 10.
        embedding_requests_per_second: Rate limit for embedding requests per
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
        disable_indexing_during_ingest: Whether to disable Qdrant HNSW
            indexing before bulk ingestion and re-enable it after all
            batches are uploaded. Improves throughput for large loads.
            Default: False.
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
    embedding_max_concurrency: int = 10
    embedding_requests_per_second: int = 5

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

    # Qdrant indexing
    disable_indexing_during_ingest: bool = False

    model_config = SettingsConfigDict(frozen=True)

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
            embedding_max_concurrency=int(os.getenv("RAY_EMBEDDING_MAX_CONCURRENCY", "10")),
            embedding_requests_per_second=int(os.getenv("RAY_EMBEDDING_REQUESTS_PER_SECOND", "5")),
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
            # Qdrant indexing
            disable_indexing_during_ingest=os.getenv("DISABLE_INDEXING_DURING_INGEST", "false").lower()
            == "true",
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
        job_id: Databricks job ID. Optional when `require_job_id=False`.
        inat_job_id: Optional dedicated Databricks iNaturalist image job ID.
        s3_autoloader_job_id: Optional dedicated Databricks S3 Auto Loader job ID.
        s3_bronze_job_id: Optional dedicated Databricks Bronze CDC consumer job ID.
        task_type: Parameter style for job tasks.
            Supported value: "python".
        workspace_path: Optional workspace path (used by notebook tasks).
    """

    host: str
    token: str
    job_id: int | None = None
    inat_job_id: int | None = None
    s3_autoloader_job_id: int | None = None
    s3_bronze_job_id: int | None = None
    task_type: Literal["python"] = "python"
    workspace_path: str | None = None

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, *, require_job_id: bool = True) -> "DatabricksRayJobConfig":
        """Create DatabricksRayJobConfig from environment variables.

        Environment Variables:
            DATABRICKS_HOST: Databricks workspace host URL.
            DATABRICKS_TOKEN: Databricks access token.
            DATABRICKS_JOB_ID: Databricks job ID (integer).
            DATABRICKS_INAT_JOB_ID: Optional dedicated iNaturalist image job ID (integer).
            DATABRICKS_S3_AUTOLOADER_JOB_ID: Optional dedicated S3 Auto Loader job ID (integer).
            DATABRICKS_FROM_BRONZE_JOB_ID: Optional dedicated Bronze CDC consumer job ID (integer).
            DATABRICKS_TASK_TYPE: Task parameter style ("python" only).
            DATABRICKS_WORKSPACE_PATH: Optional workspace path.

        Args:
            require_job_id: If True, DATABRICKS_JOB_ID must be set and valid.
                When False, DATABRICKS_JOB_ID is optional (used by CDC producer
                and consumer submissions, which use dedicated job IDs).
        """
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        job_id_raw = os.getenv("DATABRICKS_JOB_ID")
        inat_job_id_raw = os.getenv("DATABRICKS_INAT_JOB_ID")
        s3_autoloader_job_id_raw = os.getenv("DATABRICKS_S3_AUTOLOADER_JOB_ID")
        s3_bronze_job_id_raw = os.getenv("DATABRICKS_FROM_BRONZE_JOB_ID")
        task_type = os.getenv("DATABRICKS_TASK_TYPE", "python").lower()
        workspace_path = os.getenv("DATABRICKS_WORKSPACE_PATH")

        required_pairs: list[tuple[str, str | None]] = [
            ("DATABRICKS_HOST", host),
            ("DATABRICKS_TOKEN", token),
        ]
        if require_job_id:
            required_pairs.append(("DATABRICKS_JOB_ID", job_id_raw))
        missing = [name for name, value in required_pairs if not value]
        if missing:
            raise ValueError(f"Missing required Databricks config: {', '.join(missing)}")
        assert host is not None
        assert token is not None

        job_id: int | None = None
        if job_id_raw:
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

        s3_autoloader_job_id: int | None = None
        if s3_autoloader_job_id_raw:
            try:
                s3_autoloader_job_id = int(s3_autoloader_job_id_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("DATABRICKS_S3_AUTOLOADER_JOB_ID must be an integer") from exc

        s3_bronze_job_id: int | None = None
        if s3_bronze_job_id_raw:
            try:
                s3_bronze_job_id = int(s3_bronze_job_id_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("DATABRICKS_FROM_BRONZE_JOB_ID must be an integer") from exc

        valid_task_types = ("python",)
        if task_type not in valid_task_types:
            msg = f"Invalid DATABRICKS_TASK_TYPE: {task_type}. Must be one of: {valid_task_types}"
            raise ValueError(msg)

        return cls(
            host=host,
            token=token,
            job_id=job_id,
            inat_job_id=inat_job_id,
            s3_autoloader_job_id=s3_autoloader_job_id,
            s3_bronze_job_id=s3_bronze_job_id,
            task_type=task_type,
            workspace_path=workspace_path,
        )


class SemanticCacheConfig(BaseModel):
    """Semantic cache configuration.

    Controls the in-memory Qdrant-backed semantic cache for search queries.
    The cache is always present in ``Settings``; the ``enabled`` flag
    controls whether caching is active.

    Attributes:
        enabled: Whether the semantic cache is active.
        similarity_threshold: Minimum cosine similarity score
            (0.0 <= threshold <= 1.0) to treat a cache lookup as a hit.
        max_entries_per_collection: Maximum cached entries per collection.
            A random entry is evicted when this limit is reached.
        invalidation_interval_seconds: Seconds between automatic
            invalidation sweeps.
        timeout_s: Timeout in seconds for the in-memory Qdrant client.

    Environment Variables:
        SEMANTIC_CACHE_ENABLED: ``true`` or ``false``. Default: ``true``.
        SEMANTIC_CACHE_SIMILARITY_THRESHOLD: Float in [0.0, 1.0]. Default: ``0.95``.
        SEMANTIC_CACHE_MAX_ENTRIES: Positive integer. Default: ``1000``.
        SEMANTIC_CACHE_INVALIDATION_INTERVAL: Positive integer (seconds). Default: ``3600``.
        SEMANTIC_CACHE_TIMEOUT: Positive integer (seconds). Default: ``5``.
    """

    enabled: bool = True
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    max_entries_per_collection: int = Field(default=1000, gt=0)
    invalidation_interval_seconds: int = Field(default=3600, gt=0)
    timeout_s: int = Field(default=5, gt=0)

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls) -> "SemanticCacheConfig":
        """Create SemanticCacheConfig from environment variables.

        Returns:
            Configured SemanticCacheConfig instance.
        """
        return cls(
            enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true",
            similarity_threshold=float(os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.95")),
            max_entries_per_collection=int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "1000")),
            invalidation_interval_seconds=int(os.getenv("SEMANTIC_CACHE_INVALIDATION_INTERVAL", "3600")),
            timeout_s=int(os.getenv("SEMANTIC_CACHE_TIMEOUT", "5")),
        )


class Settings(BaseModel):
    """Immutable runtime configuration for the pipeline service.

    All fields are loaded from environment variables via `get_settings()`.
    The class is frozen to prevent accidental mutation after initialization.

    Attributes:
        embedding: Embedding provider configuration (provider-agnostic).
            Supports multiple providers: clip, infinity, openai, huggingface,
            sagemaker.
        vector_db: Vector database provider configuration
            (provider-agnostic). Supports providers: qdrant.
        minio: MinIO/S3 configuration for object storage. Contains
            endpoint URL, credentials, bucket name, and connection
            settings.
        k8s_namespace: Kubernetes namespace where ML components are
            deployed. Used for service discovery and resource naming.
        semantic_cache: Semantic cache configuration. The ``enabled``
            field controls whether caching is active.
    """

    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    minio: MinIOConfig
    k8s_namespace: str
    semantic_cache: SemanticCacheConfig

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
            semantic_cache=SemanticCacheConfig.from_env(),
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
