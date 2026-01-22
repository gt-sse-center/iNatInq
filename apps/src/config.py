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
- `RAY_CHECKPOINT_DIR`: Checkpoint directory
  (default: `/tmp/ray-checkpoints` or S3 URI)
- `RAY_CHECKPOINT_ENABLED`: Enable checkpointing (default: `true`)

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
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


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
    """

    provider_type: Literal["qdrant", "weaviate"]
    collection: str

    # Qdrant settings
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout: int = 300
    qdrant_circuit_breaker_threshold: int = 3
    qdrant_circuit_breaker_timeout: int = 60

    # Weaviate settings
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None
    weaviate_grpc_host: str | None = None

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str = "ml-system") -> "VectorDBConfig":
        """Create VectorDBConfig from environment variables.

        Supports:
        - VECTOR_DB_PROVIDER: Provider type (qdrant, weaviate, etc.)
        - VECTOR_DB_COLLECTION: Collection name (default: "documents")
        - QDRANT_URL: Qdrant service URL (backward compatible)
        - WEAVIATE_URL: Weaviate service URL
        - WEAVIATE_API_KEY: Weaviate API key (optional)

        Args:
            namespace: Kubernetes namespace for service discovery.

        Returns:
            Configured VectorDBConfig instance.
        """
        # Determine provider type
        provider_type = os.getenv("VECTOR_DB_PROVIDER", "qdrant").lower()

        # Validate provider type
        valid_providers = ("qdrant", "weaviate")
        if provider_type not in valid_providers:
            msg = f"Invalid VECTOR_DB_PROVIDER: {provider_type}. Must be one of: {valid_providers}"
            raise ValueError(msg)

        in_cluster = _is_in_cluster()

        # Build config based on provider type
        collection = os.getenv("VECTOR_DB_COLLECTION", "documents")

        if provider_type == "qdrant":
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

        # Weaviate is the only other valid option
        default_url = f"http://weaviate.{namespace}:8080" if in_cluster else "http://localhost:8080"
        return cls(
            provider_type="weaviate",
            collection=collection,
            weaviate_url=os.getenv("WEAVIATE_URL", default_url),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            weaviate_grpc_host=os.getenv("WEAVIATE_GRPC_HOST"),
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

        if provider_type == "qdrant":
            default_url = f"http://qdrant.{namespace}:6333" if in_cluster else "http://localhost:6333"
            return cls(
                provider_type="qdrant",
                collection=collection,
                qdrant_url=os.getenv("QDRANT_URL", default_url),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            )

        # Weaviate is the only other valid option
        default_url = f"http://weaviate.{namespace}:8080" if in_cluster else "http://localhost:8080"
        return cls(
            provider_type="weaviate",
            collection=collection,
            weaviate_url=os.getenv("WEAVIATE_URL", default_url),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            weaviate_grpc_host=os.getenv("WEAVIATE_GRPC_HOST"),
        )


class SparkJobConfig(BaseModel):
    """Configuration for Spark job execution.

    All settings are loaded from environment variables with sensible defaults.
    This allows fine-tuning Spark job performance without code changes.

    Attributes:
        master_url: Spark master URL (e.g.,
            `spark://spark-master:7077` or `local[*]`). Set to None when
            using Spark Operator (operator manages master URL).
        executor_memory: Memory per executor (e.g., `2g`, `4g`).
            Default: "2g".
        executor_cores: Number of CPU cores per executor. Default: 2.
        driver_memory: Driver memory (e.g., `1g`, `2g`). Default: "1g".
        default_parallelism: Default number of partitions for RDDs.
            Default: 200.
        shuffle_partitions: Number of partitions for shuffle operations.
            Default: 200.
        network_timeout: Network timeout for Spark operations.
            Default: "600s".
        heartbeat_interval: Executor heartbeat interval to Spark master.
            Default: "60s".
        checkpoint_dir: Checkpoint directory path (local path or S3 URI).
            Default: "/tmp/spark-checkpoints".
        checkpoint_enabled: Whether to enable checkpointing for job
            recovery. Default: True.
        checkpoint_save_interval: Save checkpoint after this many
            successful items. Default: 10.
        checkpoint_save_interval_seconds: Save checkpoint after this many
            seconds. Default: 30.0.
        partition_target_size: Target number of S3 keys per partition.
            Default: 100.
        max_partitions: Maximum number of partitions to create.
            Default: 200.
        batch_upsert_size: Batch size for vector database upserts.
            Default: 200.
        embed_batch_size: Batch size for embedding generation.
            Default: 8.
        max_concurrent_per_partition: Maximum concurrent operations per
            partition. Default: 20.
        max_concurrent_batch_upserts: Maximum concurrent batch upsert
            operations per partition. Default: 5.
        retry_max_attempts: Maximum retry attempts for failed operations.
            Default: 3.
        retry_wait_min: Minimum wait time between retries in seconds.
            Default: 2.0.
        retry_wait_max: Maximum wait time between retries in seconds.
            Default: 10.0.
        retry_multiplier: Exponential backoff multiplier for retry
            delays. Default: 1.0.
    """

    master_url: str | None
    executor_memory: str = "2g"
    executor_cores: int = 2
    driver_memory: str = "1g"
    default_parallelism: int = 200
    shuffle_partitions: int = 200
    network_timeout: str = "600s"
    heartbeat_interval: str = "60s"
    checkpoint_dir: str = "/tmp/spark-checkpoints"
    checkpoint_enabled: bool = True
    checkpoint_save_interval: int = 10
    checkpoint_save_interval_seconds: float = 30.0
    partition_target_size: int = 100
    max_partitions: int = 200
    batch_upsert_size: int = 200
    embed_batch_size: int = 8
    max_concurrent_per_partition: int = 20
    max_concurrent_batch_upserts: int = 5
    retry_max_attempts: int = 3
    retry_wait_min: float = 2.0
    retry_wait_max: float = 10.0
    retry_multiplier: float = 1.0

    model_config = SettingsConfigDict(frozen=True)

    @classmethod
    def from_env(cls, namespace: str | None = None) -> "SparkJobConfig":
        """Create SparkJobConfig from environment variables.

        Args:
            namespace: Kubernetes namespace for constructing default master
                URL.

        Returns:
            Configured SparkJobConfig instance.

        Raises:
            ValueError: If SPARK_MASTER_URL is not set and namespace is
                not provided.
        """
        # Get SPARK_MASTER_URL from environment
        # If not set and namespace provided, auto-construct for
        # standalone Spark cluster
        # If not set and running with Spark Operator, leave as None
        # (Spark Operator manages master)
        master_url = os.environ.get("SPARK_MASTER_URL")
        if master_url is not None:
            # Use provided master URL
            pass
        elif os.environ.get("SPARK_APPLICATION_ID"):
            # Running with Spark Operator - don't set master URL
            master_url = None
        elif namespace:
            # Auto-construct Spark master URL for standalone Spark cluster
            master_url = f"spark://spark-master.{namespace}:7077"
        else:
            raise ValueError(
                "SPARK_MASTER_URL is required when not using Spark "
                "Operator. Set SPARK_MASTER_URL environment variable or "
                "provide namespace to connect to external Spark cluster. "
                "When using Spark Operator, SPARK_MASTER_URL should not "
                "be set."
            )

        return cls(
            master_url=master_url,
            executor_memory=os.getenv("SPARK_EXECUTOR_MEMORY", "2g"),
            executor_cores=int(os.getenv("SPARK_EXECUTOR_CORES", "2")),
            driver_memory=os.getenv("SPARK_DRIVER_MEMORY", "1g"),
            default_parallelism=int(os.getenv("SPARK_DEFAULT_PARALLELISM", "200")),
            shuffle_partitions=int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "200")),
            network_timeout=os.getenv("SPARK_NETWORK_TIMEOUT", "600s"),
            heartbeat_interval=os.getenv("SPARK_HEARTBEAT_INTERVAL", "60s"),
            checkpoint_dir=os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints"),
            checkpoint_enabled=os.getenv("SPARK_CHECKPOINT_ENABLED", "true").lower() == "true",
            checkpoint_save_interval=int(os.getenv("SPARK_CHECKPOINT_SAVE_INTERVAL", "10")),
            checkpoint_save_interval_seconds=float(
                os.getenv("SPARK_CHECKPOINT_SAVE_INTERVAL_SECONDS", "30.0")
            ),
            partition_target_size=int(os.getenv("SPARK_PARTITION_TARGET_SIZE", "100")),
            max_partitions=int(os.getenv("SPARK_MAX_PARTITIONS", "200")),
            batch_upsert_size=int(os.getenv("SPARK_BATCH_UPSERT_SIZE", "200")),
            embed_batch_size=int(os.getenv("SPARK_EMBED_BATCH_SIZE", "8")),
            max_concurrent_per_partition=int(os.getenv("SPARK_MAX_CONCURRENT_PER_PARTITION", "20")),
            max_concurrent_batch_upserts=int(os.getenv("SPARK_MAX_CONCURRENT_BATCH_UPSERTS", "5")),
            retry_max_attempts=int(os.getenv("SPARK_RETRY_MAX_ATTEMPTS", "3")),
            retry_wait_min=float(os.getenv("SPARK_RETRY_WAIT_MIN", "2.0")),
            retry_wait_max=float(os.getenv("SPARK_RETRY_WAIT_MAX", "10.0")),
            retry_multiplier=float(os.getenv("SPARK_RETRY_MULTIPLIER", "1.0")),
        )


class RayJobConfig(BaseModel):
    """Configuration for Ray job execution.

    All settings are loaded from environment variables with sensible defaults.
    This allows fine-tuning Ray job performance without code changes.

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
    """

    num_workers: int = 4
    worker_cpus: float = 1.0
    worker_memory: int = 2_000_000_000  # 2GB
    head_cpus: float = 1.0
    head_memory: int = 1_000_000_000  # 1GB
    ray_namespace: str = "ml-pipeline"
    ray_address: str | None = None
    dashboard_address: str | None = None
    runtime_env: dict[str, Any] = Field(default_factory=dict)
    ollama_max_concurrency: int = 10
    ollama_requests_per_second: int = 5
    embed_batch_min: int = 1
    embed_batch_max: int = 8
    batch_upsert_size: int = 200
    checkpoint_dir: str = "/tmp/ray-checkpoints"
    checkpoint_enabled: bool = True

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
            num_workers=int(os.getenv("RAY_NUM_WORKERS", "0")),
            worker_cpus=float(os.getenv("RAY_WORKER_CPUS", "1.0")),
            worker_memory=int(os.getenv("RAY_WORKER_MEMORY", "500000000")),
            head_cpus=float(os.getenv("RAY_HEAD_CPUS", "1.0")),
            head_memory=int(os.getenv("RAY_HEAD_MEMORY", "200000000")),
            ray_namespace=os.getenv("RAY_NAMESPACE", "ml-pipeline"),
            ray_address=ray_address,
            dashboard_address=dashboard_address,
            runtime_env={},  # Can be extended to load from env
            ollama_max_concurrency=int(os.getenv("RAY_OLLAMA_MAX_CONCURRENCY", "10")),
            ollama_requests_per_second=int(os.getenv("RAY_OLLAMA_RPS", "5")),
            embed_batch_min=int(os.getenv("RAY_EMBED_BATCH_MIN", "1")),
            embed_batch_max=int(os.getenv("RAY_EMBED_BATCH_MAX", "8")),
            batch_upsert_size=int(os.getenv("RAY_BATCH_UPSERT_SIZE", "200")),
            checkpoint_dir=os.getenv("RAY_CHECKPOINT_DIR", "/tmp/ray-checkpoints"),
            checkpoint_enabled=os.getenv("RAY_CHECKPOINT_ENABLED", "true").lower() == "true",
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
        spark: Spark job configuration. Contains master URL,
            executor/driver resources, checkpoint settings, and
            performance tuning parameters.
        k8s_namespace: Kubernetes namespace where ML components are
            deployed. Used for service discovery and resource naming.
    """

    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    minio: MinIOConfig
    spark: SparkJobConfig
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
            spark=SparkJobConfig.from_env(namespace=ns),
            k8s_namespace=ns,
        )


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Load and return application settings (cached per process).

    This function reads all environment variables and constructs a
    `Settings` instance. The result is cached using `@lru_cache` to
    avoid re-reading env vars on every call.

    Returns:
        A frozen `Settings` instance with all configuration values.

    Note:
        Settings are loaded once per process. In containerized
        environments, env vars are static for the lifetime of the
        container, so caching is safe and efficient. For local
        development with dynamic env changes, restart the service to
        pick up new values.
    """
    return Settings.from_env()
