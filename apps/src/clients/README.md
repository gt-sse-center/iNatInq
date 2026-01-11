# Clients

External service clients for interacting with ML pipeline dependencies.

This directory contains object-oriented client wrapper classes around external
services used by the pipeline:

- **S3/MinIO**: Object storage for pipeline inputs/outputs
- **Vector Databases**: Qdrant, Weaviate (future), etc.
- **Embedding Providers**: Ollama, OpenAI (future), etc.
- **Kubernetes**: Spark job submission and management

## Design Principles

1. **Thin Wrappers**: Clients are minimal wrappers around third-party libraries,
   not business logic
2. **Object-Oriented**: Clients use class-based design with attrs for clean,
   type-safe interfaces
3. **Error Translation**: Clients translate external errors into `PipelineError`
   hierarchy
4. **No Business Logic**: Clients only handle connection, authentication, and
   basic operations
5. **Testability**: Clients can be easily mocked for testing
6. **Connection Pooling**: Clients support connection reuse for better
   performance
7. **Provider Abstraction**: Clients implement ABCs from `interfaces/` for
   provider-agnostic code

## Architecture

The clients package uses **Abstract Base Classes (ABCs)** to provide
provider-agnostic interfaces:

```
clients/
├── interfaces/          # Abstract Base Classes (ABCs)
│   ├── embedding.py     # EmbeddingProvider ABC
│   └── vector_db.py     # VectorDBProvider ABC
├── ollama.py            # OllamaClient (implements EmbeddingProvider)
├── qdrant.py            # QdrantClientWrapper (implements VectorDBProvider)
├── s3.py                # S3ClientWrapper
└── k8s.py               # KubernetesClient
```

This design allows:

- **Swapping providers** without changing service code
- **Testing** with mock providers
- **Extensibility** for new providers (just implement the ABC)

## Modules

### `interfaces/`

Abstract Base Classes (ABCs) that define provider interfaces.

**Modules:**

- **`embedding.py`**: `EmbeddingProvider` ABC and factory functions
- **`vector_db.py`**: `VectorDBProvider` ABC and factory functions

**Usage:**

```python
from clients.interfaces.embedding import (
    EmbeddingProvider,
    EmbeddingConfig,
    create_embedding_provider,
)

# Create provider from config (returns OllamaClient, OpenAIClient, etc.)
config = EmbeddingConfig.from_env()
provider = create_embedding_provider(config)

# Use provider (agnostic to implementation)
vector = provider.embed("hello world")
vector_size = provider.vector_size
```

See `interfaces/README.md` for detailed documentation.

### `s3.py`

S3-compatible storage client wrapper (MinIO in dev environments).

**Class:** `S3ClientWrapper`

**Methods:**

- `ensure_bucket(bucket: str)`: Creates a bucket if it doesn't exist (dev
  convenience)
- `put_object(bucket: str, key: str, body: bytes)`: Uploads an object to S3
- `get_object(bucket: str, key: str) -> bytes`: Downloads an object from S3
- `list_objects(bucket: str, prefix: str = "") -> list[str]`: Lists all object
  keys with the given prefix
- `client`: Property that returns the underlying boto3 S3 client

**Usage:**

```python
from clients.s3 import S3ClientWrapper

s3_client = S3ClientWrapper(
    endpoint_url="http://minio.ml-system:9000",
    access_key_id="minioadmin",
    secret_access_key="minioadmin"
)
s3_client.ensure_bucket("pipeline")
s3_client.put_object(bucket="pipeline", key="data.txt", body=b"hello")
content = s3_client.get_object(bucket="pipeline", key="data.txt")
keys = s3_client.list_objects(bucket="pipeline", prefix="inputs/")
```

### `qdrant.py`

Qdrant vector database client wrapper.

**Class:** `QdrantClientWrapper`

**Implements:** `VectorDBProvider` ABC

**Methods:**

- `ensure_collection(collection: str, vector_size: int)`: Creates a collection
  if it doesn't exist (dev convenience)
- `search(collection: str, query_vector: List[float], limit: int = 10) -> SearchResults`:
  Searches for similar vectors in a collection
- `batch_upsert(collection: str, points: list[PointStruct], vector_size: int)`:
  Batch upserts points into a collection (for high-throughput scenarios)
- `from_config(config: VectorDBConfig) -> VectorDBProvider`: Class method to
  create instance from config
- `client`: Property that returns the underlying QdrantClient

**Usage:**

```python
from clients.qdrant import QdrantClientWrapper
from qdrant_client.models import PointStruct

qdrant_client = QdrantClientWrapper(url="http://qdrant.ml-system:6333")
qdrant_client.ensure_collection(collection="documents", vector_size=768)

# Search for similar vectors
results = qdrant_client.search(
    collection="documents",
    query_vector=[0.1, 0.2, ...],  # 768-dimensional vector
    limit=10
)
# Returns: SearchResults with items (point_id, score, payload)

# Batch upsert for Spark jobs
points = [
    PointStruct(id="1", vector=[0.1, 0.2, ...], payload={"text": "hello"}),
    PointStruct(id="2", vector=[0.3, 0.4, ...], payload={"text": "world"}),
]
qdrant_client.batch_upsert(
    collection="documents",
    points=points,
    vector_size=768
)
```

**Or via factory:**

```python
from clients.interfaces.vector_db import create_vector_db_provider, VectorDBConfig

config = VectorDBConfig.from_env()
provider = create_vector_db_provider(config)  # Returns QdrantClientWrapper
results = provider.search(collection="documents", query_vector=[...], limit=10)
```

### `ollama.py`

Ollama embeddings API client.

**Class:** `OllamaClient`

**Implements:** `EmbeddingProvider` ABC

**Methods:**

- `embed(text: str) -> List[float]`: Generates embeddings for a single text via
  Ollama's API
- `embed_async(text: str) -> List[float]`: Async version of `embed()`
- `vector_size`: Property that returns the embedding dimension
- `from_config(config: EmbeddingConfig, session: requests.Session | None = None) -> EmbeddingProvider`:
  Class method to create instance from config
- `set_session(session: requests.Session)`: Sets a custom requests session for
  connection pooling
- `session`: Property that returns the requests session (creates one if needed)

**Usage:**

```python
from clients.ollama import OllamaClient
from foundation.http import create_retry_session

# Basic usage
client = OllamaClient(
    base_url="http://ollama.ml-system:11434",
    model="nomic-embed-text",
    timeout_s=60
)
embedding = client.embed("hello world")
# Returns: List[float] (768-dimensional vector)

# With connection pooling (for Spark jobs)
session = create_retry_session()
client = OllamaClient(base_url="http://ollama.ml-system:11434", model="nomic-embed-text")
client.set_session(session)  # Reuse session across multiple calls
embedding = client.embed("hello world")
```

**Or via factory:**

```python
from clients.interfaces.embedding import create_embedding_provider, EmbeddingConfig

config = EmbeddingConfig.from_env()
provider = create_embedding_provider(config)  # Returns OllamaClient
vector = provider.embed("hello world")
```

### `k8s.py`

Kubernetes client for Spark job management.

**Class:** `KubernetesClient`

**Methods:**

- `batch`: Property that returns the Kubernetes Batch API client (lazy
  initialization)
- `submit_s3_to_qdrant_job(...) -> str`: Submits an S3-to-vector-database
  processing job to the cluster
- `wait_for_job(namespace: str, name: str, timeout_s: int = 180) -> None`: Waits
  for a job to complete (with timeout)

**Usage:**

```python
from clients.k8s import KubernetesClient

# Create client (auto-detects in-cluster or local config)
client = KubernetesClient()

# Submit job
job_name = client.submit_s3_to_qdrant_job(
    namespace="ml-system",
    spark_master_url="spark://spark-master:7077",
    s3_endpoint="http://minio.ml-system:9000",
    s3_access_key_id="minioadmin",
    s3_secret_access_key="minioadmin",
    s3_bucket="pipeline",
    s3_prefix="inputs/",
    embedding_provider_type="ollama",
    embedding_vector_size=768,
    ollama_base_url="http://ollama.ml-system:11434",
    ollama_model="nomic-embed-text",
    vector_db_provider_type="qdrant",
    qdrant_url="http://qdrant.ml-system:6333",
    collection="documents",
)

# Wait for completion
client.wait_for_job(namespace="ml-system", name=job_name, timeout_s=180)
```

## Provider Abstraction

The clients package uses **Abstract Base Classes (ABCs)** to provide
provider-agnostic interfaces:

### `EmbeddingProvider` ABC

Defines the interface for embedding generation:

- `embed(text: str) -> List[float]`: Generate embedding (sync)
- `embed_async(text: str) -> List[float]`: Generate embedding (async)
- `vector_size: int`: Property returning embedding dimension
- `from_config(config: EmbeddingConfig, session: requests.Session | None = None) -> EmbeddingProvider`:
  Factory method

**Implementations:**

- `OllamaClient`: Ollama API client
- `OpenAIClient`: OpenAI API client (future)
- `HuggingFaceClient`: HuggingFace models (future)

### `VectorDBProvider` ABC

Defines the interface for vector database operations:

- `ensure_collection(collection: str, vector_size: int) -> None`: Ensure
  collection exists
- `search(collection: str, query_vector: List[float], limit: int = 10) -> SearchResults`:
  Search for similar vectors
- `batch_upsert(collection: str, points: list, vector_size: int) -> None`: Batch
  upsert points
- `from_config(config: VectorDBConfig) -> VectorDBProvider`: Factory method

**Implementations:**

- `QdrantClientWrapper`: Qdrant vector database
- `WeaviateClient`: Weaviate vector database (future)

### Factory Functions

Factory functions in `interfaces/` create provider instances from configuration:

```python
from clients.interfaces.embedding import create_embedding_provider, EmbeddingConfig
from clients.interfaces.vector_db import create_vector_db_provider, VectorDBConfig

# Create embedding provider (returns OllamaClient, OpenAIClient, etc.)
embedding_config = EmbeddingConfig.from_env()
embedding_provider = create_embedding_provider(embedding_config)

# Create vector database provider (returns QdrantClientWrapper, etc.)
vector_db_config = VectorDBConfig.from_env()
vector_db_provider = create_vector_db_provider(vector_db_config)
```

This allows services to work with any provider implementation without code
changes.

## Error Handling

All clients raise `UpstreamError` when external services fail:

- Connection errors
- Authentication failures
- Service unavailable
- Invalid responses
- Collection not found (for vector database search)

This allows the service layer to handle errors consistently without knowing the
specific client implementation.

## Search Functionality

The `search()` method in vector database providers performs vector similarity
search:

- **Input**: Query embedding vector (must match collection dimension)
- **Output**: `SearchResults` with items (point_id, score, payload) ordered by
  similarity
- **Similarity Metric**: Cosine similarity (configured in collection)
- **Score Range**: 0.0 to 1.0 (higher = more similar)

The method automatically:

- Handles missing collections (raises `UpstreamError`)
- Includes payloads in results
- Limits results to specified count
- Orders results by similarity (highest first)

## Connection Pooling

Clients support connection pooling for better performance, especially in Spark
jobs:

- **OllamaClient**: Supports custom `requests.Session` via `set_session()`
  method
- **S3ClientWrapper**: Reuses boto3 client internally (connection pooling
  handled by boto3)
- **QdrantClientWrapper**: Reuses QdrantClient internally (connection pooling
  handled by qdrant-client)
- **KubernetesClient**: Lazy initialization of Batch API client

**Example with shared session:**

```python
from clients.ollama import OllamaClient
from foundation.http import create_retry_session

# Create shared session for connection pooling
session = create_retry_session()

# Reuse session across multiple clients
client1 = OllamaClient(base_url="http://ollama:11434", model="nomic-embed-text")
client1.set_session(session)

client2 = OllamaClient(base_url="http://ollama:11434", model="nomic-embed-text")
client2.set_session(session)  # Same session, better performance
```

## Testing

Clients can be easily mocked for testing:

```python
from unittest.mock import Mock
from clients.interfaces.embedding import EmbeddingProvider

# Mock the provider (using ABC interface)
mock_provider = Mock(spec=EmbeddingProvider)
mock_provider.embed.return_value = [0.1, 0.2, ...]  # Mock embedding
mock_provider.vector_size = 768
```

## Dependencies

- `boto3`: S3/MinIO client
- `qdrant-client`: Qdrant client
- `requests`: HTTP client for Ollama
- `kubernetes`: Kubernetes Python client
- `attrs`: Class definition and validation
