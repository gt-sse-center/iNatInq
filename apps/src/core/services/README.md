# Services

Business logic orchestration layer for the iNatInq platform.

This directory contains service classes that orchestrate business operations by coordinating multiple clients and implementing workflow logic. Services are framework-agnostic and can be used from APIs, CLIs, or tests.

## Design Principles

1. **Framework-Agnostic**: No dependencies on web frameworks (FastAPI, etc.)
2. **Error Translation**: Services raise `PipelineError` hierarchy, not HTTP exceptions
3. **Orchestration**: Services coordinate multiple clients to accomplish business goals
4. **Testability**: Services use dependency injection for easy mocking
5. **Immutability**: Services are frozen attrs classes for thread safety
6. **Single Responsibility**: Each service handles one business domain

## Architecture

Services follow a clean separation of concerns:

```
┌─────────────┐
│ API Routes  │ (HTTP layer - translates to/from HTTP)
└──────┬──────┘
       │
┌──────▼──────┐
│  Services   │ (Business logic - framework-agnostic)
└──────┬──────┘
       │
┌──────▼──────┐
│   Clients   │ (External service wrappers)
└──────┬──────┘
       │
┌──────▼──────┐
│  External   │ (Ollama, Qdrant, S3, K8s, etc.)
│  Services   │
└─────────────┘
```

**Benefits:**

- Services can be used from APIs, CLIs, or notebooks
- Business logic is testable without HTTP mocking
- Easy to add new interfaces (GraphQL, gRPC, etc.)

## Directory Structure

```
services/
├── __init__.py           # Package exports
├── README.md             # This file
├── search_service.py     # Semantic search orchestration
├── spark_service.py      # Spark job management
└── ray_service.py        # Ray job orchestration
```

## Services

### `SearchService`

Orchestrates semantic search operations by coordinating embedding generation and vector database queries.

**Responsibilities:**

- Validate search parameters (query, limit, collection)
- Generate embeddings for query text
- Search vector database for similar vectors
- Return formatted results

**Dependencies:**

- `EmbeddingProvider`: For generating query embeddings
- `VectorDBProvider`: For searching vector database

**Usage:**

```python
from core.services import SearchService
from clients.interfaces.embedding import create_embedding_provider
from clients.interfaces.vector_db import create_vector_db_provider
from config import EmbeddingConfig, VectorDBConfig

# Create service
embedding_provider = create_embedding_provider(EmbeddingConfig.from_env())
vector_db_provider = create_vector_db_provider(VectorDBConfig.from_env())

service = SearchService(
    embedding_provider=embedding_provider,
    vector_db_provider=vector_db_provider,
)

# Perform search (sync)
results = service.search_documents(
    collection="documents",
    query="machine learning pipeline",
    limit=10
)

# Perform search (async)
results = await service.search_documents_async(
    collection="documents",
    query="machine learning pipeline",
    limit=10
)
```

**API Integration:**

```python
from fastapi import APIRouter
from core.services import SearchService
from core.exceptions import BadRequestError, UpstreamError

router = APIRouter()

@router.post("/search")
async def search(query: str, limit: int = 10):
    try:
        results = await search_service.search_documents_async(
            collection="documents",
            query=query,
            limit=limit
        )
        return results
    except BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except UpstreamError as e:
        raise HTTPException(status_code=502, detail=str(e))
```

### `SparkService`

Manages Spark-based data processing jobs via the Kubernetes Spark Operator.

**Responsibilities:**

- Submit Spark jobs to Kubernetes
- Track job status and execution
- List and manage running jobs
- Clean up completed jobs

**Dependencies:**

- `SparkJobClient`: Kubernetes API client for Spark Operator

**Usage:**

```python
from core.services import SparkService

# Create service
service = SparkService(namespace="ml-system")

# Submit job
result = service.submit_processing_job(
    s3_prefix="inputs/",
    collection="documents",
    executor_instances=2,
    executor_memory="1g"
)
print(f"Job submitted: {result['job_name']}")

# Check status
status = service.get_job_status(result['job_name'])
print(f"Job state: {status['state']}")

# List all jobs
jobs = service.list_jobs()
for job in jobs:
    print(f"{job['job_name']}: {job['state']}")

# Cleanup
service.delete_job(result['job_name'])
```

**Job Lifecycle:**

1. **Submit**: Creates SparkApplication CRD in Kubernetes
2. **Running**: Spark Operator launches driver and executor pods
3. **Processing**: S3 → Embeddings → Vector DB pipeline
4. **Completion**: Job finishes, pods terminate
5. **Cleanup**: Delete SparkApplication resource

### `RayService`

Orchestrates Ray jobs via the Ray Jobs API for distributed data processing.

**Responsibilities:**

- Submit non-blocking jobs to Ray cluster
- Track job status and logs
- Stop running jobs
- Handle Ray-specific errors

**Dependencies:**

- `JobSubmissionClient`: Ray Jobs API client

**Usage:**

```python
from core.services import RayService
from config import EmbeddingConfig, VectorDBConfig

# Create service
service = RayService()

# Submit job
job_id = service.submit_s3_to_qdrant(
    namespace="ml-system",
    s3_endpoint="http://minio.ml-system:9000",
    s3_access_key_id="minioadmin",
    s3_secret_access_key="minioadmin",
    s3_bucket="pipeline",
    s3_prefix="inputs/",
    embedding_config=EmbeddingConfig.from_env(),
    vector_db_config=VectorDBConfig.from_env(),
    collection="documents"
)
print(f"Job submitted: {job_id}")

# Check status
status = service.get_job_status(job_id, namespace="ml-system")
print(f"Job status: {status['status']}")

# Get logs
logs = service.get_job_logs(job_id, namespace="ml-system")
print(logs)

# Stop job if needed
service.stop_job(job_id, namespace="ml-system")
```

**Ray vs Spark:**

- **Ray**: Better for dynamic task graphs, real-time processing
- **Spark**: Better for batch processing, large-scale data transformations
- **Use Ray when**: You need low latency, dynamic workloads, or Python-native distribution
- **Use Spark when**: You have massive datasets, need SQL support, or require mature ecosystem

## Error Handling

Services raise exceptions from the `core.exceptions` hierarchy:

- `BadRequestError`: Invalid client input (e.g., empty query, invalid limit)
- `UpstreamError`: External service failure (e.g., Qdrant unavailable, S3 error)
- `PipelineTimeoutError`: Operation timeout (e.g., long-running job)

**Example:**

```python
from core.services import SearchService
from core.exceptions import BadRequestError, UpstreamError

try:
    results = service.search_documents(
        collection="documents",
        query="",  # Invalid empty query
        limit=10
    )
except BadRequestError as e:
    # Client error - return HTTP 400
    print(f"Invalid request: {e}")
except UpstreamError as e:
    # Service error - return HTTP 502
    print(f"External service failed: {e}")
```

## Testing

Services are designed for easy testing with dependency injection:

```python
from unittest.mock import MagicMock, AsyncMock
from core.services import SearchService
from core.models import SearchResults, SearchResultItem

# Create mocks
mock_embedding = MagicMock()
mock_embedding.embed.return_value = [0.1, 0.2, 0.3]

mock_vector_db = MagicMock()
mock_vector_db.search = AsyncMock(
    return_value=SearchResults(
        items=[
            SearchResultItem(
                point_id="1",
                score=0.95,
                payload={"text": "result"}
            )
        ],
        total=1
    )
)

# Create service with mocks
service = SearchService(
    embedding_provider=mock_embedding,
    vector_db_provider=mock_vector_db
)

# Test
results = service.search_documents(
    collection="test",
    query="test query",
    limit=10
)

assert len(results.items) == 1
assert results.items[0].score == 0.95
```

See `tests/unit/services/` for comprehensive test examples.

## Dependencies

Services depend on:

- **Core**: `core.exceptions`, `core.models`
- **Clients**: `clients.interfaces.*`, `clients.k8s_spark`
- **Config**: `config` module for configuration management
- **External Libraries**:
  - `attrs>=25.4.0`: For immutable service classes
  - `ray[default]>=2.40.0`: For Ray job submission
  - `kubernetes>=31.0.0`: For Spark job management

Services do NOT depend on:

- Web frameworks (FastAPI, Flask, etc.)
- HTTP request/response objects
- Database ORMs
- Framework-specific decorators

## Best Practices

### 1. Keep Services Framework-Agnostic

❌ **Bad** (couples service to FastAPI):

```python
from fastapi import HTTPException

class SearchService:
    def search(self, query: str):
        if not query:
            raise HTTPException(status_code=400, detail="Empty query")
```

✅ **Good** (framework-agnostic):

```python
from core.exceptions import BadRequestError

class SearchService:
    def search(self, query: str):
        if not query:
            raise BadRequestError("Empty query")
```

### 2. Use Dependency Injection

❌ **Bad** (hard-coded dependencies):

```python
class SearchService:
    def __init__(self):
        self.embedding = OllamaClient(...)
        self.vector_db = QdrantClient(...)
```

✅ **Good** (injected dependencies):

```python
@attrs.define(frozen=True, slots=True)
class SearchService:
    embedding_provider: EmbeddingProvider
    vector_db_provider: VectorDBProvider
```

### 3. Validate Early

Validate inputs at the service layer before calling clients:

```python
def search_documents(self, query: str, limit: int):
    # Validate early
    if not query or not query.strip():
        raise BadRequestError("Query cannot be empty")
    
    if limit < 1 or limit > 100:
        raise BadRequestError("Limit must be between 1 and 100")
    
    # Call clients only after validation
    embedding = self.embedding_provider.embed(query)
    return self.vector_db_provider.search(...)
```

### 4. Use Async for I/O-Bound Operations

Provide both sync and async versions when appropriate:

```python
def search_documents(self, query: str, limit: int) -> SearchResults:
    """Sync version - uses asyncio.run()"""
    embedding = self.embedding_provider.embed(query)
    return asyncio.run(self.vector_db_provider.search(...))

async def search_documents_async(self, query: str, limit: int) -> SearchResults:
    """Async version - native async/await"""
    embedding = await self.embedding_provider.embed_async(query)
    return await self.vector_db_provider.search(...)
```

## Future Additions

Potential new services:

- **DocumentService**: Document ingestion and preprocessing
- **CollectionService**: Vector collection management
- **EmbeddingService**: Batch embedding generation
- **MonitoringService**: Job monitoring and alerting
- **CacheService**: Result caching and invalidation
