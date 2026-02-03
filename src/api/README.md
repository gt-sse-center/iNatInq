# API Routes

FastAPI HTTP routes for the pipeline service.

This directory contains the HTTP API layer that handles request/response
serialization, dependency injection, and error translation.

## Design Principles

1. **Thin Controllers**: Routes are thin wrappers around service layer functions
2. **Error Translation**: Routes translate `PipelineError` hierarchy to HTTP
   status codes
3. **Request Validation**: Routes use Pydantic models for automatic request
   validation
4. **OpenAPI Documentation**: Routes include docstrings for automatic OpenAPI
   schema generation

## Architecture

```
HTTP Request
    ↓
API Routes (this module)
    ↓
Services (business logic)
    ↓
Clients (external services)
```

Routes handle:

- HTTP concerns (status codes, headers, serialization)
- Request/response validation via Pydantic
- Dependency injection (settings, Kubernetes clients)
- Error translation (`PipelineError` → HTTP status codes)

Routes do NOT handle:

- Business logic (delegated to services)
- External service calls (delegated to clients)
- Configuration loading (delegated to `config` module)

## Endpoints

### `GET /healthz`

Liveness and readiness probe endpoint.

**Purpose:**

- Used by Kubernetes liveness and readiness probes
- Returns simple success response without checking dependencies

**Response:**

```json
{
  "status": "ok"
}
```

**Note:** We intentionally do not check dependencies here because:

- Dependencies can be slow to start in dev environments
- Dependency checks can cause restart loops if services are temporarily
  unavailable
- The service can start accepting requests even if some dependencies aren't
  ready yet

### `POST /embed`

Generate embeddings for one or more text strings via embedding provider (Ollama,
OpenAI, etc.).

**Request:**

```json
{
  "texts": ["hello world", "foo bar"],
  "model": "nomic-embed-text" // optional, uses default if not provided
}
```

**Response:**

```json
{
  "model": "nomic-embed-text",
  "embeddings": [
    [0.1, 0.2, ...],  // 768-dimensional vector
    [0.3, 0.4, ...]   // 768-dimensional vector
  ]
}
```

**Error Responses:**

- `502 Bad Gateway`: Embedding provider is unreachable or returns an error

### `GET /search` (Semantic Search)

Perform semantic search over documents in vector database (Qdrant or Weaviate).

**Query Parameters:**

- `q` (required): Natural language query string
- `limit` (optional, default: 10): Maximum number of results (1-100)
- `collection` (optional): Collection name override
- `model` (optional): Model name override
- `provider` (optional): Vector database provider ("qdrant" or "weaviate"). If
  not specified, uses `VECTOR_DB_PROVIDER` environment variable (defaults to
  "qdrant").

**Response:**

```json
{
  "query": "machine learning pipeline",
  "model": "nomic-embed-text",
  "collection": "documents",
  "results": [
    {
      "id": "d790dd2c-99eb-4901-b9c9-538b58318fe3",
      "score": 0.9234,
      "text": "s3://pipeline/inputs/hello-a01f74c0.txt",
      "metadata": {
        "s3_key": "inputs/hello-a01f74c0.txt"
      }
    }
  ],
  "total": 1
}
```

**Error Responses:**

- `400 Bad Request`: Query is empty or limit is invalid
- `502 Bad Gateway`: Embedding provider or vector database operations fail
- `404 Not Found`: Collection doesn't exist

**Note:**

- Results are ordered by similarity score (highest first)
- Score ranges from 0.0 to 1.0 (1.0 = identical, 0.0 = completely different)
- Uses cosine similarity for vector comparison
- Collection must exist (will not be auto-created for search)

### `POST /upsert`

Embed texts and upsert the resulting vectors into Qdrant.

**Request:**

```json
{
  "texts": ["document 1", "document 2"],
  "collection": "my-docs", // optional, uses default if not provided
  "metadata": [{ "source": "file1.txt" }, { "source": "file2.txt" }], // optional
  "model": "nomic-embed-text" // optional, uses default if not provided
}
```

**Response:**

```json
{
  "collection": "my-docs",
  "model": "nomic-embed-text",
  "points_upserted": 2
}
```

**Error Responses:**

- `400 Bad Request`: Metadata length doesn't match texts length
- `502 Bad Gateway`: Embedding provider or vector database operations fail

**Note:** The collection is created automatically if it doesn't exist (dev
convenience).

### Semantic Search Endpoint

The `GET /search` endpoint performs semantic search over documents in the vector
database (Qdrant or Weaviate).

**Query Parameters:**

- `q` (required): Natural language query string
- `limit` (optional, default: 10): Maximum number of results (1-100)
- `collection` (optional): Collection name override
- `model` (optional): Model name override
- `provider` (optional): Vector database provider ("qdrant" or "weaviate"). If
  not specified, uses `VECTOR_DB_PROVIDER` environment variable (defaults to
  "qdrant").

**Response:**

```json
{
  "query": "machine learning pipeline",
  "model": "nomic-embed-text",
  "collection": "documents",
  "provider": "qdrant",
  "results": [
    {
      "id": "d790dd2c-99eb-4901-b9c9-538b58318fe3",
      "score": 0.9234,
      "text": "s3://pipeline/inputs/hello-a01f74c0.txt",
      "metadata": {
        "s3_key": "inputs/hello-a01f74c0.txt"
      }
    }
  ],
  "total": 1
}
```

**Error Responses:**

- `400 Bad Request`: Query is empty, limit is invalid, or provider is invalid
- `502 Bad Gateway`: Embedding provider or vector database operations fail
- `404 Not Found`: Collection doesn't exist

**Note:**

- Results are ordered by similarity score (highest first)
- Score ranges from 0.0 to 1.0 (1.0 = identical, 0.0 = completely different)
- Uses cosine similarity for vector comparison
- Collection must exist (will not be auto-created for search)
- Both Qdrant and Weaviate can be used simultaneously (specify provider per
  request)

### `POST /process`

Submit a Spark job to process S3 data and store embeddings in the vector
database.

**Purpose:**

- Submits a Spark job that processes S3 objects in parallel
- The Spark job generates embeddings via embedding provider (Ollama, OpenAI,
  etc.)
- The Spark job upserts embeddings into vector database (Qdrant, Weaviate, etc.)
- Waits for job completion (synchronous, blocking)

**Request:**

```json
{
  "s3_prefix": "inputs/",
  "timeout_s": 600,
  "collection": "documents",
  "model": "nomic-embed-text"
}
```

All fields are optional:

- `s3_prefix`: S3 prefix to filter objects (default: `inputs/`)
- `timeout_s`: Timeout in seconds (default: 600, range: 60-3600)
- `collection`: Vector database collection name (defaults to service default)
- `model`: Embedding model name (defaults to service default, provider-specific)

**Response:**

```json
{
  "job_name": "s3-to-qdrant-abc12345",
  "s3_prefix": "inputs/",
  "collection": "documents",
  "model": "nomic-embed-text",
  "status": "completed"
}
```

**Error Responses:**

- `502 Bad Gateway`: Job submission fails, job fails, or Kubernetes operations
  fail
- `504 Gateway Timeout`: Job doesn't complete within timeout

**Note:**

- This is a synchronous, blocking operation
- The Spark job processes S3 objects in parallel across Spark executors
- Job can take several minutes depending on data volume
- Uses service defaults for S3 endpoint, vector database URL, etc. (configured
  via environment variables)

## Error Handling

All service layer exceptions (`PipelineError` hierarchy) are translated to HTTP
status codes:

| Exception Type         | HTTP Status Code          |
| ---------------------- | ------------------------- |
| `BadRequestError`      | 400 Bad Request           |
| `UpstreamError`        | 502 Bad Gateway           |
| `PipelineTimeoutError` | 504 Gateway Timeout       |
| `PipelineError` (base) | 500 Internal Server Error |

The `_raise_http()` helper function performs this translation.

## OpenAPI Documentation

FastAPI automatically generates OpenAPI/Swagger documentation from route
docstrings:

- **Interactive UI**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc`
- **JSON Schema**: `/openapi.json`

Route docstrings should include:

- Purpose and description
- Request/response examples
- Error conditions
- Notes about behavior

## Dependencies

Routes depend on:

- `pipeline.core.services.*`: Business logic orchestration (following Go
  codebase pattern)
- `pipeline.clients.k8s`: Kubernetes client for Spark jobs
- `pipeline.config`: Configuration settings
- `pipeline.models`: Pydantic request/response models
- `pipeline.errors`: Error hierarchy

## Testing

Routes can be tested using FastAPI's `TestClient`:

```python
from fastapi.testclient import TestClient
from pipeline.app import create_app

app = create_app()
client = TestClient(app)

# Test healthz endpoint
response = client.get("/healthz")
assert response.status_code == 200
assert response.json() == {"status": "ok"}

# Test embed endpoint
response = client.post("/embed", json={"texts": ["hello"]})
assert response.status_code == 200
assert "embeddings" in response.json()

# Test search endpoint
response = client.get("/search?q=machine%20learning&limit=5")
assert response.status_code == 200
data = response.json()
assert "results" in data
assert "query" in data
assert len(data["results"]) > 0
assert "score" in data["results"][0]
```

## Request/Response Models

All request and response models are defined in `pipeline.models`:

- `EmbedRequest`: Request model for `/embed`
- `EmbedResponse`: Response model for `/embed`
- `UpsertRequest`: Request model for `/upsert`
- `UpsertResponse`: Response model for `/upsert`
- `SearchResult`: Individual search result (id, score, text, metadata)
- `SearchResponse`: Response model for `/search`

These models provide:

- Automatic request validation
- Type safety
- OpenAPI schema generation
- Documentation
