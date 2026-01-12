# Core

Core domain models, exceptions, and shared types for the iNatInq platform.

This directory contains fundamental data structures and exception classes that are shared across the application, providing a common vocabulary for the entire system.

## Design Principles

1. **Domain-Driven**: Models represent domain concepts, not implementation details
2. **Type Safety**: All models use attrs with type hints for validation
3. **Immutability**: Models are frozen by default to prevent accidental mutations
4. **Serializable**: Models can be easily converted to/from JSON
5. **Framework Agnostic**: No dependencies on web frameworks or external services

## Directory Structure

```
core/
├── __init__.py          # Package exports
├── exceptions.py        # Exception hierarchy
├── models.py            # Domain models and data structures
└── services/            # Business logic orchestration layer
    ├── __init__.py
    ├── README.md
    ├── search_service.py    # Semantic search orchestration
    ├── spark_service.py     # Spark job management
    └── ray_service.py       # Ray job orchestration
```

## Modules

### `services/`

Business logic orchestration layer that coordinates multiple clients to accomplish workflows.

**Purpose:**

- Provides framework-agnostic business logic
- Orchestrates multiple clients for complex operations  
- Implements validation and error handling
- Can be used from APIs, CLIs, or tests

**Services:**

- `SearchService`: Semantic search orchestration
- `SparkService`: Spark job management via Kubernetes
- `RayService`: Ray job orchestration

See `services/README.md` for detailed documentation.

### `exceptions.py`

Core exception classes for the application.

**Purpose:**

- Provides a consistent exception hierarchy
- Re-exports foundation exceptions for convenience
- Centralizes error handling patterns

**Classes:**

- `PipelineError`: Base exception class for all application errors
- `BadRequestError`: Exception for invalid client input (HTTP 400)
- `UpstreamError`: Exception when external services fail (HTTP 502) - re-exported from `foundation.exceptions`
- `PipelineTimeoutError`: Exception when operations timeout (HTTP 504)

**Usage:**

```python
from core.exceptions import UpstreamError, BadRequestError

# Raise when an external service fails
raise UpstreamError(f"Failed to connect to Qdrant: {error}")

# Raise when client provides invalid input
raise BadRequestError("Text list cannot be empty")
```

**Note:** `UpstreamError` is re-exported from `foundation.exceptions` to ensure consistency across the codebase, as it's raised by circuit breakers and client wrappers.

### `models.py`

Domain models and data structures used across the application.

**Purpose:**

- Defines core data types for vector search and storage
- Provides type-safe data structures using attrs
- Ensures consistency across services and clients

**Classes:**

#### `SearchResultItem`

Represents a single search result from a vector database.

**Attributes:**

- `point_id: str`: Unique identifier for the point/document
- `score: float`: Similarity score (0.0 to 1.0, higher = more similar)
- `payload: dict[str, Any]`: Associated metadata/data

**Usage:**

```python
from core.models import SearchResultItem

result = SearchResultItem(
    point_id="doc-123",
    score=0.95,
    payload={"text": "Example document", "source": "wikipedia"}
)

print(f"Found document {result.point_id} with score {result.score}")
print(f"Text: {result.payload['text']}")
```

#### `SearchResults`

Container for multiple search results.

**Attributes:**

- `items: list[SearchResultItem]`: List of search results, ordered by similarity

**Usage:**

```python
from core.models import SearchResults, SearchResultItem

results = SearchResults(
    items=[
        SearchResultItem(point_id="doc-1", score=0.95, payload={"text": "..."}),
        SearchResultItem(point_id="doc-2", score=0.87, payload={"text": "..."}),
    ]
)

# Iterate over results
for result in results.items:
    print(f"{result.point_id}: {result.score}")

# Access best result
best = results.items[0]
```

#### `VectorPoint`

Represents a vector point for upserting to a vector database.

**Attributes:**

- `id: str`: Unique identifier for the point
- `vector: list[float]`: Embedding vector (dimension must match collection)
- `payload: dict[str, Any]`: Associated metadata/data

**Usage:**

```python
from core.models import VectorPoint

point = VectorPoint(
    id="doc-123",
    vector=[0.1, 0.2, 0.3, ...],  # 768-dimensional vector
    payload={"text": "Example document", "source": "wikipedia"}
)

# Use with vector database client
client.batch_upsert(
    collection="documents",
    points=[point],
    vector_size=768
)
```

**Note:** The `VectorPoint` class is compatible with `qdrant_client.models.PointStruct` and can be used interchangeably in most cases.

## Integration

Core models are used throughout the application:

### In Clients

```python
from clients.qdrant import QdrantClientWrapper
from core.models import SearchResults, VectorPoint

# Search returns SearchResults
client = QdrantClientWrapper(url="http://qdrant:6333")
results: SearchResults = client.search(
    collection="documents",
    query_vector=[...],
    limit=10
)

# Upsert uses VectorPoint
points = [
    VectorPoint(id="1", vector=[...], payload={"text": "..."}),
    VectorPoint(id="2", vector=[...], payload={"text": "..."}),
]
client.batch_upsert(collection="documents", points=points, vector_size=768)
```

### In Services

```python
from core.models import SearchResults
from core.exceptions import UpstreamError

def search_service(query: str) -> SearchResults:
    try:
        # Generate embedding
        vector = embedding_client.embed(query)
        
        # Search vector database
        results = vector_db_client.search(
            collection="documents",
            query_vector=vector,
            limit=10
        )
        
        return results
    except Exception as e:
        raise UpstreamError(f"Search failed: {e}")
```

## Type Safety

All models use attrs with:

- **Type hints**: All fields have explicit type annotations
- **Frozen**: Models are immutable by default
- **Slots**: Reduces memory usage and improves performance
- **Validation**: Automatic type checking and validation

**Example:**

```python
from core.models import SearchResultItem

# Valid
result = SearchResultItem(point_id="doc-1", score=0.95, payload={})

# Invalid - will raise TypeError
result = SearchResultItem(point_id="doc-1", score="not a float", payload={})

# Immutable - will raise AttributeError
result.score = 0.5  # Cannot modify frozen dataclass
```

## Serialization

Models can be easily serialized to/from dictionaries and JSON:

```python
from core.models import SearchResultItem
import attrs

# Create model
result = SearchResultItem(point_id="doc-1", score=0.95, payload={"text": "..."})

# Convert to dict
result_dict = attrs.asdict(result)
# {'point_id': 'doc-1', 'score': 0.95, 'payload': {'text': '...'}}

# Convert from dict
result = SearchResultItem(**result_dict)

# JSON serialization
import json
json_str = json.dumps(attrs.asdict(result))
```

## Testing

Core models are thoroughly tested:

```python
from core.models import SearchResultItem, SearchResults

def test_search_result():
    result = SearchResultItem(point_id="test", score=0.9, payload={"key": "value"})
    assert result.point_id == "test"
    assert result.score == 0.9
    assert result.payload["key"] == "value"

def test_search_results():
    results = SearchResults(items=[
        SearchResultItem(point_id="1", score=0.9, payload={}),
        SearchResultItem(point_id="2", score=0.8, payload={}),
    ])
    assert len(results.items) == 2
    assert results.items[0].score > results.items[1].score
```

## Dependencies

Core models have minimal dependencies:

- `attrs>=25.4.0`: For dataclass-like functionality
- Python standard library

Core does NOT depend on:

- Web frameworks (FastAPI, Flask, etc.)
- External service clients
- Database libraries
- Configuration management

## Future Additions

Potential additions to the core directory:

- **Additional Models**: More domain models as needed (e.g., Document, Collection, Job)
- **Value Objects**: Domain value objects (e.g., VectorEmbedding, CollectionName)
- **Enums**: Shared enumerations (e.g., JobStatus, CollectionStatus)
- **Constants**: Shared constants and configuration values
