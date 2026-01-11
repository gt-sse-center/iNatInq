# Provider Interfaces

Abstract Base Classes (ABCs) that define provider-agnostic interfaces for
embedding generation and vector database operations.

This sub-package contains the abstraction layer that allows the pipeline to work
with different providers (Ollama, OpenAI, Qdrant, Weaviate, etc.) without code
changes.

## Design Principles

1. **Provider Agnostic**: Services work with ABCs, not concrete implementations
2. **Extensibility**: New providers can be added by implementing the ABC
3. **Factory Pattern**: Factory functions create providers from configuration
4. **Type Safety**: ABCs provide clear contracts with type hints
5. **Configuration Driven**: Providers are created from configuration objects

## Architecture

```
┌─────────────────┐
│   Services      │  Use ABCs (provider-agnostic)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Interfaces    │  ABCs define contracts
│   (this module) │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Ollama  │ │OpenAI  │  Concrete implementations
│Client  │ │Client  │  (in parent clients/ package)
└────────┘ └────────┘
```

## Modules

### `embedding.py`

Abstract interface for embedding generation providers.

**ABC:** `EmbeddingProvider`

**Abstract Methods:**

- `embed(text: str) -> list[float]`: Generate embedding for a single text (sync)
- `embed_async(text: str) -> list[float]`: Generate embedding for a single text
  (async)
- `vector_size: int`: Property returning the embedding dimension
- `from_config(config: EmbeddingConfig, session: requests.Session | None = None) -> EmbeddingProvider`:
  Class method to create instance from config

**Configuration:** `EmbeddingConfig`

- `provider_type: EmbeddingProviderType`: Provider type ("ollama", "openai",
  etc.)
- `vector_size: int | None`: Expected vector dimension (optional, auto-detected)
- Provider-specific fields (e.g., `ollama_url`, `ollama_model`,
  `openai_api_key`)

**Factory Function:**
`create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider`

**Usage:**

```python
from clients.interfaces.embedding import (
    EmbeddingProvider,
    EmbeddingConfig,
    create_embedding_provider,
)

# Create config from environment
config = EmbeddingConfig.from_env()

# Create provider (returns OllamaClient, OpenAIClient, etc.)
provider = create_embedding_provider(config)

# Use provider (agnostic to implementation)
vector = provider.embed("hello world")
vector_size = provider.vector_size
```

**Implementations:**

- `OllamaClient` (in `clients/ollama.py`)
- `OpenAIClient` (future)
- `HuggingFaceClient` (future)

**Registering New Providers:**

```python
from clients.interfaces.embedding import (
    EmbeddingProvider,
    register_provider,
)

class MyEmbeddingClient(EmbeddingProvider):
    # Implement abstract methods
    ...

# Register the provider
register_provider("myprovider", MyEmbeddingClient)

# Now it can be used via factory
config = EmbeddingConfig(provider_type="myprovider", ...)
provider = create_embedding_provider(config)
```

### `vector_db.py`

Abstract interface for vector database providers.

**ABC:** `VectorDBProvider`

**Abstract Methods:**

- `ensure_collection(collection: str, vector_size: int) -> None`: Ensure
  collection exists
- `search(collection: str, query_vector: list[float], limit: int = 10) -> SearchResults`:
  Search for similar vectors
- `batch_upsert(collection: str, points: list, vector_size: int) -> None`: Batch
  upsert points
- `from_config(config: VectorDBConfig) -> VectorDBProvider`: Class method to
  create instance from config

**Configuration:** `VectorDBConfig`

- `provider_type: VectorDBProviderType`: Provider type ("qdrant", "weaviate",
  etc.)
- Provider-specific fields (e.g., `qdrant_url`, `weaviate_url`)

**Factory Function:**
`create_vector_db_provider(config: VectorDBConfig) -> VectorDBProvider`

**Usage:**

```python
from clients.interfaces.vector_db import (
    VectorDBProvider,
    VectorDBConfig,
    create_vector_db_provider,
)

# Create config from environment
config = VectorDBConfig.from_env()

# Create provider (returns QdrantClientWrapper, WeaviateClient, etc.)
provider = create_vector_db_provider(config)

# Use provider (agnostic to implementation)
provider.ensure_collection(collection="documents", vector_size=768)
results = provider.search(
    collection="documents",
    query_vector=[0.1, 0.2, ...],
    limit=10
)
```

**Implementations:**

- `QdrantClientWrapper` (in `clients/qdrant.py`)
- `WeaviateClient` (future)

**Registering New Providers:**

```python
from clients.interfaces.vector_db import (
    VectorDBProvider,
    register_provider,
)

class MyVectorDBClient(VectorDBProvider):
    # Implement abstract methods
    ...

# Register the provider
register_provider("myvectordb", MyVectorDBClient)

# Now it can be used via factory
config = VectorDBConfig(provider_type="myvectordb", ...)
provider = create_vector_db_provider(config)
```

## Provider Registry

Both ABCs use a registry pattern to map provider types to implementation
classes:

- **Registration**: Providers register themselves via `register_provider()`
- **Discovery**: Factory functions look up providers in the registry
- **Extensibility**: New providers can be registered at runtime

Default providers (Ollama, Qdrant) are registered automatically on module
import.

## Configuration

Configuration classes (`EmbeddingConfig`, `VectorDBConfig`) support:

- **Environment Variables**: `from_env()` class method loads from environment
- **Provider Detection**: Automatically detects provider type from environment
- **Backward Compatibility**: Supports legacy environment variable names
- **Type Safety**: Uses attrs for validation and type hints

**Example:**

```python
# Environment variables
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=nomic-embed-text

# Load config
config = EmbeddingConfig.from_env()
provider = create_embedding_provider(config)
```

## Benefits

1. **Swappable Providers**: Change providers via configuration, no code changes
2. **Testability**: Mock providers easily using ABC interface
3. **Extensibility**: Add new providers by implementing ABC and registering
4. **Type Safety**: ABCs provide clear contracts with type hints
5. **Consistency**: All providers follow the same interface

## Testing

Mock providers using the ABC interface:

```python
from unittest.mock import Mock
from clients.interfaces.embedding import EmbeddingProvider

# Create mock provider
mock_provider = Mock(spec=EmbeddingProvider)
mock_provider.embed.return_value = [0.1, 0.2, ...]
mock_provider.vector_size = 768

# Use in services (works with any EmbeddingProvider)
from core.services.embedding_service import EmbeddingService
service = EmbeddingService(embedding_provider=mock_provider)
```

## Dependencies

- `attrs`: Configuration classes
- `requests`: HTTP session support (for embedding providers)
- `abc`: Abstract Base Classes
