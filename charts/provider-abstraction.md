# Provider Abstraction Class Diagram

Shows the ABC pattern for embedding and vector DB providers, with factory functions and registry.

```mermaid
classDiagram
    direction TB

    class EmbeddingProvider {
        <<abstract>>
        +embed_text(text: str) list~float~*
        +embed_text_batch(texts: list~str~) list~list~float~~*
        +embed_image(image_bytes: bytes) list~float~*
        +embed_image_batch(images: list~bytes~) list~list~float~~*
        +close()
        +vector_size: int*
        +model_name: str*
        +from_config(config: EmbeddingConfig) EmbeddingProvider$
    }

    class CLIPClient {
        -_url: str
        -_model: str
        -_breaker: CircuitBreaker
        -_async_breaker: CircuitBreaker
        +embed_text(text) list~float~
        +embed_image(image_bytes) list~float~
        +vector_size: int
        +model_name: str
    }

    class InfinityClient {
        -_url: str
        -_model: str
        -_breaker: CircuitBreaker
        -_async_breaker: CircuitBreaker
        +embed_text(text) list~float~
        +embed_image(image_bytes) list~float~
        +vector_size: int
        +model_name: str
    }

    class VectorDBProvider {
        <<abstract>>
        +ensure_collection_async(collection, vector_size)*
        +search_async(collection, query_vector, limit) SearchResults*
        +get_collection_info_async(collection) CollectionInfo*
        +batch_upsert_async(collection, points, vector_size)*
        +close()*
        +from_config(config: VectorDBConfig) VectorDBProvider$
    }

    class QdrantClientWrapper {
        -_client: AsyncQdrantClient
        -_breaker: CircuitBreaker
        -_async_breaker: CircuitBreaker
        +ensure_collection_async(collection, vector_size)
        +search_async(collection, query_vector, limit) SearchResults
        +batch_upsert_async(collection, points, vector_size)
        +close()
    }

    class EmbeddingConfig {
        +provider_type: ProviderType
        +url: str
        +model: str
        +from_env() EmbeddingConfig$
    }

    class VectorDBConfig {
        +provider: str
        +url: str
        +from_env() VectorDBConfig$
    }

    class ErrorClassifier {
        <<protocol>>
        +is_retriable(exc: BaseException) bool
        +get_error_details(exc: BaseException) dict
    }

    class CircuitBreakerMixin {
        <<mixin>>
        -_breaker: CircuitBreaker
        -_async_breaker: CircuitBreaker
        +_init_circuit_breaker()
        +_circuit_breaker_config() tuple
    }

    EmbeddingProvider <|-- CLIPClient : implements
    EmbeddingProvider <|-- InfinityClient : implements
    VectorDBProvider <|-- QdrantClientWrapper : implements
    CircuitBreakerMixin <|-- CLIPClient : uses
    CircuitBreakerMixin <|-- InfinityClient : uses
    CircuitBreakerMixin <|-- QdrantClientWrapper : uses

    EmbeddingConfig ..> EmbeddingProvider : from_config()
    VectorDBConfig ..> VectorDBProvider : from_config()

    note for EmbeddingProvider "Factory: create_embedding_provider(config)\nRegistry: _PROVIDER_REGISTRY[ProviderType → class]"
    note for VectorDBProvider "Factory: create_vector_db_provider(config)\nRegistry: _PROVIDER_REGISTRY[str → class]"
```

## Key Concepts

### Factory + Registry Pattern

Providers register themselves at import time via `register_provider()`. The factory function looks up the provider class by type and calls `from_config()`:

```
EmbeddingConfig(provider_type="clip")
    → _PROVIDER_REGISTRY["clip"] → CLIPClient
    → CLIPClient.from_config(config) → instance
```

### Adding a New Provider

1. Create class inheriting from `EmbeddingProvider` or `VectorDBProvider`
2. Implement all abstract methods
3. Call `register_provider("my_provider", MyProvider)` at module level
4. Add `ProviderType` enum value in `config.py`

### Circuit Breaker Integration

Every client uses dual circuit breakers (sync + async) via `CircuitBreakerMixin`:
- `@with_circuit_breaker("service")` wraps sync methods
- `@with_circuit_breaker_async("service")` wraps async methods
- `CircuitBreakerError` → `UpstreamError` conversion is automatic
