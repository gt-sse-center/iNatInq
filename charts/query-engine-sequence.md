# Query Engine Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI Router
    participant SS as ImageSearchService
    participant Cache as Semantic Cache<br/>(In-Memory Qdrant)
    participant EP as EmbeddingProvider<br/>(CLIP / Infinity)
    participant VDP as VectorDBProvider<br/>(QdrantClientWrapper)
    participant Embed as CLIP / Infinity
    participant VDB as Qdrant

    C->>+API: GET /search/images?q=red circle&limit=10

    Note over API: Parse query params,<br/>create providers via factories

    API->>API: create_embedding_provider(config)
    API->>API: create_vector_db_provider(config)
    API->>+SS: search_images_async(collection, query, limit)

    SS->>SS: Validate query (non-empty, limit 1-100)

    Note over SS: Step 1: Generate Query Embedding

    SS->>+EP: embed_text(query="red circle")
    EP->>+Embed: POST /embed {"inputs": ["red circle"]}
    Embed-->>-EP: {"embeddings": [[0.123, -0.456, ...]]}
    EP-->>-SS: vector[512]

    Note over SS: Step 2: Check Semantic Cache

    SS->>+Cache: lookup(collection, query_vector, limit)

    alt Cache Hit (similarity >= 0.95)
        Cache-->>SS: SearchResults (cached)
        Note over SS: Return cached results
    else Cache Miss
        Cache-->>-SS: None

        Note over SS: Step 3: Vector Similarity Search

        SS->>+VDP: search_async(collection, query_vector, limit=10)
        VDP->>+VDB: POST /collections/{name}/points/search
        VDB-->>-VDP: [{id, score, payload}, ...]
        VDP-->>-SS: SearchResults(items)

        Note over SS: Step 4: Store in Cache

        SS->>Cache: store(collection, query_vector, query_text, results, limit)
    end

    SS-->>-API: SearchResults

    API->>API: Convert to ImageSearchResponse

    API-->>-C: 200 OK<br/>{"query": "red circle",<br/>"results": [...], "total": 10}
```

## Sequence Description

| Step | Component | Action |
| ---- | --------- | ------ |
| 1-2 | Client → API | Send search request with query and parameters |
| 3-5 | API | Create providers via factory, instantiate ImageSearchService |
| 6 | ImageSearchService | Validate input (query non-empty, limit in range) |
| 7-9 | ImageSearchService → Embedding | Generate embedding vector for query text |
| 10 | ImageSearchService → Cache | Cosine similarity lookup against cached queries |
| 11 | (Cache Hit) | Return stored results, skip vector DB |
| 12-15 | (Cache Miss) → VectorDB | Perform cosine similarity search in Qdrant |
| 16 | ImageSearchService → Cache | Store results for future similar queries |
| 17-18 | API → Client | Format and return ranked results |

## Error Handling

- **400 Bad Request**: Empty query, invalid limit, invalid provider
- **502 Bad Gateway**: CLIP/Infinity or Qdrant service failure (UpstreamError)
- **504 Gateway Timeout**: Operation timeout (PipelineTimeoutError)

## Semantic Cache Details

- **Backend**: In-memory Qdrant instance (`:memory:`)
- **Similarity threshold**: 0.95 (configurable via `SEMANTIC_CACHE_SIMILARITY_THRESHOLD`)
- **Max entries**: 1000 per collection (random eviction at capacity)
- **Invalidation**: `DELETE /cache` endpoint or automatic sweep every 3600s
- **Graceful degradation**: Cache failures are logged and swallowed; search continues without cache
