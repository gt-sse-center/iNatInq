# Query Engine Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI Router
    participant SS as SearchService
    participant EP as EmbeddingProvider
    participant VDP as VectorDBProvider
    participant CLIP as CLIP Service
    participant VDB as Vector Database<br/>(Qdrant)

    C->>+API: GET /search?q=machine learning&limit=10
    
    Note over API: Parse query params,<br/>resolve provider config
    
    API->>API: create_embedding_provider(config)
    API->>API: create_vector_db_provider(config)
    API->>+SS: SearchService(embedding_provider, vector_db_provider)

    SS->>SS: Validate query (non-empty, limit 1-100)

    Note over SS: Step 1: Generate Query Embedding
    
    SS->>+EP: embed(query="machine learning")
    EP->>+CLIP: POST /embed<br/>{"inputs": ["machine learning"]}
    CLIP-->>-EP: {"embeddings": [[0.123, -0.456, ...]]}
    EP-->>-SS: vector[512]

    Note over SS: Step 2: Vector Similarity Search
    
    SS->>+VDP: search_async(collection, query_vector, limit=10)
    
    VDP->>+VDB: POST /collections/{name}/points/search
    VDB-->>-VDP: [{id, score, payload}, ...]
    
    VDP-->>-SS: SearchResults(items, total)

    Note over SS: Step 3: Format Response
    
    SS-->>-API: SearchResults

    API->>API: Convert to Pydantic models
    
    API-->>-C: 200 OK<br/>{"query": "machine learning",<br/>"results": [...], "total": 10}
```

## Sequence Description

| Step | Component | Action |
| ---- | --------- | ------ |
| 1-2 | Client → API | Send search request with query and parameters |
| 3-5 | API | Create providers, instantiate SearchService |
| 6 | SearchService | Validate input (query non-empty, limit in range) |
| 7-9 | SearchService → CLIP | Generate embedding vector for query text |
| 10-13 | SearchService → VectorDB | Perform cosine similarity search |
| 14-16 | API → Client | Format and return ranked results |

## Error Handling

- **400 Bad Request**: Empty query, invalid limit, invalid provider
- **502 Bad Gateway**: CLIP or Vector DB service failure
- **404 Not Found**: Collection doesn't exist
