# Query Engine Flow Chart

```mermaid
flowchart TB
    subgraph Client
        A[HTTP Request<br/>GET /search/images?q=...&limit=10]
    end

    subgraph API["FastAPI Routes"]
        B[Parse Query Parameters]
        C[Create EmbeddingProvider<br/>+ VectorDBProvider]
        D[Instantiate ImageSearchService]
    end

    subgraph Service["ImageSearchService"]
        E[Validate Query]
        F[Generate Query Embedding]
        G{Semantic Cache<br/>Lookup}
        H[Search Vector Database]
        I[Store Results in Cache]
        J[Format Results]
    end

    subgraph Providers["Provider Layer"]
        K[EmbeddingProvider<br/>CLIP / Infinity]
        L[VectorDBProvider<br/>QdrantClientWrapper]
        M[CacheClient<br/>In-Memory Qdrant]
    end

    subgraph External["External Services"]
        N[(CLIP / Infinity)]
        O[(Qdrant)]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E -->|Valid| F
    E -->|Invalid| Z[400 Bad Request]
    F --> K
    K --> N
    N --> G
    G -->|Cache Hit| J
    G -->|Cache Miss| H
    H --> L
    L --> O
    O --> I
    I --> M
    M --> J
    J --> Y[200 OK<br/>ImageSearchResponse]

    style A fill:#e1f5fe
    style Y fill:#c8e6c9
    style Z fill:#ffcdd2
    style N fill:#fff3e0
    style O fill:#fce4ec
    style M fill:#e8eaf6
```

## Flow Description

1. **Client Request**: User sends semantic search query via `GET /search/images?q=...&limit=10`
2. **API Layer**: Parses parameters, creates embedding and vector DB providers via factory functions
3. **Service Layer**: `ImageSearchService` validates query, orchestrates embedding + cache + search
4. **Embedding**: Generates query vector via `EmbeddingProvider` (CLIP or Infinity)
5. **Cache Check**: Looks up query embedding in semantic cache (cosine similarity >= 0.95 threshold)
6. **Cache Hit**: Returns cached results immediately (skips vector DB)
7. **Cache Miss**: Searches Qdrant for similar vectors, stores results in cache for future queries
8. **Response**: Returns `ImageSearchResponse` with ranked results and similarity scores
