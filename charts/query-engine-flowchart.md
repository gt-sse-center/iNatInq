# Query Engine Flow Chart

```mermaid
flowchart TB
    subgraph Client
        A[HTTP Request<br/>GET /search?q=...]
    end

    subgraph API["FastAPI Routes"]
        B[Parse Query Parameters]
        C{Provider<br/>specified?}
        D[Use Default<br/>from Settings]
        E[Create Provider Config]
    end

    subgraph Service["SearchService"]
        F[Validate Query]
        G[Generate Query Embedding]
        H[Search Vector Database]
        I[Format Results]
    end

    subgraph Providers["Provider Layer"]
        J[EmbeddingProvider]
        L[Qdrant Client]
    end

    subgraph External["External Services"]
        N[(CLIP Service)]
        O[(Qdrant)]
    end

    A --> B
    B --> C
    C -->|No| D
    C -->|Yes| E
    D --> F
    E --> F
    F -->|Valid| G
    F -->|Invalid| Z[400 Bad Request]
    G --> J
    J --> N
    N --> H
    H --> L
    L --> O
    O --> I
    I --> Y[JSON Response]

    style A fill:#e1f5fe
    style Y fill:#c8e6c9
    style Z fill:#ffcdd2
    style N fill:#fff3e0
    style O fill:#fce4ec
```

## Flow Description

1. **Client Request**: User sends semantic search query via `GET /search?q=...`
2. **API Layer**: Parses parameters, determines vector DB provider
3. **Service Layer**: Validates query, orchestrates embedding + search
4. **Provider Layer**: Abstracts embedding and vector DB implementations
5. **External Services**: CLIP generates embeddings, Qdrant performs similarity search
6. **Response**: Ranked results with similarity scores returned to client
