# Ingestion Engine Flow Chart

```mermaid
flowchart TB
    subgraph Client
        A[HTTP Request<br/>POST /ray/jobs or /spark/jobs]
    end

    subgraph API["FastAPI Routes"]
        B[Parse Job Request]
        C{Engine<br/>Type?}
        D[RayService]
        E[SparkService]
    end

    subgraph JobSubmission["Job Submission"]
        F[Submit to Ray Cluster]
        G[Create SparkApplication<br/>in Kubernetes]
    end

    subgraph Processing["Distributed Processing"]
        H[List S3 Objects]
        I[Partition Keys<br/>Across Workers]
        J[Fetch S3 Content]
        K[Rate-Limited<br/>Embedding Generation]
        L[Create Vector Points]
        M[Batch Upsert to<br/>Vector Databases]
    end

    subgraph Workers["Worker Tasks"]
        N[Ray Remote Task]
        O[Spark RDD Partition]
    end

    subgraph External["External Services"]
        P[(MinIO/S3)]
        Q[(Ollama)]
        R[(Qdrant)]
        S[(Weaviate)]
    end

    A --> B
    B --> C
    C -->|Ray| D
    C -->|Spark| E
    D --> F
    E --> G
    F --> N
    G --> O
    N --> H
    O --> H
    H --> P
    P --> I
    I --> J
    J --> P
    P --> K
    K --> Q
    Q --> L
    L --> M
    M --> R
    M --> S

    subgraph Response["Job Response"]
        T[202 Accepted<br/>+ Job ID]
    end

    D --> T
    E --> T

    style A fill:#e1f5fe
    style T fill:#c8e6c9
    style P fill:#fff9c4
    style Q fill:#fff3e0
    style R fill:#fce4ec
    style S fill:#f3e5f5
    style N fill:#e8f5e9
    style O fill:#e3f2fd
```

## Flow Description

1. **Client Request**: User submits ingestion job via `POST /ray/jobs` or `POST /spark/jobs`
2. **API Layer**: Routes to appropriate service (RayService or SparkService)
3. **Job Submission**: Creates distributed job in Ray cluster or Kubernetes SparkApplication
4. **Object Discovery**: Lists S3 objects matching the prefix
5. **Parallel Processing**: Partitions work across Ray tasks or Spark executors
6. **Content Fetch**: Each worker fetches assigned S3 objects
7. **Embedding Generation**: Rate-limited calls to Ollama for vector embeddings
8. **Vector Point Creation**: Constructs points with embeddings + metadata
9. **Database Upsert**: Batch upserts to both Qdrant and Weaviate
10. **Async Response**: Returns job ID immediately (202 Accepted)
