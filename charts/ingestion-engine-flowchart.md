# Ingestion Engine Flow Chart

```mermaid
flowchart TB
    subgraph Client
        A[HTTP Request<br/>POST /ray/jobs/images or /databricks/jobs/images]
    end

    subgraph API["FastAPI Routes"]
        B[Parse Job Request]
        C{Engine<br/>Type?}
        D[RayService]
        E[DatabricksRayService]
    end

    subgraph JobSubmission["Job Submission"]
        F[Submit to Local Ray Cluster]
        G[Submit Databricks Job<br/>via Jobs API]
    end

    subgraph Processing["Distributed Processing"]
        H[List S3 Objects]
        I[Partition Keys<br/>Across Workers]
        J[Fetch S3 Content]
        K[Rate-Limited<br/>Embedding Generation]
        L[Create Vector Points]
        M[Batch Upsert to<br/>Qdrant]
    end

    subgraph Workers["Worker Tasks"]
        N[Ray Remote Task]
    end

    subgraph External["External Services"]
        P[(MinIO/S3)]
        Q[(CLIP / Infinity)]
        R[(Qdrant)]
    end

    A --> B
    B --> C
    C -->|Ray| D
    C -->|Databricks| E
    D --> F
    E --> G
    F --> N
    G --> N
    N --> H
    H --> P
    P --> I
    I --> J
    J --> P
    P --> K
    K --> Q
    Q --> L
    L --> M
    M --> R

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
    style N fill:#e8f5e9
```

## Flow Description

1. **Client Request**: User submits ingestion job via `POST /ray/jobs/images` or `POST /databricks/jobs/images`
2. **API Layer**: Routes to appropriate service (RayService or DatabricksRayService)
3. **Job Submission**: Creates distributed job in local Ray cluster or Databricks (Ray on Spark)
4. **Object Discovery**: Lists S3 objects matching the prefix
5. **Parallel Processing**: Partitions work across Ray remote tasks
6. **Content Fetch**: Each worker fetches assigned S3 objects
7. **Embedding Generation**: Rate-limited calls to CLIP/Infinity service for vector embeddings
8. **Vector Point Creation**: Constructs points with embeddings + metadata
9. **Database Upsert**: Batch upserts to Qdrant
10. **Async Response**: Returns job ID immediately (202 Accepted)
