# Ingestion Engine Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as FastAPI Router
    participant SVC as Ray/Spark Service
    participant Cluster as Ray Cluster /<br/>K8s Spark Operator
    participant Worker as Worker Task<br/>(Ray Remote / Spark Partition)
    participant S3 as MinIO/S3
    participant Ollama as Ollama Service
    participant Qdrant as Qdrant
    participant Weaviate as Weaviate

    C->>+API: POST /ray/jobs or /spark/jobs<br/>{"s3_prefix": "inputs/", "collection": "docs"}

    Note over API: Load configs from environment
    
    API->>+SVC: submit_s3_to_qdrant(params)
    
    alt Ray Engine
        SVC->>+Cluster: ray.init() + submit job script
    else Spark Engine
        SVC->>+Cluster: Create SparkApplication CRD
    end
    
    Cluster-->>-SVC: job_id / job_name
    SVC-->>-API: job identifier
    API-->>-C: 202 Accepted<br/>{"job_id": "raysubmit_xxx", "status": "submitted"}

    Note over Cluster: Async Job Execution Begins

    rect rgb(240, 248, 255)
        Note over Cluster,Weaviate: Distributed Processing (runs asynchronously)
        
        Cluster->>+S3: List objects with prefix "inputs/"
        S3-->>-Cluster: [key1, key2, key3, ...]
        
        Cluster->>Cluster: Partition keys across workers
        
        par Worker 1
            Cluster->>+Worker: process_batch([key1, key2])
        and Worker 2
            Cluster->>+Worker: process_batch([key3, key4])
        and Worker N
            Cluster->>+Worker: process_batch([keyN-1, keyN])
        end
        
        loop For each key in batch
            Worker->>+S3: GET object content
            S3-->>-Worker: text content
            
            Worker->>Worker: Rate limit (wait if needed)
            
            Worker->>+Ollama: POST /api/embeddings<br/>{"prompt": "content..."}
            Ollama-->>-Worker: {"embedding": [...]}
            
            Worker->>Worker: Create PointStruct with payload
        end
        
        Note over Worker: Batch upsert (200 points max)
        
        Worker->>+Qdrant: PUT /collections/{name}/points<br/>{points: [...]}
        Qdrant-->>-Worker: {"status": "ok"}
        
        Worker->>+Weaviate: POST /v1/batch/objects
        Weaviate-->>-Worker: {"status": "ok"}
        
        Worker-->>-Cluster: ProcessingResult[]
    end

    C->>+API: GET /ray/jobs/{job_id}
    API->>+SVC: get_job_status(job_id)
    SVC->>+Cluster: Query job status
    Cluster-->>-SVC: {"status": "SUCCEEDED"}
    SVC-->>-API: status info
    API-->>-C: {"job_id": "...", "status": "SUCCEEDED"}
```

## Sequence Description

| Phase | Steps | Description |
| ----- | ----- | ----------- |
| **Submission** | 1-6 | Client submits job, service creates Ray/Spark job, returns immediately |
| **Discovery** | 7-8 | Job lists S3 objects matching prefix |
| **Distribution** | 9 | Keys partitioned across workers (Ray tasks or Spark executors) |
| **Processing** | 10-17 | Each worker: fetch content → rate-limit → embed → create points |
| **Upsert** | 18-21 | Batch upsert vectors to both Qdrant and Weaviate |
| **Status Check** | 22-27 | Client polls for job completion |

## Key Implementation Details

### Rate Limiting

- Configurable RPS limit to prevent Ollama overload
- Async semaphore for concurrency control
- Dynamic batch sizing (shrink on failure, grow on success)

### Fault Tolerance

- Checkpoint support for resumable processing (Spark)
- Per-key error tracking (success/failure with error message)
- Graceful degradation if one vector DB fails

### Parallelism

- **Ray**: Remote tasks with configurable batch sizes
- **Spark**: RDD partitions processed in parallel executors
