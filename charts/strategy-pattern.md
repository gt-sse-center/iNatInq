# Strategy Pattern / Ingestion Architecture

Shows how the `IngestionPipeline` uses the Strategy Pattern to abstract Ray cluster lifecycle across local and Databricks environments.

```mermaid
classDiagram
    direction TB

    class ClusterStrategy {
        <<protocol>>
        +init()
        +shutdown()
        +get_runtime_env() dict~str, Any~
        +config: RayJobConfig
    }

    class LocalRayStrategy {
        +config: RayJobConfig
        +num_cpus: int | None
        +include_dashboard: bool
        +dashboard_host: str | None
        +dashboard_port: int | None
        +init()
        +shutdown()
        +get_runtime_env() dict
        +from_env(namespace) LocalRayStrategy$
        +is_active: bool
    }

    class DatabricksStrategy {
        -_config: RayJobConfig
        -_cluster: Any
        +init()
        +shutdown()
        +get_runtime_env() dict
        +from_env(namespace) DatabricksStrategy$
        +config: RayJobConfig
    }

    class IngestionPipeline {
        +cluster_strategy: ClusterStrategy
        +ray_config: RayJobConfig
        +minio_config: MinIOConfig
        +vector_config: VectorDBConfig
        +embed_config: EmbeddingConfig
        +run(s3_prefix) JobResult
        +from_env(cluster_strategy, namespace) IngestionPipeline$
        -_execute(s3_prefix, logger, start) JobResult
        -_load_checkpoint(s3, keys) tuple
        -_create_batches(keys) list
        -_submit_tasks(batches, total, limiter) list
        -_collect_results(futures, total, logger, start) list
    }

    class JobResult {
        +successful: int
        +failed: int
        +total: int
        +elapsed_seconds: float
        +rate_per_second: float
    }

    class RayJobConfig {
        +ray_address: str
        +ray_namespace: str
        +s3_batch_size: int
        +num_workers: int
        +checkpoint_enabled: bool
        +from_env(namespace) RayJobConfig$
    }

    ClusterStrategy <|.. LocalRayStrategy : implements
    ClusterStrategy <|.. DatabricksStrategy : implements
    IngestionPipeline --> ClusterStrategy : uses
    IngestionPipeline --> JobResult : returns
    IngestionPipeline --> RayJobConfig : reads config
    LocalRayStrategy --> RayJobConfig : reads config
    DatabricksStrategy --> RayJobConfig : reads config
```

## Pipeline Execution Flow

```mermaid
sequenceDiagram
    participant Caller as API / Entrypoint
    participant Pipeline as IngestionPipeline
    participant Strategy as ClusterStrategy
    participant S3 as S3ClientWrapper
    participant CP as CheckpointManager
    participant Ray as Ray Cluster
    participant Workers as Ray Workers
    participant Qdrant as Qdrant

    Caller->>Pipeline: run(s3_prefix)
    Pipeline->>Strategy: init()

    alt LocalRayStrategy
        Strategy->>Ray: ray.init(address=RAY_ADDRESS)
    else DatabricksStrategy
        Strategy->>Ray: setup_ray_cluster(max_worker_nodes)
        Strategy->>Ray: ray.init(address=cluster.address)
    end

    Pipeline->>S3: list_objects(bucket, prefix)
    S3-->>Pipeline: keys[]

    Pipeline->>CP: load(checkpoint_path)
    CP-->>Pipeline: processed_keys set
    Note over Pipeline: Filter out already-processed keys

    Pipeline->>Pipeline: _create_batches(keys)

    loop For each batch
        Pipeline->>Ray: task_fn.remote(batch, configs...)
        Ray->>Workers: schedule task
    end

    loop ray.wait() collection loop
        Pipeline->>Ray: ray.wait(futures)
        Ray-->>Pipeline: ready results
        Note over Pipeline: Log progress periodically
    end

    Pipeline->>CP: save(checkpoint_path, processed)
    Pipeline->>Strategy: shutdown()
    Strategy->>Ray: ray.shutdown()

    Pipeline-->>Caller: JobResult(successful, failed, total, elapsed, rate)
```

## Strategy Comparison

| Aspect | LocalRayStrategy | DatabricksStrategy |
|--------|------------------|--------------------|
| **Cluster Lifecycle** | Connects to existing cluster via `RAY_ADDRESS` | Creates Ray-on-Spark cluster via `setup_ray_cluster()` |
| **Runtime Env** | Config-provided `runtime_env` dict | Auto-builds with `PYTHONPATH`, passthrough env vars, `INATINQ_SRC_DIR` working dir |
| **Shutdown** | `ray.shutdown()` only (doesn't own cluster) | `ray.shutdown()` + `cluster.shutdown()` (owns the Spark cluster) |
| **Context Manager** | Yes (`__enter__`/`__exit__`) | No |
| **Use Case** | Local dev (Docker Compose), K8s deployments | Azure Databricks production jobs |

## Adding a New Strategy

1. Create `src/core/ingestion/strategies/my_strategy.py`
2. Implement the `ClusterStrategy` protocol:
   ```python
   @attrs.define
   class MyStrategy:
       config: RayJobConfig

       def init(self) -> None: ...
       def shutdown(self) -> None: ...
       def get_runtime_env(self) -> dict[str, Any]: ...
   ```
3. Wire it into the appropriate entrypoint or service
4. No changes needed to `IngestionPipeline` — it depends on the protocol, not concrete classes
