# Ingestion Pipeline

Distributed image ingestion from S3 to vector databases using Ray.

## Current Architecture

```
ingestion/
├── ray/
│   ├── process_s3_to_vector_dbs.py    # Local Ray entrypoint
│   ├── process_s3_images.py           # Image pipeline entrypoint
│   ├── ray_cluster.py                 # Local Ray init/shutdown
│   └── rate_limiter.py                # CLIP rate limiting actor
├── databricks/
│   ├── process_s3_to_vector_dbs.py    # Databricks entrypoint
│   └── run_ingest.py                  # Databricks job runner
├── tasks/
│   └── image_processing.py            # @ray.remote for images
├── checkpoint.py                      # Shared checkpoint logic
└── image_utils.py                     # Shared image preprocessing
```

## Pipeline Flow

Both Ray (local) and Databricks environments follow the same flow:

1. **Init** → Load config, initialize Ray (local cluster or Databricks Spark cluster)
2. **List** → Enumerate S3 objects from prefix
3. **Checkpoint** → Filter already-processed keys
4. **Batch** → Split keys into configurable batch sizes
5. **Distribute** → Submit `process_image_batch_ray.remote()` tasks to workers
6. **Collect** → `ray.wait()` loop with progress logging
7. **Checkpoint** → Persist successful keys
8. **Shutdown** → Cleanup Ray resources

## Environment Differences

| Aspect | Ray (local) | Databricks |
|--------|-------------|------------|
| Cluster init | `init_ray_cluster()` | `setup_ray_cluster()` via Spark |
| Env vars | Direct from environment | `python_params` from argv |
| Working dir | Local filesystem | Workspace path via `INATINQ_SRC_DIR` |
| Shutdown | `shutdown_ray_cluster()` | Custom `_shutdown_ray_cluster()` |

---

## Refactoring Proposal

### Problem

`databricks/process_s3_to_vector_dbs.py` duplicates ~90% of `ray/process_s3_to_vector_dbs.py`. The only differences are cluster initialization and environment variable handling.

### Proposed Architecture: Strategy Pattern

```
ingestion/
├── pipeline.py                    # Unified orchestrator
├── strategies/
│   ├── base.py                    # Abstract ClusterStrategy
│   ├── local_ray.py               # LocalRayStrategy
│   └── databricks.py              # DatabricksStrategy
├── tasks/
│   └── image_processing.py        # @ray.remote for images
├── checkpoint.py
└── image_utils.py
```

### Core Abstractions

#### ClusterStrategy Protocol

```python
class ClusterStrategy(Protocol):
    """Abstract interface for Ray cluster lifecycle management."""

    def init(self) -> None:
        """Initialize Ray cluster connection."""
        ...

    def shutdown(self) -> None:
        """Shutdown Ray cluster and cleanup resources."""
        ...

    def get_runtime_env(self) -> dict[str, Any]:
        """Return runtime environment configuration for Ray workers."""
        ...
```

#### Unified Pipeline

```python
@attrs.define
class IngestionPipeline:
    """Unified ingestion orchestrator for all environments."""

    cluster_strategy: ClusterStrategy
    config: PipelineConfig

    def run(
        self,
        s3_prefix: str,
    ) -> JobResult:
        """Execute the image ingestion pipeline.

        Args:
            s3_prefix: S3 prefix to process (e.g., "images/")

        Returns:
            JobResult with success/failure counts and timing
        """
        self.cluster_strategy.init()
        try:
            keys = self._list_and_filter(s3_prefix)
            results = self._process_batches(keys)
            return self._finalize(results)
        finally:
            self.cluster_strategy.shutdown()
```

#### Environment Detection Factory

```python
def create_pipeline(
    env: Literal["local", "databricks", "k8s"] | None = None,
) -> IngestionPipeline:
    """Create pipeline with appropriate cluster strategy.

    Args:
        env: Target environment. If None, auto-detect from environment.

    Returns:
        Configured IngestionPipeline instance
    """
    if env is None:
        env = _detect_environment()

    match env:
        case "databricks":
            return IngestionPipeline(
                cluster_strategy=DatabricksStrategy(),
                config=PipelineConfig.from_env(),
            )
        case "local":
            return IngestionPipeline(
                cluster_strategy=LocalRayStrategy(),
                config=PipelineConfig.from_env(),
            )
        case "k8s":
            return IngestionPipeline(
                cluster_strategy=KubeRayStrategy(),
                config=PipelineConfig.from_env(),
            )
```

### Refactoring Steps

1. **Extract shared orchestration** → `IngestionPipeline` class with batch/wait/checkpoint loop
2. **Abstract cluster lifecycle** → `ClusterStrategy` protocol for init/shutdown
3. **Merge entrypoints** → Single `main()` that detects environment or accepts flag
4. **Unify env handling** → Common config loader for env vars and Databricks `python_params`
5. **Keep task functions** → `image_processing.py` remains unchanged

### Benefits

- **~200 lines removed** from duplicated orchestration code
- **Single source of truth** for pipeline logic
- **Easy to add environments** (K8s, AWS EMR, GCP Dataproc)
- **Testable** via mock strategies
- **Type-safe** with Protocol-based contracts

---

## Usage

### Local Ray

```bash
# Via Makefile
make ray-image-job-submit IMAGE_PREFIX=images/ IMAGE_COLLECTION=documents

# Direct
python -m core.ingestion.ray.process_s3_to_vector_dbs inputs/
python -m core.ingestion.ray.process_s3_images images/
```

### Databricks

```bash
# Cluster management
make azure-databricks-build
make azure-databricks-up

# Job is triggered via Databricks Jobs API
# See zarf/databricks/README.md for details
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `S3_PREFIX` | S3 prefix to process | `images/` |
| `RAY_NUM_WORKERS` | Number of Ray workers | `4` |
| `RAY_S3_BATCH_SIZE` | Keys per batch | `50` |
| `RAY_EMBED_BATCH_MAX` | Embeddings per batch | `32` |
| `RAY_BATCH_UPSERT_SIZE` | Vectors per upsert | `100` |
| `RAY_CHECKPOINT_ENABLED` | Enable checkpointing | `true` |

See `config.py` for full configuration options.
