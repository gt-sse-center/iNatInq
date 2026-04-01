# Ingestion Pipeline

Distributed image ingestion from S3 to vector databases using Ray.

## Architecture

The ingestion pipeline uses the **Strategy Pattern** to abstract Ray cluster lifecycle across environments. A unified `IngestionPipeline` orchestrator delegates cluster management to environment-specific strategies (`LocalRayStrategy`, `DatabricksStrategy`), eliminating code duplication between entrypoints.

```
ingestion/
├── pipeline.py                        # Unified IngestionPipeline orchestrator
├── interfaces/                        # Pipeline abstractions
│   ├── factories.py                   # Environment detection and pipeline factory
│   ├── operations.py                  # Operation ABCs (list, batch, process)
│   ├── pipeline.py                    # Pipeline protocol definition
│   └── types.py                       # Shared type definitions
├── strategies/                        # ClusterStrategy implementations
│   ├── base.py                        # Abstract ClusterStrategy
│   ├── local_ray.py                   # LocalRayStrategy (local Ray cluster)
│   └── databricks.py                  # DatabricksStrategy (Ray on Spark)
├── ray/                               # Local Ray entrypoints
│   └── process_s3_images.py           # Image pipeline entrypoint
├── databricks/                        # Databricks entrypoints and runners
│   ├── batch_runner.py                # Batch job runner
│   ├── cdc.py                         # Change Data Capture pipeline
│   ├── process_inat_images.py         # iNaturalist image pipeline
│   ├── process_s3_autoloader.py       # S3 autoloader pipeline
│   ├── process_s3_images.py           # S3 image pipeline
│   ├── process_s3_images_from_bronze.py  # Bronze layer image pipeline
│   ├── run_ingest.py                  # Generic Databricks job runner
│   ├── run_ingest_image.py            # Image ingestion runner
│   ├── run_ingest_image_from_bronze.py   # Bronze layer ingestion runner
│   ├── run_ingest_inat_image.py       # iNaturalist ingestion runner
│   ├── run_ingest_s3_autoloader.py    # S3 autoloader runner
│   └── runtime.py                     # Databricks runtime utilities
├── shared/                            # Cross-environment utilities
│   ├── batching.py                    # Batch splitting logic
│   ├── env_keys.py                    # Environment variable key constants
│   ├── logging.py                     # Progress logging utilities
│   ├── qdrant_indexing.py             # Qdrant collection/index management
│   └── rate_limiter.py                # Embedding API rate limiting actor
└── tasks/                             # Ray remote task functions
    └── image_processing.py            # @ray.remote image processing tasks
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

## Strategy Pattern

### ClusterStrategy Protocol

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

### Implementations

| Strategy | Environment | Cluster Init |
|----------|-------------|-------------|
| `LocalRayStrategy` | Local development | `ray.init()` on local machine |
| `DatabricksStrategy` | Databricks workspace | Ray on Spark cluster |

## Environment Differences

| Aspect | Ray (local) | Databricks |
|--------|-------------|------------|
| Cluster init | `LocalRayStrategy.init()` | `DatabricksStrategy.init()` via Spark |
| Env vars | Direct from environment | `python_params` from argv |
| Working dir | Local filesystem | Workspace path via `INATINQ_SRC_DIR` |
| Shutdown | `LocalRayStrategy.shutdown()` | `DatabricksStrategy.shutdown()` |

## Usage

### Local Ray

```bash
# Via CLI
uv run inq ray submit --prefix images/ --collection documents

# Direct
python -m core.ingestion.ray.process_s3_images images/
```

### Databricks

```bash
# Cluster management
uv run inq databricks build
uv run inq databricks up

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
