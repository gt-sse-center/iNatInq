# Databricks (Azure)

This directory contains Databricks cluster/job specs and helper scripts for the
iNatInq pipeline.

## Files

- `dev/inatinq-azure-databricks-cluster.json.example`: Cluster spec template.
- `dev/inatinq-ml-pipeline-job.yml.example`: Databricks Job spec template.
- `azure-databricks-build.py`: Create or update the cluster from the spec.
- `azure-databricks-up.py`: Start the cluster.
- `azure-databricks-down.py`: Terminate the cluster.
- `azure-databricks-cdc-notebooks.py`: Load env, start cluster, validate/upload CDC notebooks, optionally run them.
- `azure-databricks-configure-minio-s3a.py`: Create secret scope entries and apply S3A Spark conf on a cluster.

## Local environment

Databricks secrets live in a gitignored file:

```bash
cp zarf/databricks/dev/env.local.example zarf/databricks/dev/.env.local
# Edit zarf/databricks/dev/.env.local with your Databricks credentials
```

Databricks specs should also live in `zarf/databricks/dev/` (gitignored):

```bash
cp zarf/databricks/dev/inatinq-azure-databricks-cluster.json.example \
  zarf/databricks/dev/inatinq-azure-databricks-cluster.json
cp zarf/databricks/dev/inatinq-ml-pipeline-job.yml.example \
  zarf/databricks/dev/inatinq-ml-pipeline-job.yml
# Edit the dev specs with your cluster/job IDs
```

### Required variables

Databricks job/CLI settings:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_JOB_ID`
- `DATABRICKS_INAT_JOB_ID` (required for dedicated iNaturalist image job submission)
- `DATABRICKS_S3_AUTOLOADER_JOB_ID` (required for dedicated Auto Loader job submission)
- `DATABRICKS_FROM_BRONZE_JOB_ID` (required for dedicated Bronze incremental Ray job submission)
- `DATABRICKS_TASK_TYPE` (default: `python`)
- `DATABRICKS_CLUSTER_ID` (optional override for cluster start/stop)
- `INATINQ_SRC_DIR` (optional; override the workspace src path)

### Ray tuning (Databricks runtime)

- `RAY_NUM_WORKERS`
- `RAY_WORKER_CPUS`

### Databricks image entrypoints

- S3 image job: `run_ingest_image.py` -> `process_s3_images.py`
- iNaturalist image job: `run_ingest_inat_image.py` -> `process_inat_images.py`
- Auto Loader Bronze job: `run_ingest_s3_autoloader.py` -> `process_s3_autoloader.py`
- Bronze CDC Ray job: `run_ingest_image_from_bronze.py` -> `process_s3_images_from_bronze.py`

### CDC (Auto Loader + Bronze) runtime params

Auto Loader job required params:

- `S3_BUCKET`
- `AUTOLOADER_BRONZE_TABLE`
- `AUTOLOADER_SCHEMA_LOCATION`
- `AUTOLOADER_CHECKPOINT_LOCATION`

Auto Loader optional params:

- `AUTOLOADER_FILE_FORMAT` (default: `binaryFile`)
- `AUTOLOADER_INCLUDE_EXISTING_FILES` (default: `true`)
- `AUTOLOADER_MAX_FILES_PER_TRIGGER`
- `AUTOLOADER_TRIGGER_MODE` (`availableNow`, `once`, `processingTime`)
- `AUTOLOADER_TRIGGER_INTERVAL` (for `processingTime`)

MinIO-backed Auto Loader:

- Set `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY`.
- Optional MinIO flags: `S3_USE_SSL` and `S3_PATH_STYLE`.
- Source path is derived from `S3_BUCKET` and optional `S3_PREFIX` (`s3://<bucket>/<prefix>`).
- Derived source path is normalized to `s3a://...` when explicit `S3_ENDPOINT` is set.

Bronze CDC Ray job params:

- `AUTOLOADER_BRONZE_TABLE` (required)
- `CDC_PROGRESS_TABLE` (optional; default: `<bronze_table>_progress`)
- `CDC_PROGRESS_ID` (optional; default: `s3_bronze_image_ingestion`)
- `CDC_WINDOW_SIZE` (optional; default: `5000`)
- `CDC_KEY_COL` (optional; default: `s3_key`)
- `CDC_WATERMARK_COL` (optional; default: `discovered_at`)

### CDC workflow scheduling with overlap protection

The job spec example includes `inatinq_ml_pipeline_cdc_workflow_job`, which:

1. Runs `run_ingest_s3_autoloader.py`
2. Then runs `run_ingest_image_from_bronze.py` via `depends_on`
3. Schedules every 15 minutes with `quartz_cron_expression`
4. Prevents overlap using `max_concurrent_runs: 1`

### CDC test notebooks

Databricks source notebooks for repeatable CDC validation live under:

- `zarf/databricks/dev/notebooks/cdc_test_common.py`
- `zarf/databricks/dev/notebooks/cdc_producer_test.py`
- `zarf/databricks/dev/notebooks/cdc_consumer_test.py`

Recommended order:

1. Run `cdc_producer_test.py` to validate Bronze table contract and producer-like append behavior.
2. Run `cdc_consumer_test.py` to validate CDC window/cursor semantics using direct Bronze table manipulation.

Environment defaults:

- Both notebooks auto-load `zarf/databricks/dev/.env.local` when present.
- You can override with widget `env_file` to point to a different env file.
- Databricks credential vars (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, etc.) and `INATINQ_SRC_DIR` are applied from the env file when available.
- Notebooks create ephemeral test-only Bronze/progress tables for each run and drop them at the end.
- Notebooks reject non-test table names to avoid accidental writes into production/shared tables.

### iNaturalist image job (`run_ingest_inat_image.py` -> `process_inat_images.py`)

Required:

- `INAT_MAX_ROWS` (must be a positive integer; job fails fast if missing)

Optional (with defaults):

- `INAT_METADATA_URL` (default: `s3://inaturalist-open-data/photos.csv.gz`)
- `INAT_IMAGE_SIZE` (default: `medium`)
- `INAT_PHOTO_BASE_URL` (default: `https://inaturalist-open-data.s3.amazonaws.com/photos`)
- `INAT_TIMEOUT_S` (default: `120`)
- `INAT_CB_FAILURE_THRESHOLD` (default: `5`)
- `INAT_CB_RECOVERY_TIMEOUT_S` (default: `30`)

### Vector DB settings

Provider selection:

- `VECTOR_DB_PROVIDER` (`qdrant` or `weaviate`)
- `VECTOR_DB_COLLECTION` (optional)

Qdrant Cloud:

- `QDRANT_URL`
- `QDRANT_API_KEY`

Weaviate Cloud:

- `WEAVIATE_URL`
- `WEAVIATE_API_KEY`
- `WEAVIATE_GRPC_HOST`

### Ollama (optional external)

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`

### S3/MinIO (object storage)

- `S3_ENDPOINT`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_BUCKET`
- `S3_PREFIX`
- `S3_USE_SSL`
- `S3_PATH_STYLE`
- `S3_TIMEOUT`
- `S3_MAX_RETRIES`
- `S3_RETRY_MIN_WAIT`
- `S3_RETRY_MAX_WAIT`
- `S3_CIRCUIT_BREAKER_THRESHOLD`
- `S3_CIRCUIT_BREAKER_TIMEOUT`

### Dead Letter Queue (optional)

- `DLQ_BACKEND`
- `DLQ_REDIS_HOST`
- `DLQ_REDIS_PORT`
- `DLQ_REDIS_DATABASE_NUMBER`

## Make targets (recommended)

These targets use `zarf/databricks/dev/.env.local` and
`zarf/databricks/dev/inatinq-azure-databricks-cluster.json` by default:

```bash
make azure-databricks-build
make azure-databricks-up
make azure-databricks-down
make azure-databricks-cdc-notebooks
make azure-databricks-configure-minio-s3a
```

### CDC notebook bootstrap script

Use this script when you want one command to:

1. Read `zarf/databricks/dev/.env.local`
2. Start/wait `DATABRICKS_CLUSTER_ID`
3. Verify CDC notebooks exist in workspace (or upload them)
4. Optionally submit producer/consumer notebooks as Databricks runs

Quick start:

```bash
make azure-databricks-cdc-notebooks
```

This target now runs the reliable validation path by default:

1. Starts/waits for the configured cluster
2. Uploads current local CDC notebook sources to workspace (`--upload-notebooks`)
3. Executes producer and consumer test notebooks (`--run-notebooks`)

Step-by-step local validation:

1. Create local env file:

```bash
cp zarf/databricks/dev/env.local.example zarf/databricks/dev/.env.local
```

2. Set required values in `zarf/databricks/dev/.env.local`:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_CLUSTER_ID`
- `INATINQ_SRC_DIR` (workspace path to repo `src/`, for example `/Workspace/Users/<user>/iNatInq/src`)

3. Run the default reliable CDC notebook validation:

```bash
make azure-databricks-cdc-notebooks
```

4. Optional: run producer or consumer only when debugging:

```bash
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local \
  --upload-notebooks \
  --run-notebooks \
  --only producer
```

```bash
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local \
  --upload-notebooks \
  --run-notebooks \
  --only consumer
```

5. Success criteria:

- Producer run finishes with `SUCCESS`.
- Consumer run finishes with `SUCCESS`.
- Test tables are dropped automatically at notebook end (including failure paths via cleanup safeguards).

Direct usage examples:

```bash
# Start cluster + validate notebook paths from INATINQ_SRC_DIR parent.
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local

# Same flow, and write Databricks CLI profile (~/.databrickscfg).
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local \
  --configure-cli

# Upload notebook sources to workspace (overwrites existing workspace files).
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local \
  --upload-notebooks

# Submit notebook runs on the running cluster (producer then consumer).
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local \
  --run-notebooks

# Full reliable validation path (equivalent to make target behavior).
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local \
  --upload-notebooks \
  --run-notebooks

# Optional: provide deterministic suffix used by both notebook runs.
uv run zarf/databricks/azure-databricks-cdc-notebooks.py \
  --env-file zarf/databricks/dev/.env.local \
  --run-notebooks \
  --test-run-suffix ci_001
```

Notes:

- Interactive notebook “attach” is a UI action; this script starts the cluster and prints notebook URLs.
- For automated validation, use `--run-notebooks` instead of manual attach+Run All.
- Upload path normalization is applied to avoid creating new workspace notebook names like `*.py.py`.

`azure-databricks-configure-minio-s3a.py` reads `ENV_FILE` (default:
`zarf/databricks/dev/.env.local`) and uses:

- `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_CLUSTER_ID`
- `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`
- Optional:
  - `DATABRICKS_S3_SECRET_SCOPE` (default: `inatinq-minio`)
  - `DATABRICKS_S3_ACCESS_KEY_NAME` (default: `s3-access-key`)
  - `DATABRICKS_S3_SECRET_KEY_NAME` (default: `s3-secret-key`)
  - `DATABRICKS_RESTART_CLUSTER_AFTER_S3A_CONFIG` (`true`/`false`, default: `false`)

## Passing iNat params to the Databricks run

Important: `zarf/databricks/dev/.env.local` is used by local helper scripts.
Databricks task runtime values must still be passed as `python_params` (KEY=VALUE)
at run submission time (Jobs API / service submission / task parameters).

Example `python_params` for iNat image ingestion:

```text
INAT_MAX_ROWS=50000
INAT_METADATA_URL=s3://inaturalist-open-data/photos.csv.gz
INAT_IMAGE_SIZE=medium
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_COLLECTION=documents
INATINQ_SRC_DIR=/Workspace/Users/<user>/iNatInq/apps/src
```

## Direct script usage

```bash
ENV_FILE=zarf/databricks/dev/.env.local \
CLUSTER_SPEC_FILE=zarf/databricks/dev/inatinq-azure-databricks-cluster.json \
zarf/databricks/azure-databricks-build.py
```
