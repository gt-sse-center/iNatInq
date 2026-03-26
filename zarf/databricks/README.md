# Databricks (Azure)

This directory contains Databricks cluster/job specs and helper scripts for the
iNatInq pipeline.

## Directory structure

```text
zarf/databricks/
├── dev/
│   ├── env.local.example
│   ├── .env.local                                # gitignored
│   ├── inatinq-azure-databricks-cluster.json.example
│   ├── inatinq-azure-databricks-cluster.json     # gitignored
│   ├── inatinq-ml-pipeline-job.yml.example
│   ├── inatinq-ml-pipeline-job.yml               # gitignored
│   └── notebooks/
│       ├── cdc_test_common.py
│       ├── cdc_producer_test.py
│       └── cdc_consumer_test.py
├── azure-databricks-build.py
├── azure-databricks-up.py
├── azure-databricks-down.py
├── azure-databricks-cdc-notebooks.py
└── azure-databricks-configure-minio-s3a.py
```

## Getting started

### 1) Create local env and spec files

```bash
cp zarf/databricks/dev/env.local.example zarf/databricks/dev/.env.local
cp zarf/databricks/dev/inatinq-azure-databricks-cluster.json.example \
  zarf/databricks/dev/inatinq-azure-databricks-cluster.json
cp zarf/databricks/dev/inatinq-ml-pipeline-job.yml.example \
  zarf/databricks/dev/inatinq-ml-pipeline-job.yml
```

Then edit the copied files with your Databricks host/token and your cluster/job IDs.

### 2) Set minimum env values (sufficient for `process_s3_images`)

Set these in `zarf/databricks/dev/.env.local`:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_JOB_ID` (used to submit the S3 image Databricks job)
- `INATINQ_SRC_DIR` (required by Databricks runtime; workspace path to repo `src/`, for example `/Workspace/Users/<user>/iNatInq/src`)
- `DATABRICKS_CLUSTER_ID` (needed for helper commands like `make azure-databricks-up/down` and CDC notebook bootstrap)

Set these runtime `python_params` when submitting `run_ingest_image.py`:

- `S3_BUCKET`
- `S3_PREFIX` (optional)
- `VECTOR_DB_TARGETS` (recommended to set explicitly, for example `qdrant`)
- If targeting Qdrant: `QDRANT_URL` (+ `QDRANT_API_KEY` when required)
- For MinIO/S3-compatible endpoints: `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`

### 3) Build and manage the cluster

```bash
make azure-databricks-build
make azure-databricks-up
make azure-databricks-down
```

### 4) Submit the S3 image job (core smoke test)

Use the API route documented in `src/api/README.md` (`POST /databricks/jobs/images`)
to submit the Databricks image job.

Example:

```bash
curl -X POST http://localhost:8000/databricks/jobs/images \
  -H "Content-Type: application/json" \
  -d '{
    "source": "s3",
    "collection": "documents",
    "s3_prefix": "images"
  }'
```

Request body fields:

- `source` (`s3` or `inat`)
- `collection`
- Optional: `s3_prefix`, `image_max_items`, `image_page_size`

Expected response (`202`): `run_id`, `status`, `namespace`, `source`, `s3_prefix`, `collection`, `submitted_at`.

Smoke test success criteria:

- API returns `202` with `run_id`
- Job starts on the configured Databricks cluster
- `run_ingest_image.py` resolves `INATINQ_SRC_DIR` and launches Ray
- `process_s3_images.py` processes keys from your configured `S3_BUCKET`/`S3_PREFIX`

## Configuration params (reference)

### Databricks job/CLI settings

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_CLUSTER_ID` (optional override for cluster start/stop)
- `DATABRICKS_TASK_TYPE` (default: `python`)
- `INATINQ_SRC_DIR` (required for Databricks ingestion entrypoints; workspace src path)

### Databricks job IDs

- `DATABRICKS_JOB_ID` (required for S3 image job submission: `run_ingest_image.py`)
- `DATABRICKS_INAT_JOB_ID` (required for dedicated iNaturalist image job submission)
- `DATABRICKS_S3_AUTOLOADER_JOB_ID` (required for dedicated Auto Loader job submission)
- `DATABRICKS_FROM_BRONZE_JOB_ID` (required for dedicated Bronze incremental Ray job submission)

### Databricks image entrypoints

- S3 image job: `run_ingest_image.py` -> `process_s3_images.py`
- iNaturalist image job: `run_ingest_inat_image.py` -> `process_inat_images.py`
- Auto Loader Bronze job: `run_ingest_s3_autoloader.py` -> `process_s3_autoloader.py`
- Bronze CDC Ray job: `run_ingest_image_from_bronze.py` -> `process_s3_images_from_bronze.py`

### CDC (Auto Loader + Bronze) runtime params

Auto Loader required:

- `S3_BUCKET`
- `AUTOLOADER_BRONZE_TABLE`
- `AUTOLOADER_SCHEMA_LOCATION`
- `AUTOLOADER_CHECKPOINT_LOCATION`

Auto Loader optional:

- `AUTOLOADER_FILE_FORMAT` (default: `binaryFile`)
- `AUTOLOADER_INCLUDE_EXISTING_FILES` (default: `true`)
- `AUTOLOADER_MAX_FILES_PER_TRIGGER`
- `AUTOLOADER_TRIGGER_MODE` (`availableNow`, `once`, `processingTime`)
- `AUTOLOADER_TRIGGER_INTERVAL` (for `processingTime`)

MinIO-backed Auto Loader notes:

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

- `VECTOR_DB_PROVIDER` (`qdrant`)
- `VECTOR_DB_COLLECTION` (optional)

Qdrant Cloud:

- `QDRANT_URL`
- `QDRANT_API_KEY`

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
### Ray tuning (Databricks runtime)

- `RAY_NUM_WORKERS`
- `RAY_WORKER_CPUS`

## Workflows and notebooks

### CDC workflow scheduling with overlap protection

The job spec example includes `inatinq_ml_pipeline_cdc_workflow_job`, which:

1. Runs `run_ingest_s3_autoloader.py`
2. Then runs `run_ingest_image_from_bronze.py` via `depends_on`
3. Schedules every 15 minutes with `quartz_cron_expression`
4. Prevents overlap using `max_concurrent_runs: 1`

### CDC test notebooks

Databricks source notebooks for repeatable CDC validation:

- `zarf/databricks/dev/notebooks/cdc_test_common.py`
- `zarf/databricks/dev/notebooks/cdc_producer_test.py`
- `zarf/databricks/dev/notebooks/cdc_consumer_test.py`

Recommended order:

1. Run `cdc_producer_test.py` to validate Bronze table contract and producer-like append behavior.
2. Run `cdc_consumer_test.py` to validate CDC window/cursor semantics using direct Bronze table manipulation.

Notebook defaults:

- Both notebooks auto-load `zarf/databricks/dev/.env.local` when present.
- You can override with widget `env_file` to point to a different env file.
- Databricks credential vars (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, etc.) and `INATINQ_SRC_DIR` are applied from the env file when available.
- Notebooks create ephemeral test-only Bronze/progress tables for each run and drop them at the end.
- Notebooks reject non-test table names to avoid accidental writes into production/shared tables.

Optional automated CDC validation helper:

```bash
make azure-databricks-cdc-notebooks
```

This command starts/waits for the cluster, uploads notebook sources, and runs producer+consumer validation notebooks.

### Passing runtime params to Databricks runs

`zarf/databricks/dev/.env.local` is used by local helper scripts.
Databricks task runtime values still must be passed as `python_params` (KEY=VALUE)
at run submission time (Jobs API / service submission / task parameters).

Example `python_params` for S3 image ingestion (`run_ingest_image.py` -> `process_s3_images.py`):

```text
S3_BUCKET=pipeline
S3_PREFIX=images
VECTOR_DB_TARGETS=qdrant
QDRANT_URL=https://your-qdrant.example.com
INATINQ_SRC_DIR=/Workspace/Users/<user>/iNatInq/src
```

Example `python_params` for iNat image ingestion:

```text
INAT_MAX_ROWS=50000
INAT_METADATA_URL=s3://inaturalist-open-data/photos.csv.gz
INAT_IMAGE_SIZE=medium
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_COLLECTION=documents
INATINQ_SRC_DIR=/Workspace/Users/<user>/iNatInq/src
```

## Script reference

### `azure-databricks-cdc-notebooks.py`

Optional debug runs (producer-only or consumer-only):

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

- Interactive notebook "attach" is a UI action; this script starts the cluster and prints notebook URLs.
- For automated validation, use `--run-notebooks` instead of manual attach+Run All.
- Upload path normalization is applied to avoid creating new workspace notebook names like `*.py.py`.

### `azure-databricks-configure-minio-s3a.py`

Reads `ENV_FILE` (default: `zarf/databricks/dev/.env.local`) and uses:

- `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_CLUSTER_ID`
- `S3_ENDPOINT`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`
- Optional:
  - `DATABRICKS_S3_SECRET_SCOPE` (default: `inatinq-minio`)
  - `DATABRICKS_S3_ACCESS_KEY_NAME` (default: `s3-access-key`)
  - `DATABRICKS_S3_SECRET_KEY_NAME` (default: `s3-secret-key`)
  - `DATABRICKS_RESTART_CLUSTER_AFTER_S3A_CONFIG` (`true`/`false`, default: `false`)

### `azure-databricks-build.py`

```bash
ENV_FILE=zarf/databricks/dev/.env.local \
CLUSTER_SPEC_FILE=zarf/databricks/dev/inatinq-azure-databricks-cluster.json \
zarf/databricks/azure-databricks-build.py
```
