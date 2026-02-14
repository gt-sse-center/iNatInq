# Databricks (Azure)

This directory contains Databricks cluster/job specs and helper scripts for the
iNatInq pipeline.

## Files

- `dev/inatinq-azure-databricks-cluster.json.example`: Cluster spec template.
- `dev/inatinq-ml-pipeline-job.yml.example`: Databricks Job spec template.
- `azure-databricks-build.py`: Create or update the cluster from the spec.
- `azure-databricks-up.py`: Start the cluster.
- `azure-databricks-down.py`: Terminate the cluster.

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
- `DATABRICKS_TASK_TYPE` (default: `python`)
- `DATABRICKS_CLUSTER_ID` (optional override for cluster start/stop)
- `INATINQ_SRC_DIR` (optional; override the workspace src path)

### Ray tuning (Databricks runtime)

- `RAY_NUM_WORKERS`
- `RAY_WORKER_CPUS`

### Databricks image entrypoints

- S3 image job: `run_ingest_image.py` -> `process_s3_images.py`
- iNaturalist image job: `run_ingest_inat_image.py` -> `process_inat_images.py`
- `IMAGE_MAX_ITEMS` (optional): applies a max-image cap to both image jobs

### iNaturalist image job (`run_ingest_inat_image.py` -> `process_inat_images.py`)

Optional:

- `IMAGE_MAX_ITEMS` (positive integer cap applied to image processing)

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
- `S3_USE_SSL`
- `S3_PATH_STYLE`
- `S3_TIMEOUT`
- `S3_MAX_RETRIES`
- `S3_RETRY_MIN_WAIT`
- `S3_RETRY_MAX_WAIT`
- `S3_CIRCUIT_BREAKER_THRESHOLD`
- `S3_CIRCUIT_BREAKER_TIMEOUT`

## Make targets (recommended)

These targets use `zarf/databricks/dev/.env.local` and
`zarf/databricks/dev/inatinq-azure-databricks-cluster.json` by default:

```bash
make azure-databricks-build
make azure-databricks-up
make azure-databricks-down
```

## Passing iNat params to the Databricks run

Important: `zarf/databricks/dev/.env.local` is used by local helper scripts.
Databricks task runtime values must still be passed as `python_params` (KEY=VALUE)
at run submission time (Jobs API / service submission / task parameters).

Example `python_params` for iNat image ingestion:

```text
IMAGE_MAX_ITEMS=50000
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
