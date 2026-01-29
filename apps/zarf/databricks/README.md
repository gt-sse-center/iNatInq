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
- `DATABRICKS_TASK_TYPE` (default: `python`)
- `DATABRICKS_CLUSTER_ID` (optional override for cluster start/stop)

### Ray tuning (Databricks runtime)

- `RAY_NUM_WORKERS`
- `RAY_WORKER_CPUS`

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

## Direct script usage

```bash
ENV_FILE=zarf/databricks/dev/.env.local \
CLUSTER_SPEC_FILE=zarf/databricks/dev/inatinq-azure-databricks-cluster.json \
zarf/databricks/azure-databricks-build.py
```
