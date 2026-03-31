# Postman Collection

API collection and environment for the iNatInq ML Pipeline service.

## Files

| File | Description |
|------|-------------|
| `iNatInq-Pipeline-API.postman_collection.json` | Full API collection with all endpoints |
| `iNatInq-Local.postman_environment.json` | Local development environment variables |

## Import into Postman

1. Open Postman
2. Click **Import** (top-left)
3. Drag both JSON files or click "Upload Files"
4. Select the **iNatInq Local** environment (top-right dropdown)

## Endpoints

### Health
- `GET /healthz` - Liveness probe

### Image Search
- `GET /search/images` - Text-to-image semantic search (CLIP/Qdrant)

### Ray Jobs
- `POST /ray/jobs/images` - Submit image ingestion job
- `POST /ray/jobs/process-dlq` - Submit DLQ processing job
- `GET /ray/jobs/{job_id}` - Get job status
- `GET /ray/jobs/{job_id}/logs` - Get job logs
- `DELETE /ray/jobs/{job_id}` - Stop job

### Databricks Jobs
- `POST /databricks/jobs/images` - Submit image ingestion job
- `POST /databricks/jobs/cdc-producer` - Submit CDC producer job
- `POST /databricks/jobs/cdc-consumer` - Submit CDC consumer job
- `POST /databricks/jobs/process-dlq` - Submit DLQ processing job
- `GET /databricks/jobs/{run_id}` - Get run status
- `GET /databricks/jobs/{run_id}/logs` - Get run logs/output
- `DELETE /databricks/jobs/{run_id}` - Stop run

### Cache
- `DELETE /cache` - Invalidate semantic cache

### Metrics
- `POST /ingestion/metrics` - Record ingestion metrics

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `base_url` | `http://localhost:8000` | API base URL |
| `collection` | `documents` | Vector DB collection |
| `s3_prefix` | `inputs/` | S3 prefix for jobs |
| `job_id` | (auto-set) | Ray job ID |
| `run_id` | (auto-set) | Databricks run ID |

## Usage Tips

1. **Start services first**: `uv run inq up` from the repository root
2. **Run Health Check** to verify connectivity
3. **Submit a job** - the response auto-saves `job_id` or `job_name`
4. **Check status** - uses the saved job identifier
