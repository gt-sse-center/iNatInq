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

### Vector Store
- `GET /search` - Semantic search (Qdrant or Weaviate)

### Spark Jobs
- `POST /spark/jobs` - Submit job
- `GET /spark/jobs` - List all jobs
- `GET /spark/jobs/{job_name}` - Get job status
- `DELETE /spark/jobs/{job_name}` - Delete job

### Ray Jobs
- `POST /ray/jobs` - Submit job
- `GET /ray/jobs/{job_id}` - Get job status
- `GET /ray/jobs/{job_id}/logs` - Get job logs
- `DELETE /ray/jobs/{job_id}` - Stop job

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `base_url` | `http://localhost:8000` | API base URL |
| `collection` | `documents` | Vector DB collection |
| `s3_prefix` | `inputs/` | S3 prefix for jobs |
| `job_id` | (auto-set) | Ray job ID |
| `job_name` | (auto-set) | Spark job name |

## Usage Tips

1. **Start services first**: `make up` from `apps/` directory
2. **Run Health Check** to verify connectivity
3. **Submit a job** - the response auto-saves `job_id` or `job_name`
4. **Check status** - uses the saved job identifier

