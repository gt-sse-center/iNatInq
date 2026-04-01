# Zarf - Infrastructure Configuration

This directory contains infrastructure configuration for the iNatInq ML pipeline application.

## Directory Structure

```
zarf/
├── compose/           # Docker Compose configurations
│   └── dev/          # Development environment
│       ├── docker-compose.yaml
│       ├── pipeline.env          # Default config (committed)
│       ├── env.local.example     # Template for cloud credentials
│       └── .env.local            # Local overrides (gitignored)
├── databricks/        # Databricks cluster/job specs and scripts
│   ├── dev/           # Databricks env overrides (gitignored)
│   │   ├── env.local.example
│   │   └── .env.local
│   │   ├── inatinq-azure-databricks-cluster.json.example
│   │   └── inatinq-ml-pipeline-job.yml.example
│   ├── azure-databricks-build.py # Create/update cluster from spec
│   ├── azure-databricks-up.py    # Start cluster
│   └── azure-databricks-down.py  # Terminate cluster
├── docker/           # Dockerfiles
│   ├── base/         # Base images (heavy dependencies)
│   │   └── Dockerfile.pipeline-base
│   └── dev/          # Development Dockerfiles
│       └── Dockerfile.pipeline
├── scripts/          # Infrastructure scripts
│   └── init-minio.sh
└── README.md
```

## Quick Start

### Docker Compose (Local Development)

```bash
# Start all services
cd apps
docker compose -f zarf/compose/dev/docker-compose.yaml up -d

# View logs
docker compose -f zarf/compose/dev/docker-compose.yaml logs -f

# Stop all services
docker compose -f zarf/compose/dev/docker-compose.yaml down

# Stop and remove volumes (clean slate)
docker compose -f zarf/compose/dev/docker-compose.yaml down -v
```

### Service Endpoints

Once running, services are available at:

| Service | URL | Description |
|---------|-----|-------------|
| Pipeline API | <http://localhost:8000> | FastAPI application |
| Pipeline Docs | <http://localhost:8000/docs> | OpenAPI documentation |
| CLIP Server | <http://localhost:8001> | CLIP embedding service |
| Infinity Server | <http://localhost:7997> | SigLIP2 embedding service |
| MinIO Console | <http://localhost:9001> | Object storage UI |
| MinIO API | <http://localhost:9000> | S3-compatible API |
| Qdrant Dashboard | <http://localhost:6333/dashboard> | Vector DB UI |
| Redis | <http://localhost:8334> | DLQ and semantic cache |
| Ray Dashboard | <http://localhost:8265> | Ray cluster UI |

### Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| MinIO | minioadmin | minioadmin |

## Architecture

The Docker Compose stack provides the local development environment:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose Network                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  MinIO   │  │  Qdrant  │  │   CLIP   │  │ Infinity │  │  Redis   │      │
│  │  :9000   │  │  :6333   │  │  :8001   │  │  :7997   │  │  :8334   │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │              │             │             │             │             │
│       └──────────────┴─────────────┴─────────────┴─────────────┘             │
│                                    │                                         │
│                                    ▼                                         │
│                  ┌────────────────────────┐                                  │
│                  │       Pipeline         │                                  │
│                  │        :8000           │                                  │
│                  └───────────┬────────────┘                                  │
│                              │                                               │
│       ┌──────────────────────┼──────────────────────┐                        │
│       │                      │                      │                        │
│       ▼                      ▼                      ▼                        │
│  ┌──────────┐          ┌──────────┐          ┌──────────┐                    │
│  │ Ray Head │◄────────►│Ray Worker│          │Ray Worker│                    │
│  │  :8265   │          │          │          │          │                    │
│  └──────────┘          └──────────┘          └──────────┘                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Development Workflow

### Rebuilding Pipeline Image

```bash
# Rebuild after code changes
docker compose -f zarf/compose/dev/docker-compose.yaml build pipeline

# Rebuild and restart
docker compose -f zarf/compose/dev/docker-compose.yaml up -d --build pipeline
```

### Hot Reload (Development)

Uncomment the volume mount in `docker-compose.yaml` for hot reload:

```yaml
volumes:
  - ../../../src:/app/src:ro
```

### Viewing Logs

```bash
# All services
docker compose -f zarf/compose/dev/docker-compose.yaml logs -f

# Specific service
docker compose -f zarf/compose/dev/docker-compose.yaml logs -f pipeline
```

### Scaling Ray Workers

```bash
docker compose -f zarf/compose/dev/docker-compose.yaml up -d --scale ray-worker=3
```

## Configuration

### Environment Variables

Environment variables are defined in `compose/dev/pipeline.env`. Key configurations:

- **VECTOR_DB_PROVIDER**: `qdrant`
- **EMBEDDING_PROVIDER**: `clip` (default for local dev)
- **S3_ENDPOINT**: MinIO endpoint URL

### Using Cloud Vector Database

To use a cloud-hosted Qdrant instance instead of the local container:

**Qdrant Cloud** ([cloud.qdrant.io](https://cloud.qdrant.io/)):

```bash
export QDRANT_URL=https://your-cluster.region.cloud.qdrant.io
export QDRANT_API_KEY=your-api-key
docker compose -f zarf/compose/dev/docker-compose.yaml up -d
```

**Using a config file** (gitignored):

```bash
cp zarf/compose/dev/env.local.example zarf/compose/dev/.env.local
# Edit .env.local with your cloud credentials
```

### Azure Databricks

Databricks secrets live in `zarf/databricks/dev/.env.local` (gitignored). Copy the template:

```bash
cp zarf/databricks/dev/env.local.example zarf/databricks/dev/.env.local
# Edit .env.local with your Databricks credentials
```

Databricks specs should live in `zarf/databricks/dev/` (gitignored). Copy the templates:

```bash
cp zarf/databricks/dev/inatinq-azure-databricks-cluster.json.example \
  zarf/databricks/dev/inatinq-azure-databricks-cluster.json
cp zarf/databricks/dev/inatinq-ml-pipeline-job.yml.example \
  zarf/databricks/dev/inatinq-ml-pipeline-job.yml
# Edit the dev specs with your cluster/job IDs
```

Manage the Databricks cluster (requires Databricks CLI):

```bash
uv run inq databricks build
uv run inq databricks up
uv run inq databricks down
```

The local containers will still run but won't be used when cloud credentials are set.

## Troubleshooting

### Services Not Starting

```bash
# Check service status
docker compose -f zarf/compose/dev/docker-compose.yaml ps

# Check logs for errors
docker compose -f zarf/compose/dev/docker-compose.yaml logs pipeline
```

### Ray Connection Issues

Ensure Ray head is healthy before workers connect:

```bash
docker compose -f zarf/compose/dev/docker-compose.yaml logs ray-head
```

### MinIO Bucket Issues

The `minio-init` service creates the bucket. Verify:

```bash
docker compose -f zarf/compose/dev/docker-compose.yaml logs minio-init
```
