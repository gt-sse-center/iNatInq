# Docker Compose Configurations

This directory contains Docker Compose configurations for different environments.

## Directory Structure

```
compose/
└── dev/                      # Development environment
    ├── docker-compose.yaml   # Main compose file
    └── pipeline.env          # Environment variables
```

## Environments

### Development (`dev/`)

Full ML stack for local development:

- **MinIO**: S3-compatible object storage (+ init container for bucket setup)
- **Qdrant**: Vector database for image embeddings
- **CLIP**: Local CLIP embedding server for text-to-image search
- **Infinity**: SigLIP2 image embedding service via `michaelf34/infinity`
- **Redis**: Dead letter queue backend and semantic cache
- **Ray**: Distributed computing (head + worker nodes)
- **Pipeline**: FastAPI application (orchestrator)

```bash
# Start
docker compose -f zarf/compose/dev/docker-compose.yaml up -d

# Stop
docker compose -f zarf/compose/dev/docker-compose.yaml down
```

## Adding New Environments

Create a new directory with:

1. `docker-compose.yaml` - Service definitions
2. `<service>.env` - Environment variables

Example structure for staging:

```
compose/
├── dev/
│   ├── docker-compose.yaml
│   └── pipeline.env
└── stage/
    ├── docker-compose.yaml
    └── pipeline.env
```

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Pipeline | 8000 | FastAPI application |
| CLIP | 8001 | CLIP embedding server |
| Infinity | 7997 | SigLIP2 embedding server |
| MinIO API | 9000 | S3-compatible API |
| MinIO Console | 9001 | Web UI |
| Qdrant HTTP | 6333 | REST API |
| Qdrant gRPC | 6334 | gRPC API |
| Redis | 8334 | DLQ and semantic cache |
| Ray Dashboard | 8265 | Web UI |
| Ray Client | 10001 | Client connection |

## Web UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| Pipeline Docs | http://localhost:8000/docs | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Qdrant Dashboard | http://localhost:6333/dashboard | - |
| Ray Dashboard | http://localhost:8265 | - |



