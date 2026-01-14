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

- **MinIO**: S3-compatible object storage
- **Qdrant**: Vector database (primary)
- **Weaviate**: Vector database (alternative)
- **Ollama**: Local embedding service
- **Ray**: Distributed computing (head + workers)
- **Pipeline**: FastAPI application

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
| MinIO API | 9000 | S3-compatible API |
| MinIO Console | 9001 | Web UI |
| Qdrant HTTP | 6333 | REST API |
| Qdrant gRPC | 6334 | gRPC API |
| Weaviate | 8080 | REST API |
| Ollama | 11434 | LLM API |
| Ray Dashboard | 8265 | Web UI |
| Ray Client | 10001 | Client connection |

