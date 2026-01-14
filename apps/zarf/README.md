# Zarf - Infrastructure Configuration

This directory contains infrastructure configuration for the iNatInq ML pipeline application.

## Directory Structure

```
zarf/
├── compose/           # Docker Compose configurations
│   └── dev/          # Development environment
│       ├── docker-compose.yaml
│       └── pipeline.env
├── docker/           # Dockerfiles
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
| Pipeline API | http://localhost:8000 | FastAPI application |
| Pipeline Docs | http://localhost:8000/docs | OpenAPI documentation |
| MinIO Console | http://localhost:9001 | Object storage UI |
| MinIO API | http://localhost:9000 | S3-compatible API |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector DB UI |
| Weaviate | http://localhost:8080 | Vector DB API |
| Ollama | http://localhost:11434 | Embedding service |
| Ray Dashboard | http://localhost:8265 | Ray cluster UI |

### Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| MinIO | minioadmin | minioadmin |

## Architecture

The Docker Compose stack emulates the Kubernetes `ml-system` namespace from `modern-web-application/zarf/k8s/dev/`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker Compose Network                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  MinIO   │    │  Qdrant  │    │ Weaviate │    │  Ollama  │          │
│  │  :9000   │    │  :6333   │    │  :8080   │    │  :11434  │          │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘          │
│       │               │               │               │                  │
│       └───────────────┼───────────────┼───────────────┘                  │
│                       │               │                                  │
│                       ▼               ▼                                  │
│                  ┌────────────────────────┐                             │
│                  │       Pipeline         │                             │
│                  │        :8000           │                             │
│                  └───────────┬────────────┘                             │
│                              │                                           │
│       ┌──────────────────────┼──────────────────────┐                   │
│       │                      │                      │                   │
│       ▼                      ▼                      ▼                   │
│  ┌──────────┐          ┌──────────┐          ┌──────────┐              │
│  │ Ray Head │◄────────►│Ray Worker│          │Ray Worker│              │
│  │  :8265   │          │          │          │          │              │
│  └──────────┘          └──────────┘          └──────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
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
docker compose -f zarf/compose/dev/docker-compose.yaml logs -f ollama
```

### Scaling Ray Workers

```bash
docker compose -f zarf/compose/dev/docker-compose.yaml up -d --scale ray-worker=3
```

## Configuration

### Environment Variables

Environment variables are defined in `compose/dev/pipeline.env`. Key configurations:

- **VECTOR_DB_PROVIDER**: `qdrant` or `weaviate`
- **EMBEDDING_PROVIDER**: `ollama` (default for local dev)
- **S3_ENDPOINT**: MinIO endpoint URL

### Switching Vector Database

Edit `pipeline.env`:

```bash
# Use Qdrant (default)
VECTOR_DB_PROVIDER=qdrant

# Or use Weaviate
VECTOR_DB_PROVIDER=weaviate
```

## Comparison with Kubernetes

| Feature | Kubernetes | Docker Compose |
|---------|------------|----------------|
| Service Discovery | DNS (`service.namespace`) | Container names |
| Storage | PersistentVolumeClaims | Docker volumes |
| Secrets | K8s Secrets/ConfigMaps | `.env` files |
| Scaling | HPA/Replicas | `--scale` flag |
| Spark | Spark Operator | Not included* |
| Health Checks | Probes | HEALTHCHECK |

*For Spark workloads, use Ray or integrate with an external Spark cluster.

## Troubleshooting

### Services Not Starting

```bash
# Check service status
docker compose -f zarf/compose/dev/docker-compose.yaml ps

# Check logs for errors
docker compose -f zarf/compose/dev/docker-compose.yaml logs ollama
```

### Ollama Model Not Loading

The `ollama-init` service pulls the model after Ollama starts. Check its logs:

```bash
docker compose -f zarf/compose/dev/docker-compose.yaml logs ollama-init
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

