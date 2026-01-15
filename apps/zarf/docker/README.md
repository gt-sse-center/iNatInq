# Docker Images

This directory contains Dockerfiles for building container images.

## Directory Structure

```
docker/
└── dev/                      # Development environment
    └── Dockerfile.pipeline   # Pipeline service image
```

## Building Images

### Pipeline Service

```bash
# From apps/ directory
docker build -f zarf/docker/dev/Dockerfile.pipeline -t apps-pipeline:dev .

# Or via Docker Compose
docker compose -f zarf/compose/dev/docker-compose.yaml build pipeline
```

## Image Details

### Dockerfile.pipeline

Multi-stage build for the FastAPI pipeline service:

- **Base**: `python:3.11-slim`
- **Dependencies**: Installed via `uv` for speed
- **Runtime**: Non-root user, health check included

**Build context**: `apps/` directory

**Exposed ports**: `8000` (HTTP API)

## Adding New Dockerfiles

Follow the naming convention:

```
docker/<environment>/Dockerfile.<service>
```

Examples:
- `docker/dev/Dockerfile.pipeline`
- `docker/dev/Dockerfile.worker`
- `docker/prod/Dockerfile.pipeline`

