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
| Weaviate Console | 8081 | GraphQL Playground |
| Ollama | 11434 | LLM API |
| Ray Dashboard | 8265 | Web UI |
| Ray Client | 10001 | Client connection |

## Web UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| Pipeline Docs | http://localhost:8000/docs | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Qdrant Dashboard | http://localhost:6333/dashboard | - |
| Weaviate Console | http://localhost:8081 | - |
| Ray Dashboard | http://localhost:8265 | - |

## Weaviate GraphQL Queries

Access the Weaviate Console at http://localhost:8081 and use these queries:

### Count Objects

```graphql
{
  Aggregate {
    Documents {
      meta {
        count
      }
    }
  }
}
```

### Get Objects with Properties

```graphql
{
  Get {
    Documents(limit: 10) {
      text
      s3_key
      s3_bucket
      _additional {
        id
        vector
      }
    }
  }
}
```

### Search by Vector Similarity

```graphql
{
  Get {
    Documents(
      nearVector: {
        vector: [0.1, 0.2, ...]  # Your query vector
      }
      limit: 5
    ) {
      text
      s3_key
      _additional {
        distance
      }
    }
  }
}
```

### Filter by Property

```graphql
{
  Get {
    Documents(
      where: {
        path: ["s3_key"]
        operator: Like
        valueText: "*moby*"
      }
    ) {
      text
      s3_key
    }
  }
}
```

### Combined: Filter + Limit

```graphql
{
  Get {
    Documents(
      where: {
        path: ["s3_bucket"]
        operator: Equal
        valueText: "pipeline"
      }
      limit: 5
    ) {
      text
      s3_key
      s3_bucket
    }
  }
}

