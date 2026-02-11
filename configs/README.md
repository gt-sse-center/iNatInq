# iNatInq Configuration Guide

Quick start guide for configuring iNatInq vector search system.

## Quick Start

1. **Copy the secrets template:**
   ```bash
   cp configs/secrets.example.yaml configs/secrets.yaml
   # Edit secrets.yaml with your credentials
   ```

2. **Run with an environment:**
   ```bash
   # Local development (default)
   make run

   # Other environments
   ENV=dev make run
   ENV=staging make run
   ENV=prod make run
   ```

3. **Validate your configuration:**
   ```bash
   make validate-config
   ```

## Directory Structure

```
configs/
├── config.yaml              # Base configuration (all defaults)
├── secrets.example.yaml     # Template for credentials (copy to secrets.yaml)
├── environments/            # Environment-specific overrides
│   ├── local.yaml          # Local development (Docker Compose)
│   ├── dev.yaml            # Shared development
│   ├── staging.yaml        # Pre-production
│   └── prod.yaml           # Production
├── examples/                # Cloud provider sample templates (untested)
│   ├── aws.yaml            # AWS deployment
│   ├── azure.yaml          # Azure deployment
│   └── gcp.yaml            # GCP deployment
└── schemas/
    └── config.schema.json   # JSON Schema for validation
```

## Configuration Layering

Configuration is merged in order (later overrides earlier):

```
config.yaml (base) → environments/{ENV}.yaml → secrets.yaml → environment variables
```

**Example:** Running with `ENV=prod`:
1. Load `config.yaml` (defaults)
2. Merge `environments/prod.yaml` (production overrides)
3. Merge `secrets.yaml` (credentials)
4. Apply environment variables (final overrides)

## Environment Variables

Any setting can be overridden via environment variable. Use `${VAR_NAME}` syntax
in YAML files for runtime substitution:

```yaml
storage:
  endpoint: "${STORAGE_ENDPOINT}"    # Substituted at runtime
  bucket: "my-bucket"                # Static value
```

### Common Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `STORAGE_ENDPOINT` | S3/MinIO endpoint | `https://s3.amazonaws.com` |
| `S3_ACCESS_KEY_ID` | Storage access key | - |
| `S3_SECRET_ACCESS_KEY` | Storage secret key | - |
| `QDRANT_URL` | Qdrant endpoint | `http://qdrant:6333` |
| `QDRANT_API_KEY` | Qdrant API key | - |
| `OLLAMA_URL` | Ollama endpoint | `http://ollama:11434` |
| `OPENAI_API_KEY` | OpenAI API key | - |

## Key Configuration Sections

### Storage (S3/MinIO)

```yaml
storage:
  endpoint: "http://localhost:9000"
  bucket: "inatinq-data"
  region: "us-east-1"
  use_ssl: false              # true for production
  path_style: true            # true for MinIO, false for AWS S3
  timeout_seconds: 30
  retry:
    max_attempts: 3
```

### Vector Database

Supports Qdrant and Weaviate. Set `search_provider` to choose:

```yaml
vector_databases:
  search_provider: "qdrant"   # or "weaviate"
  collection: "documents"

  qdrant:
    url: "http://localhost:6333"
    index:
      distance_metric: "cosine"   # cosine, euclidean, dot
      hnsw_m: 16                   # HNSW connections per node (4-128)
      hnsw_ef_construct: 100      # Build-time search width (10-500)
      indexing_threshold: 10000   # Points before indexing starts
    sharding:
      enabled: false
      shard_count: 1
      replication_factor: 1
```

### Embeddings

Multiple providers supported:

```yaml
embeddings:
  provider: "ollama"          # ollama, openai, huggingface, sagemaker
  vector_size: null           # Auto-detect from model

  ollama:
    url: "http://localhost:11434"
    model: "nomic-embed-text"
    max_batch_size: 32
```

### Processing & Ray

```yaml
processing:
  workers: 4
  batch_size: 50
  checkpoint:
    enabled: true
    directory: "./checkpoints"

ray:
  address: null               # null = start local cluster
  cluster:
    num_workers: 2
    worker_cpus: 1.0
    worker_memory_bytes: 500000000  # 500MB
```

## Cloud Deployments

See `configs/examples/` for cloud-specific configurations. **These are sample
templates, not tested deployments.** You will need to adapt them to your
specific cloud setup and credentials.

- **AWS:** `examples/aws.yaml` - S3, SageMaker, EKS
- **Azure:** `examples/azure.yaml` - Blob Storage, AKS
- **GCP:** `examples/gcp.yaml` - GCS, GKE

Usage:
```bash
# Copy cloud example as your environment overlay
cp configs/examples/aws.yaml configs/environments/prod.yaml
# Edit with your specific values
```

## Validation

Validate configuration against JSON Schema:

```bash
# Validate all configs
make validate-config

# Manual validation with Python
python -c "
import json
import yaml
from jsonschema import validate

with open('configs/schemas/config.schema.json') as f:
    schema = json.load(f)
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
validate(config, schema)
print('Valid!')
"
```

## Secrets Management

**Development:** Use `secrets.yaml` (gitignored)

**Production:** Use environment variables or a secrets manager:
- AWS Secrets Manager
- Azure Key Vault
- HashiCorp Vault
- Kubernetes Secrets

Never commit `secrets.yaml` with real values.

## Integration Status

**Current State:** YAML configs are **wired into the application** via `src/config_loader.py`.

### How It Works (YAML-as-defaults)

The YAML config system provides structured, documented defaults that are applied
as environment variable defaults at process startup. Environment variables always
take precedence, so existing deployments are fully backward compatible.

**Runtime layering order** (later overrides earlier):

```
hardcoded defaults < config.yaml < environments/{ENV}.yaml < secrets.yaml < env vars
```

**Implementation:**

1. `get_settings()` in `src/config.py` calls `initialize_config()` before building `Settings`
2. `initialize_config()` in `src/config_loader.py` loads and merges YAML files
3. Merged values are applied as env var defaults (only if the env var is not already set)
4. All existing `from_env()` methods continue reading `os.getenv()` unchanged

```python
# src/config.py
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from config_loader import initialize_config
    initialize_config()          # Apply YAML defaults to env vars
    return Settings.from_env()   # Reads os.getenv() as before
```

**Key behaviors:**

- Env vars always win (backward compatible for all current deployments)
- `${VAR_NAME}` syntax in YAML values is resolved from `os.environ`
- Missing YAML files are silently skipped (base config.yaml is optional too)
- `initialize_config()` is idempotent (safe to call multiple times)
- Set `ENV` or `PIPELINE_ENV` to select the environment overlay (e.g. `ENV=dev`)

---

## Troubleshooting

### Config not loading
- Check file exists: `ls configs/environments/${ENV}.yaml`
- Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('file.yaml'))"`

### Environment variable not substituted
- Ensure variable is exported: `export VAR_NAME=value`
- Check syntax: `${VAR_NAME}` not `$VAR_NAME`

### Schema validation fails
- Run `python configs/validate.py` for detailed errors
- Check required fields in `config.schema.json`
