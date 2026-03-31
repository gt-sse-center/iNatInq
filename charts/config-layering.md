# Configuration Layering Diagram

Shows the merge order of YAML configs and environment variables, and how they flow into the application.

```mermaid
flowchart LR
    subgraph Sources["Configuration Sources (later overrides earlier)"]
        direction TB
        A["config.yaml<br/>(base defaults)"]
        B["environments/{ENV}.yaml<br/>(environment overrides)"]
        C["secrets.yaml<br/>(credentials, gitignored)"]
        D["Environment Variables<br/>(always win)"]
    end

    subgraph Loader["config_loader.py"]
        E[Load & Parse YAML]
        F["deep_merge()<br/>base → env → secrets"]
        G["Resolve ${VAR_NAME}<br/>substitutions"]
        H["Apply as env var defaults<br/>(only if not already set)"]
    end

    subgraph Config["config.py"]
        I["get_settings()<br/>@lru_cache(maxsize=1)"]
        J["Settings.from_env()<br/>os.getenv() calls"]
    end

    subgraph Settings["Frozen Pydantic Settings"]
        K[EmbeddingConfig]
        L[VectorDBConfig]
        M[MinIOConfig]
        N[RayJobConfig]
        O[SemanticCacheConfig]
        P[DatabricksRayJobConfig]
    end

    A --> E
    B --> E
    C --> E
    E --> F
    F --> G
    G --> H
    D --> H

    H --> I
    I -->|"initialize_config()<br/>(idempotent)"| J
    J --> K
    J --> L
    J --> M
    J --> N
    J --> O
    J --> P

    style A fill:#e8f5e9
    style B fill:#e8f5e9
    style C fill:#fff3e0
    style D fill:#e1f5fe
    style I fill:#f3e5f5
```

## Layering Order

```
hardcoded defaults < config.yaml < environments/{ENV}.yaml < secrets.yaml < env vars
```

| Priority | Source | Purpose | Example |
|----------|--------|---------|---------|
| 1 (lowest) | Hardcoded defaults | Fallback values in `from_env()` methods | `QDRANT_URL` defaults to `localhost:6333` |
| 2 | `configs/config.yaml` | Base configuration shared across all environments | Storage bucket, batch sizes, HNSW params |
| 3 | `configs/environments/{ENV}.yaml` | Environment-specific overrides | Production endpoint URLs, SSL settings |
| 4 | `configs/secrets.yaml` | Credentials (gitignored) | API keys, passwords |
| 5 (highest) | Environment variables | Runtime overrides, always win | `QDRANT_URL=http://qdrant:6333` |

## Environment Selection

Set `ENV` or `PIPELINE_ENV` to select the environment overlay:

```bash
ENV=prod uv run inq up    # Loads configs/environments/prod.yaml
ENV=dev uv run inq up     # Loads configs/environments/dev.yaml
# No ENV set               # Loads configs/environments/local.yaml (default)
```

## Auto-Detection

`_is_in_cluster()` in `config.py` auto-detects the runtime environment:

```mermaid
flowchart TD
    A{PIPELINE_ENV set?} -->|Yes| B[Use explicit value]
    A -->|No| C{K8s service account<br/>token exists?}
    C -->|Yes| D[cluster mode]
    C -->|No| E{KUBERNETES_SERVICE_HOST<br/>set?}
    E -->|Yes| D
    E -->|No| F[local mode]

    D --> G["Service URLs:<br/>http://qdrant.{ns}:6333<br/>http://minio.{ns}:9000<br/>http://clip.{ns}:8000"]
    F --> H["Service URLs:<br/>http://localhost:6333<br/>http://localhost:9000<br/>http://localhost:8000"]
```

## Key Behaviors

- `initialize_config()` is **idempotent** — safe to call multiple times
- `get_settings()` is **cached** via `@lru_cache(maxsize=1)` — one Settings instance per process
- `${VAR_NAME}` syntax in YAML values is resolved from `os.environ` at load time
- Missing YAML files are silently skipped (graceful degradation)
- Env vars set before `initialize_config()` always take precedence over YAML values
