# src/

Application source code for the iNatInq pipeline.

## Configuration

iNatInq uses a **YAML-as-defaults** configuration system. YAML files provide structured, documented defaults that are bridged to environment variables at startup. Env vars always take priority.

### How It Works

```
                   ┌─────────────┐
                   │ config.yaml │  Base defaults
                   └──────┬──────┘
                          │ deep merge
              ┌───────────▼────────────┐
              │ environments/{ENV}.yaml │  Per-environment overrides
              └───────────┬────────────┘
                          │ deep merge
                  ┌───────▼────────┐
                  │  secrets.yaml  │  Credentials (gitignored)
                  └───────┬────────┘
                          │ apply as defaults
                  ┌───────▼────────┐
                  │  os.environ    │  Env vars always win
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │  config.py     │  from_env() reads os.getenv()
                  └────────────────┘
```

1. At startup, `get_settings()` in `config.py` calls `config_loader.initialize_config()`
2. `config_loader` loads and merges YAML files from `configs/`
3. Merged values are set as env var defaults (only if the var is **not already set**)
4. All existing `from_env()` methods continue reading `os.getenv()` unchanged

### Key Files

| File | Role |
|------|------|
| `config.py` | Pydantic/attrs settings classes with `from_env()` constructors |
| `config_loader.py` | YAML loading, deep merging, env-var bridging |
| `configs/config.yaml` | Base defaults for all settings |
| `configs/environments/` | Per-environment overrides (`dev.yaml`, `local.yaml`, etc.) |
| `configs/secrets.yaml` | Credentials (gitignored, see `secrets.example.yaml`) |

### Selecting an Environment

Set the `ENV` (or `PIPELINE_ENV`) env var to load an environment overlay:

```bash
# Uses configs/environments/dev.yaml on top of config.yaml
ENV=dev uv run inq up

# Base config only (no overlay)
uv run inq up
```

### Overriding Settings

Any YAML setting can be overridden by its corresponding env var:

```bash
# YAML sets S3_BUCKET=pipeline, but env var wins
S3_BUCKET=my-bucket uv run inq up
```

The full mapping from YAML paths to env var names is defined in `YAML_TO_ENV_MAP` inside `config_loader.py`.

### Validating Configuration

```bash
# Check that YAML files parse and merge correctly
uv run inq dev validate-config
```

## Directory Structure

```
src/
├── api/              # FastAPI routes and request/response models
├── clients/          # External service clients (S3, Qdrant, CLIP, Infinity)
├── core/             # Domain logic
│   ├── ingestion/    # Ray & Databricks processing pipelines
│   └── services/     # Business logic (search, job orchestration)
├── foundation/       # Cross-cutting utilities (retry, circuit breaker, logging)
├── config.py         # Pydantic/attrs settings (reads env vars via from_env())
└── config_loader.py  # YAML config loading and env-var bridging
```

See individual module READMEs for detailed documentation:

- [`api/README.md`](api/README.md)
- [`clients/README.md`](clients/README.md)
- [`core/README.md`](core/README.md)
- [`foundation/README.md`](foundation/README.md)
