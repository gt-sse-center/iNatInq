# Scripts

Operational scripts for health checks and smoke testing.

## Scripts

| Script | Description | Used By |
|--------|-------------|---------|
| `providers_health.sh` | HTTP health checks for configured providers (Qdrant) | `inq smoke health`, `inq docker health` |
| `smoke_providers.sh` | Wrapper that loads `.env.local` and runs the Python smoke test | `inq smoke providers` |
| `smoke_providers.py` | End-to-end smoke test: embed text, upsert to vector DB, search | `smoke_providers.sh` |

## Usage

These scripts are typically invoked via the CLI rather than directly:

```bash
# Health check providers
uv run inq smoke health

# Full smoke test (embed → upsert → search)
uv run inq smoke providers

# Both
uv run inq smoke all
```

## Configuration

Scripts load environment variables from `zarf/compose/dev/.env.local`. Override with `ENV_FILE`:

```bash
ENV_FILE=/path/to/custom.env bash zarf/scripts/providers_health.sh
```
