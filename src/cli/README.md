# CLI

Typer-based developer CLI providing the `inq` command. All sub-commands are registered as Typer sub-apps in `app.py`.

## Command Groups

| Command | Module | Description |
|---------|--------|-------------|
| `inq up / down / status` | `app.py` | Top-level convenience aliases for Docker |
| `inq docker` | `docker.py` | Docker Compose lifecycle (up, down, logs, health) |
| `inq dev` | `dev.py` | Development utilities (serve, validate-config) |
| `inq test` | `test.py` | Test runner shortcuts |
| `inq search` | `search.py` | Semantic search (text and image queries) |
| `inq ray` | `ray.py` | Ray job submission and monitoring |
| `inq vectordb` | `vectordb.py` | Vector database operations (count, delete, list) |
| `inq synthetic` | `synthetic.py` | Synthetic data generation and upload |
| `inq smoke` | `smoke.py` | Provider health checks and smoke tests |
| `inq databricks` | `databricks.py` | Databricks cluster and job management |
| `inq ui` | `ui.py` | Streamlit UI launcher |

## Architecture

```
cli/
├── app.py           # Root Typer app, registers all sub-apps
├── _util.py         # Shared helpers (run, REPO_ROOT, env file paths)
├── __main__.py      # `python -m cli` entry point
├── docker.py        # Docker Compose lifecycle
├── dev.py           # Development server and config validation
├── test.py          # Test runner shortcuts
├── search.py        # Semantic search commands
├── ray.py           # Ray job submission
├── vectordb.py      # Vector DB management
├── synthetic.py     # Synthetic data generation
├── smoke.py         # Health checks and smoke tests
├── databricks.py    # Databricks cluster/job management
└── ui.py            # Streamlit UI launcher
```

## Adding a New Command Group

1. Create `src/cli/<name>.py` with a `typer.Typer()` app
2. Import and register in `app.py`: `app.add_typer(<module>.app, name="<name>")`
3. Add tests in `tests/unit/cli/test_<name>.py`

## Entry Point

The CLI is registered as a console script in `pyproject.toml`:

```toml
[project.scripts]
inq = "cli.app:app"
```
