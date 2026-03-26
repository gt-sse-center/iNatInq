"""Dev command group for development tasks."""

from __future__ import annotations

import typer

from cli._util import run

app = typer.Typer(help="Development tasks.")


@app.command()
def lint() -> None:
    """Run linters."""
    typer.echo("Running linters...")
    run(["uv", "run", "ruff", "check", "src/", "tests/"])


@app.command(name="format")
def format_code() -> None:
    """Format code."""
    typer.echo("Formatting code...")
    run(["uv", "run", "ruff", "format", "src/", "tests/"])
    run(["uv", "run", "ruff", "check", "--fix", "src/", "tests/"])


@app.command()
def typecheck() -> None:
    """Run type checker."""
    typer.echo("Running type checker...")
    run(["uv", "run", "mypy", "src/"])


@app.command(name="validate-config")
def validate_config() -> None:
    """Validate YAML configuration."""
    typer.echo("Validating YAML configuration...")
    run(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "from config_loader import load_yaml_config; "
                "c = load_yaml_config(); "
                "print('Base config: OK (' + str(len(c)) + ' top-level keys)'); "
                "import os, pathlib; "
                "env_dir = pathlib.Path('configs/environments'); "
                "[print(f'  + {f.stem}: OK') for f in sorted(env_dir.glob('*.yaml')) if load_yaml_config(env=f.stem) is not None]; "
                "print('All configs valid.')"
            ),
        ]
    )


@app.command()
def serve() -> None:
    """Start development server."""
    typer.echo("Starting development server...")
    run(["uv", "run", "uvicorn", "api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])
