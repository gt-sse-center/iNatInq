"""Smoke test command group."""

from __future__ import annotations

import typer

from cli._util import run, SMOKE_ENV_FILE

app = typer.Typer(help="Smoke tests.")


@app.command()
def health() -> None:
    """Run provider health checks."""
    run(["bash", "zarf/scripts/providers_health.sh"], env={"ENV_FILE": SMOKE_ENV_FILE})


@app.command()
def providers() -> None:
    """Run provider smoke tests."""
    run(["bash", "zarf/scripts/smoke_providers.sh"], env={"ENV_FILE": SMOKE_ENV_FILE})


@app.command(name="all")
def all_checks() -> None:
    """Run all smoke tests (health then providers)."""
    health()
    providers()
