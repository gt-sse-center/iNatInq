"""UI command group for opening web dashboards."""

from __future__ import annotations

import typer

from cli._util import open_browser

app = typer.Typer(help="Open web dashboards.")


@app.command(name="all")
def all_uis() -> None:
    """Open all web UIs."""
    typer.echo("Opening all UIs...")
    pipeline()
    minio()
    qdrant()
    ray()


@app.command()
def pipeline() -> None:
    """Open Pipeline API docs."""
    url = "http://localhost:8000/docs"
    typer.echo(f"Opening Pipeline API docs: {url}")
    open_browser(url)


@app.command()
def minio() -> None:
    """Open MinIO console."""
    url = "http://localhost:9001"
    typer.echo(f"Opening MinIO console: {url}")
    typer.echo("Login: minioadmin / minioadmin")
    open_browser(url)


@app.command()
def qdrant() -> None:
    """Open Qdrant dashboard."""
    url = "http://localhost:6333/dashboard"
    typer.echo(f"Opening Qdrant dashboard: {url}")
    open_browser(url)


@app.command()
def ray() -> None:
    """Open Ray dashboard."""
    url = "http://localhost:8265"
    typer.echo(f"Opening Ray dashboard: {url}")
    open_browser(url)
