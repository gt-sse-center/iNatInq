"""iNatInq developer CLI - main application."""

from __future__ import annotations

import importlib.metadata
from typing import Annotated

import typer

# Import sub-apps (deferred to avoid circular imports until all groups are created)
from cli import databricks as databricks_module
from cli import dev as dev_module
from cli import docker as docker_module
from cli import ray as ray_module
from cli import search as search_module
from cli import smoke as smoke_module
from cli import synthetic as synthetic_module
from cli import test as test_module
from cli import ui as ui_module
from cli import vectordb as vectordb_module

# Create the root CLI app
app = typer.Typer(help="iNatInq developer CLI.")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        version = importlib.metadata.version("apps")
        typer.echo(f"iNatInq CLI version {version}")
        raise typer.Exit()


@app.callback()
def main(
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Developer CLI for iNatInq."""


# Top-level convenience aliases
@app.command()
def up() -> None:
    """Start all services (alias for 'inq docker up')."""
    docker_module.up()


@app.command()
def down() -> None:
    """Stop all services (alias for 'inq docker down')."""
    docker_module.down()


@app.command()
def status() -> None:
    """Show service health status (alias for 'inq docker health')."""
    docker_module.health()


# Register sub-apps
app.add_typer(docker_module.app, name="docker")
app.add_typer(dev_module.app, name="dev")
app.add_typer(test_module.app, name="test")
app.add_typer(ui_module.app, name="ui")
app.add_typer(smoke_module.app, name="smoke")
app.add_typer(synthetic_module.app, name="synthetic")
app.add_typer(search_module.app, name="search")
app.add_typer(ray_module.app, name="ray")
app.add_typer(vectordb_module.app, name="vectordb")
app.add_typer(databricks_module.app, name="databricks")
