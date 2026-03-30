"""Synthetic data command group."""

from __future__ import annotations

import shutil
from typing import Annotated

import typer

from cli._util import run, REPO_ROOT

app = typer.Typer(help="Synthetic data generation.")


@app.command()
def generate(
    count: Annotated[int, typer.Option("--count", "-c", help="Number of images to generate.")] = 100,
    size: Annotated[int, typer.Option("--size", "-s", help="Image size in pixels.")] = 512,
) -> None:
    """Generate synthetic images."""
    run(
        [
            "uv",
            "run",
            "python",
            "syntheticdata/synthetic_data.py",
            "generate-images",
            "--count",
            str(count),
            "--size",
            str(size),
        ]
    )


@app.command()
def upload(
    endpoint: Annotated[
        str, typer.Option("--endpoint", "-e", help="MinIO endpoint URL.")
    ] = "http://localhost:9000",
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="S3 prefix for uploads.")] = "images/",
) -> None:
    """Upload synthetic images to MinIO."""
    run(
        [
            "uv",
            "run",
            "python",
            "syntheticdata/synthetic_data.py",
            "upload-images",
            "--endpoint",
            endpoint,
            "--prefix",
            prefix,
        ]
    )


@app.command()
def setup(
    count: Annotated[int, typer.Option("--count", "-c", help="Number of images to generate.")] = 100,
    size: Annotated[int, typer.Option("--size", "-s", help="Image size in pixels.")] = 512,
    endpoint: Annotated[
        str, typer.Option("--endpoint", "-e", help="MinIO endpoint URL.")
    ] = "http://localhost:9000",
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="S3 prefix for uploads.")] = "images/",
) -> None:
    """Generate and upload synthetic images (convenience command)."""
    run(
        [
            "uv",
            "run",
            "python",
            "syntheticdata/synthetic_data.py",
            "setup-images",
            "--count",
            str(count),
            "--size",
            str(size),
            "--endpoint",
            endpoint,
            "--prefix",
            prefix,
        ]
    )


@app.command()
def clean() -> None:
    """Remove generated synthetic images."""
    typer.echo("Removing generated synthetic images...")
    imgs_dir = REPO_ROOT / "syntheticdata" / "data" / "imgs"
    if imgs_dir.exists():
        shutil.rmtree(imgs_dir)
        typer.echo(f"✅ Cleaned {imgs_dir}")
    else:
        typer.echo(f"Directory not found: {imgs_dir}")
