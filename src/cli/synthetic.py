"""Synthetic data command group."""

from __future__ import annotations

import shutil
import sys
from typing import Annotated

import typer

from cli._util import REPO_ROOT

# Add bench/synthetic to sys.path so ImageGenerator/MinIOUploader can be imported.
_SYNTHETIC_DIR = REPO_ROOT / "bench" / "synthetic"
if str(_SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNTHETIC_DIR))

from synthetic_data import ImageGenerator, MinIOUploader  # noqa: E402

app = typer.Typer(help="Synthetic data generation.")

# Default paths
_DEFAULT_OUTPUT_DIR = _SYNTHETIC_DIR / "data" / "imgs"


@app.command()
def generate(
    count: Annotated[int, typer.Option("--count", "-c", help="Number of images to generate.")] = 100,
    size: Annotated[int, typer.Option("--size", "-s", help="Image size in pixels.")] = 512,
) -> None:
    """Generate synthetic images."""
    generator = ImageGenerator(output_dir=_DEFAULT_OUTPUT_DIR, size=size)
    generator.generate(count=count)


@app.command()
def upload(
    endpoint: Annotated[
        str, typer.Option("--endpoint", "-e", help="MinIO endpoint URL.")
    ] = "http://localhost:9000",
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="S3 prefix for uploads.")] = "images/",
) -> None:
    """Upload synthetic images to MinIO."""
    uploader = MinIOUploader(endpoint=endpoint)
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=_DEFAULT_OUTPUT_DIR,
            prefix=prefix,
            pattern="*.png",
        )
        if failed > 0:
            raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None


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
    generator = ImageGenerator(output_dir=_DEFAULT_OUTPUT_DIR, size=size)
    generator.generate(count=count)

    uploader = MinIOUploader(endpoint=endpoint)
    try:
        _successful, failed = uploader.upload_directory(
            input_dir=_DEFAULT_OUTPUT_DIR,
            prefix=prefix,
            pattern="*.png",
        )
        if failed > 0:
            raise typer.Exit(code=1)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def clean() -> None:
    """Remove generated synthetic images."""
    typer.echo("Removing generated synthetic images...")
    imgs_dir = REPO_ROOT / "bench" / "synthetic" / "data" / "imgs"
    if imgs_dir.exists():
        shutil.rmtree(imgs_dir)
        typer.echo(f"Cleaned {imgs_dir}")
    else:
        typer.echo(f"Directory not found: {imgs_dir}")
