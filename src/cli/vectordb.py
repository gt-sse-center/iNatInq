"""Vector DB command group including S3 and E2E operations."""

from __future__ import annotations

import time
from typing import Annotated

import requests
import typer

from cli._util import docker_compose, run, DEFAULT_S3_BUCKET

app = typer.Typer(help="Vector database and E2E operations.")


@app.command()
def count(
    collection: Annotated[str, typer.Option("--collection", "-c", help="Collection name.")] = "documents",
) -> None:
    """Count documents in Qdrant collection."""
    typer.echo(f"Counting documents in Qdrant collection '{collection}'...")

    try:
        response = requests.get(f"http://localhost:6333/collections/{collection}", timeout=5)
        response.raise_for_status()
        data = response.json()
        points_count = data.get("result", {}).get("points_count", 0)
        typer.echo(f"Qdrant documents: {points_count}")
    except requests.RequestException as e:
        typer.echo(f"Qdrant: Collection not found or service unavailable ({e})", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def clear(
    collection: Annotated[str, typer.Option("--collection", "-c", help="Collection name.")] = "documents",
) -> None:
    """Delete Qdrant collection."""
    typer.echo(f"Deleting Qdrant collection '{collection}'...")

    try:
        response = requests.delete(f"http://localhost:6333/collections/{collection}", timeout=5)
        response.raise_for_status()
        typer.echo("✅ Qdrant collection deleted")
    except requests.RequestException as e:
        typer.echo(f"Qdrant delete failed or collection doesn't exist ({e})", err=True)
        raise typer.Exit(code=1) from e


@app.command(name="s3-count")
def s3_count(
    bucket: Annotated[str, typer.Option("--bucket", "-b", help="S3 bucket name.")] = DEFAULT_S3_BUCKET,
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="S3 prefix.")] = "images/",
) -> None:
    """Count objects in MinIO bucket."""
    typer.echo("Counting objects in MinIO bucket...")

    try:
        result = run(
            docker_compose("exec", "-T", "minio", "mc", "ls", "--recursive", f"minio/{bucket}/{prefix}"),
            capture=True,
        )
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        count = len([line for line in lines if line])
        typer.echo(f"S3 objects: {count}")
    except Exception as e:
        typer.echo(f"S3 count failed or service unavailable ({e})", err=True)
        raise typer.Exit(code=1) from e


@app.command(name="s3-clear")
def s3_clear(
    bucket: Annotated[str, typer.Option("--bucket", "-b", help="S3 bucket name.")] = DEFAULT_S3_BUCKET,
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="S3 prefix.")] = "images/",
) -> None:
    """Clear objects from MinIO bucket."""
    typer.echo(f"Clearing S3 bucket at prefix {prefix}...")

    try:
        run(
            docker_compose(
                "exec", "-T", "minio", "mc", "rm", "--recursive", "--force", f"minio/{bucket}/{prefix}"
            )
        )
        typer.echo("✅ S3 cleared")
    except Exception as e:
        typer.echo(f"S3 clear failed ({e})", err=True)
        raise typer.Exit(code=1) from e


@app.command(name="e2e")
def e2e_images(
    image_count: Annotated[int, typer.Option("--count", "-c", help="Number of images to generate.")] = 100,
    size: Annotated[int, typer.Option("--size", "-s", help="Image size in pixels.")] = 512,
    collection: Annotated[str, typer.Option("--collection", help="Vector DB collection name.")] = "documents",
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="S3 prefix.")] = "images/",
    wait: Annotated[int, typer.Option("--wait", "-w", help="Seconds to wait after job submission.")] = 60,
) -> None:
    """Run end-to-end image pipeline test."""
    typer.echo("╔══════════════════════════════════════════════════════════════════════════╗")
    typer.echo("║                      End-to-End Image Pipeline Test                      ║")
    typer.echo("╚══════════════════════════════════════════════════════════════════════════╝")
    typer.echo("")

    # Step 1: Clear collections
    typer.echo("Step 1: Clear image collections...")
    try:
        clear(collection=collection)
    except typer.Exit:
        typer.echo("⚠️  Collection clear failed, continuing...")
    typer.echo("")

    # Step 2: Generate and upload images
    typer.echo(f"Step 2: Generate and upload {image_count} images...")
    from cli import synthetic as synthetic_module

    synthetic_module.setup(count=image_count, size=size, prefix=prefix)
    typer.echo("")

    # Step 3: Submit Ray job
    typer.echo("Step 3: Submit Ray image job...")
    from cli import ray as ray_module

    ray_module.submit(prefix=prefix, collection=collection)
    typer.echo("")

    # Step 4: Wait
    typer.echo(f"Waiting {wait} seconds for image job to complete...")
    time.sleep(wait)
    typer.echo("")

    # Step 5: Check counts
    typer.echo("Step 4: Check image counts...")
    try:
        count(collection=collection)
    except typer.Exit:
        typer.echo("⚠️  Count check failed")
    typer.echo("")

    # Step 6: Test search
    typer.echo("Step 5: Test image search...")
    from cli import search as search_module

    try:
        search_module.images(query="red circle", limit=3)
    except typer.Exit:
        typer.echo("⚠️  Search failed")
    typer.echo("")

    typer.echo("✅ End-to-end image test complete!")
