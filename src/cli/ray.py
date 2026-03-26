"""Ray command group for job management."""

from __future__ import annotations

from typing import Annotated

import requests
import typer

app = typer.Typer(help="Ray job management.")


@app.command()
def submit(
    prefix: Annotated[str, typer.Option("--prefix", "-p", help="S3 prefix for images.")] = "images/",
    collection: Annotated[
        str, typer.Option("--collection", "-c", help="Vector DB collection name.")
    ] = "documents",
) -> None:
    """Submit Ray image job to process S3 images."""
    typer.echo(f"Submitting Ray image job to process S3 prefix: {prefix}")

    try:
        response = requests.post(
            "http://localhost:8000/ray/jobs/images",
            json={"s3_bucket": "pipeline", "s3_prefix": prefix, "collection": collection},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        job_id = data.get("job_id", "N/A")
        typer.echo(f"Job ID: {job_id}")
    except requests.RequestException as e:
        typer.echo(f"Error submitting job: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def status() -> None:
    """Show status of Ray jobs."""
    typer.echo("Fetching Ray job status...")

    try:
        response = requests.get("http://localhost:8265/api/jobs/", timeout=5)
        response.raise_for_status()
        jobs = response.json()

        if not jobs:
            typer.echo("No jobs found")
            return

        for job in jobs[:5]:
            job_id = job.get("job_id", "N/A")[:20]
            job_status = job.get("status", "unknown")
            typer.echo(f"{job_id}: {job_status}")
    except requests.RequestException as e:
        typer.echo(f"Error fetching jobs: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command()
def logs() -> None:
    """Show logs for the latest Ray job."""
    try:
        # Get latest job ID
        response = requests.get("http://localhost:8265/api/jobs/", timeout=5)
        response.raise_for_status()
        jobs = response.json()

        if not jobs:
            typer.echo("No jobs found")
            return

        job_id = jobs[0]["job_id"]
        typer.echo(f"Fetching logs for job: {job_id}")

        # Get logs
        logs_response = requests.get(f"http://localhost:8265/api/jobs/{job_id}/logs", timeout=5)
        logs_response.raise_for_status()
        typer.echo(logs_response.text)
    except requests.RequestException as e:
        typer.echo(f"Error fetching logs: {e}", err=True)
        raise typer.Exit(code=1) from e
