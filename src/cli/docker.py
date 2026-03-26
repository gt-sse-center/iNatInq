"""Docker command group for managing compose services."""

from __future__ import annotations

import shutil
import subprocess
from enum import StrEnum
from typing import Annotated

import typer

from cli._util import docker_compose, run, run_interactive, SMOKE_ENV_FILE

app = typer.Typer(help="Docker Compose operations.")


class Service(StrEnum):
    """Available services for logs and shell access."""

    pipeline = "pipeline"
    qdrant = "qdrant"
    ray_head = "ray-head"
    minio = "minio"
    clip = "clip"
    redis = "redis"


@app.command()
def up() -> None:
    """Start all services."""
    typer.echo("Starting all services...")
    run(docker_compose("up", "-d"))

    typer.echo("")
    typer.echo("✅ Services started!")
    typer.echo("")
    typer.echo("📝 Service endpoints:")
    typer.echo("   Pipeline API:     http://localhost:8000")
    typer.echo("   Pipeline Docs:    http://localhost:8000/docs")
    typer.echo("   MinIO Console:    http://localhost:9001")
    typer.echo("   Qdrant Dashboard: http://localhost:6333/dashboard")
    typer.echo("   Ray Dashboard:    http://localhost:8265")
    typer.echo("   CLIP API:         http://localhost:8001")


@app.command()
def down() -> None:
    """Stop all services."""
    typer.echo("Stopping all services...")
    run(docker_compose("down"))


@app.command()
def logs() -> None:
    """Tail logs from all services."""
    run(docker_compose("logs", "-f"))


@app.command(name="log")
def log_service(
    service: Annotated[Service, typer.Argument(help="Service to tail logs from.")],
) -> None:
    """Tail logs from a specific service."""
    run(docker_compose("logs", "-f", service.value))


@app.command()
def ps() -> None:
    """Show container status."""
    run(docker_compose("ps"))


@app.command(name="build-base")
def build_base() -> None:
    """Build pipeline base image (heavy dependencies)."""
    typer.echo("Building pipeline base image (heavy dependencies)...")
    typer.echo("This may take 5-10 minutes on first build...")
    run(
        [
            "docker",
            "build",
            "-f",
            "zarf/docker/base/Dockerfile.pipeline-base",
            "-t",
            "inatinq/pipeline-base:0.1.0",
            ".",
        ]
    )
    typer.echo("✅ Base image built: inatinq/pipeline-base:0.1.0")


def _ensure_base_image() -> None:
    """Ensure the pipeline base image exists, building it if needed."""
    try:
        subprocess.run(
            ["docker", "image", "inspect", "inatinq/pipeline-base:0.1.0"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        typer.echo("⚠️  Base image not found. Building it first...")
        build_base()


@app.command()
def build() -> None:
    """Build pipeline image."""
    typer.echo("Building pipeline image...")
    _ensure_base_image()
    run(docker_compose("build", "pipeline"))


@app.command()
def rebuild() -> None:
    """Rebuild and restart pipeline."""
    typer.echo("Rebuilding and restarting pipeline...")
    _ensure_base_image()
    run(docker_compose("up", "-d", "--build", "pipeline"))


@app.command()
def clean() -> None:
    """Stop services and remove volumes."""
    typer.echo("⚠️  Stopping services and removing volumes...")
    run(docker_compose("down", "-v"))
    typer.echo("✅ Cleanup complete")


@app.command()
def restart() -> None:
    """Restart all services."""
    typer.echo("Restarting all services...")
    run(docker_compose("restart"))
    typer.echo("✅ Services restarted")


@app.command()
def health() -> None:
    """Check health status of all services."""
    typer.echo("╔══════════════════════════════════════════════════════════════════════════╗")
    typer.echo("║                         Service Health Status                            ║")
    typer.echo("╚══════════════════════════════════════════════════════════════════════════╝")
    typer.echo("")

    # Show container status
    run(docker_compose("ps", "--format", "table {{.Name}}\\t{{.Status}}\\t{{.Ports}}"))
    typer.echo("")

    # Delegate provider health to script
    run(["bash", "zarf/scripts/providers_health.sh"], env={"ENV_FILE": SMOKE_ENV_FILE})

    # Inline health checks
    typer.echo("┌── Local Compose Health Checks ───────────────────────────────────────────────┐")

    # Pipeline API
    typer.echo("│ Pipeline API:    ", nl=False)
    try:
        subprocess.run(
            ["curl", "-sf", "http://localhost:8000/healthz"],
            check=True,
            capture_output=True,
        )
        typer.echo("✅ healthy")
    except subprocess.CalledProcessError:
        typer.echo("❌ unhealthy")

    # Qdrant
    typer.echo("│ Qdrant:          ", nl=False)
    try:
        subprocess.run(
            ["curl", "-sf", "http://localhost:6333/readyz"],
            check=True,
            capture_output=True,
        )
        typer.echo("✅ healthy")
    except subprocess.CalledProcessError:
        typer.echo("❌ unhealthy")

    # MinIO
    typer.echo("│ MinIO:           ", nl=False)
    try:
        subprocess.run(
            ["curl", "-sf", "http://localhost:9000/minio/health/live"],
            check=True,
            capture_output=True,
        )
        typer.echo("✅ healthy")
    except subprocess.CalledProcessError:
        typer.echo("❌ unhealthy")

    # Ray
    typer.echo("│ Ray:             ", nl=False)
    try:
        subprocess.run(
            ["curl", "-sf", "http://localhost:8265"],
            check=True,
            capture_output=True,
        )
        typer.echo("✅ healthy")
    except subprocess.CalledProcessError:
        typer.echo("❌ unhealthy")

    # Redis
    typer.echo("│ Redis:           ", nl=False)
    try:
        subprocess.run(
            ["docker", "exec", "redis", "redis-cli", "ping"],
            check=True,
            capture_output=True,
        )
        typer.echo("✅ healthy")
    except subprocess.CalledProcessError:
        typer.echo("❌ unhealthy")

    # CLIP
    typer.echo("│ CLIP:            ", nl=False)
    try:
        subprocess.run(
            ["curl", "-sf", "http://localhost:8001/"],
            check=True,
            capture_output=True,
        )
        typer.echo("✅ healthy")
    except subprocess.CalledProcessError:
        typer.echo("❌ unhealthy")

    typer.echo("└──────────────────────────────────────────────────────────────────────────────┘")
    typer.echo("")
    typer.echo("📝 Local Compose Endpoints:")
    typer.echo("   Pipeline API:     http://localhost:8000")
    typer.echo("   Pipeline Docs:    http://localhost:8000/docs")
    typer.echo("   MinIO Console:    http://localhost:9001 (minioadmin/minioadmin)")
    typer.echo("   Qdrant Dashboard: http://localhost:6333/dashboard")
    typer.echo("   Ray Dashboard:    http://localhost:8265")
    typer.echo("   CLIP API:         http://localhost:8001")


@app.command()
def shell(
    service: Annotated[
        Service,
        typer.Argument(help="Service to open shell in. Redis uses redis-cli."),
    ],
) -> None:
    """Open interactive shell in a service container."""
    if service == Service.redis:
        typer.echo("Opening redis-cli in Redis container...")
        run_interactive(docker_compose("exec", "-it", "redis", "redis-cli"))
    else:
        typer.echo(f"Opening shell in {service} container...")
        run_interactive(
            docker_compose("exec", "pipeline" if service == Service.pipeline else service.value, "/bin/bash")
        )


@app.command()
def scale(
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of Ray worker replicas.")] = 1,
) -> None:
    """Scale Ray workers."""
    typer.echo(f"Scaling Ray workers to {workers} replicas...")
    run(docker_compose("up", "-d", "--scale", f"ray-worker={workers}"))
    typer.echo(f"✅ Ray workers scaled to {workers}")


@app.command()
def lazy() -> None:
    """Open lazydocker TUI."""
    if shutil.which("lazydocker") is None:
        typer.echo("❌ lazydocker not installed. Run: brew install lazydocker", err=True)
        raise typer.Exit(code=1)

    run_interactive(["lazydocker"])
