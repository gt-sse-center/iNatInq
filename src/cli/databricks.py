"""Databricks command group for cluster management."""

from __future__ import annotations

from typing import Annotated

import typer

from cli._util import ensure_databricks_env, run, DATABRICKS_ENV_FILE, DATABRICKS_CLUSTER_SPEC

app = typer.Typer(help="Azure Databricks cluster management.")


@app.command()
def build(
    env_file: Annotated[str, typer.Option("--env-file", help="Databricks env file.")] = DATABRICKS_ENV_FILE,
    cluster_spec: Annotated[
        str, typer.Option("--cluster-spec", help="Cluster spec file.")
    ] = DATABRICKS_CLUSTER_SPEC,
) -> None:
    """Create or update Databricks cluster from spec."""
    ensure_databricks_env(env_file)
    run(
        ["zarf/databricks/azure-databricks-build.py"],
        env={"ENV_FILE": env_file, "CLUSTER_SPEC_FILE": cluster_spec},
    )


@app.command()
def up(
    env_file: Annotated[str, typer.Option("--env-file", help="Databricks env file.")] = DATABRICKS_ENV_FILE,
    cluster_spec: Annotated[
        str, typer.Option("--cluster-spec", help="Cluster spec file.")
    ] = DATABRICKS_CLUSTER_SPEC,
) -> None:
    """Start Databricks cluster."""
    ensure_databricks_env(env_file)
    run(
        ["zarf/databricks/azure-databricks-up.py"],
        env={"ENV_FILE": env_file, "CLUSTER_SPEC_FILE": cluster_spec},
    )


@app.command()
def down(
    env_file: Annotated[str, typer.Option("--env-file", help="Databricks env file.")] = DATABRICKS_ENV_FILE,
    cluster_spec: Annotated[
        str, typer.Option("--cluster-spec", help="Cluster spec file.")
    ] = DATABRICKS_CLUSTER_SPEC,
) -> None:
    """Terminate Databricks cluster."""
    ensure_databricks_env(env_file)
    run(
        ["zarf/databricks/azure-databricks-down.py"],
        env={"ENV_FILE": env_file, "CLUSTER_SPEC_FILE": cluster_spec},
    )


@app.command(name="cdc-notebooks")
def cdc_notebooks(
    env_file: Annotated[str, typer.Option("--env-file", help="Databricks env file.")] = DATABRICKS_ENV_FILE,
) -> None:
    """Upload and run CDC validation notebooks."""
    ensure_databricks_env(env_file)
    run(
        [
            "uv",
            "run",
            "zarf/databricks/azure-databricks-cdc-notebooks.py",
            "--env-file",
            env_file,
            "--upload-notebooks",
            "--run-notebooks",
        ]
    )


@app.command(name="configure-s3a")
def configure_s3a(
    env_file: Annotated[str, typer.Option("--env-file", help="Databricks env file.")] = DATABRICKS_ENV_FILE,
) -> None:
    """Configure MinIO S3A access for Databricks cluster."""
    ensure_databricks_env(env_file)
    run(["zarf/databricks/azure-databricks-configure-minio-s3a.py"], env={"ENV_FILE": env_file})
