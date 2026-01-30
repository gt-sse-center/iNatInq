#!/usr/bin/env python3
"""Create Databricks cluster and validate job configuration."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _write(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{message}\n")


def _read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # Commands are assembled from fixed args within this script.
    return subprocess.run(cmd, text=True, check=False, capture_output=True)  # noqa: S603


def _load_cluster_spec(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _write(f"Invalid JSON in cluster spec: {path}", error=True)
        _write(f"{exc}", error=True)
        _write("Ensure the file is valid JSON (not YAML) and not empty.", error=True)
        sys.exit(1)


def main() -> int:
    """Ensure a Databricks cluster exists and validate job specs."""
    repo_root = Path(__file__).resolve().parents[2]
    env_file = Path(os.environ.get("ENV_FILE", repo_root / "zarf/databricks/dev/.env.local"))
    cluster_spec_file = Path(
        os.environ.get(
            "CLUSTER_SPEC_FILE",
            repo_root / "zarf/databricks/dev/inatinq-azure-databricks-cluster.json",
        )
    )
    job_spec_file = Path(
        os.environ.get(
            "JOB_SPEC_FILE",
            repo_root / "zarf/databricks/dev/inatinq-ml-pipeline-job.yml",
        )
    )

    if not env_file.exists():
        _write(f"Missing env file: {env_file}", error=True)
        _write(
            "Create it from zarf/databricks/dev/env.local.example or set ENV_FILE.",
            error=True,
        )
        return 1

    if not cluster_spec_file.exists():
        _write(f"Missing cluster spec file: {cluster_spec_file}", error=True)
        return 1

    env_data = _read_env_file(env_file)
    os.environ.update(env_data)

    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        _write(
            "Missing required Databricks env vars (DATABRICKS_HOST, DATABRICKS_TOKEN).",
            error=True,
        )
        return 1

    if shutil.which("databricks") is None:
        _write("Missing databricks CLI. Install it to use this target.", error=True)
        return 1

    cluster_spec = _load_cluster_spec(cluster_spec_file)
    spec = cluster_spec.get("spec", cluster_spec)

    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID") or cluster_spec.get("cluster_id")
    if cluster_id:
        exists = _run(["databricks", "clusters", "get", "--cluster-id", cluster_id]).returncode == 0
        if exists:
            _write(f"Cluster {cluster_id} already exists; skipping build.")
        else:
            _write(f"Cluster {cluster_id} not found; creating from {cluster_spec_file} (cluster_id ignored).")
            cmd = ["databricks", "clusters", "create", "--json", json.dumps(spec)]
            result = _run(cmd)
            if result.returncode != 0:
                _write(f"Command failed: {' '.join(cmd)}", error=True)
                _write(result.stderr.strip() or "Failed to create Databricks cluster.", error=True)
                return result.returncode
    else:
        _write(f"Creating cluster from {cluster_spec_file}...")
        cmd = ["databricks", "clusters", "create", "--json", json.dumps(spec)]
        result = _run(cmd)
        if result.returncode != 0:
            _write(f"Command failed: {' '.join(cmd)}", error=True)
            _write(result.stderr.strip() or "Failed to create Databricks cluster.", error=True)
            return result.returncode

    job_id = os.environ.get("DATABRICKS_JOB_ID")
    if job_spec_file.exists() and job_id:
        job_exists = _run(["databricks", "jobs", "get", "--job-id", job_id]).returncode == 0
        if job_exists:
            _write(f"Job {job_id} already exists; skipping job create.")
        else:
            _write(f"Job {job_id} not found; create it from {job_spec_file} if needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
