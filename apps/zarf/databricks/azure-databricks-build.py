#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


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
    return subprocess.run(cmd, text=True, check=False, capture_output=True)


def _load_cluster_spec(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in cluster spec: {path}", file=sys.stderr)
        print(f"{exc}", file=sys.stderr)
        print("Ensure the file is valid JSON (not YAML) and not empty.", file=sys.stderr)
        sys.exit(1)


def main() -> int:
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
        print(f"Missing env file: {env_file}", file=sys.stderr)
        print("Create it from zarf/databricks/dev/env.local.example or set ENV_FILE.", file=sys.stderr)
        return 1

    if not cluster_spec_file.exists():
        print(f"Missing cluster spec file: {cluster_spec_file}", file=sys.stderr)
        return 1

    env_data = _read_env_file(env_file)
    os.environ.update(env_data)

    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        print("Missing required Databricks env vars (DATABRICKS_HOST, DATABRICKS_TOKEN).", file=sys.stderr)
        return 1

    if shutil.which("databricks") is None:
        print("Missing databricks CLI. Install it to use this target.", file=sys.stderr)
        return 1

    cluster_spec = _load_cluster_spec(cluster_spec_file)
    spec = cluster_spec.get("spec", cluster_spec)

    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID") or cluster_spec.get("cluster_id")
    if cluster_id:
        exists = _run(["databricks", "clusters", "get", "--cluster-id", cluster_id]).returncode == 0
        if exists:
            print(f"Cluster {cluster_id} already exists; skipping build.")
        else:
            print(
                f"Cluster {cluster_id} not found; creating from {cluster_spec_file} (cluster_id ignored)."
            )
            _run(["databricks", "clusters", "create", "--json", json.dumps(spec)])
    else:
        print(f"Creating cluster from {cluster_spec_file}...")
        _run(["databricks", "clusters", "create", "--json", json.dumps(spec)])

    job_id = os.environ.get("DATABRICKS_JOB_ID")
    if job_spec_file.exists() and job_id:
        job_exists = _run(["databricks", "jobs", "get", "--job-id", job_id]).returncode == 0
        if job_exists:
            print(f"Job {job_id} already exists; skipping job create.")
        else:
            print(f"Job {job_id} not found; create it from {job_spec_file} if needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
