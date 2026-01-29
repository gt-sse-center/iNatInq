#!/usr/bin/env python3
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


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = Path(os.environ.get("ENV_FILE", repo_root / "zarf/databricks/dev/.env.local"))
    cluster_spec_file = Path(
        os.environ.get(
            "CLUSTER_SPEC_FILE",
            repo_root / "zarf/databricks/dev/inatinq-azure-databricks-cluster.json",
        )
    )

    if not env_file.exists():
        print(f"Missing env file: {env_file}", file=sys.stderr)
        print("Create it from zarf/databricks/dev/env.local.example or set ENV_FILE.", file=sys.stderr)
        return 1

    if not cluster_spec_file.exists():
        print(f"Missing cluster spec file: {cluster_spec_file}", file=sys.stderr)
        return 1

    os.environ.update(_read_env_file(env_file))

    if not os.environ.get("DATABRICKS_HOST") or not os.environ.get("DATABRICKS_TOKEN"):
        print("Missing required Databricks env vars (DATABRICKS_HOST, DATABRICKS_TOKEN).", file=sys.stderr)
        return 1

    if shutil.which("databricks") is None:
        print("Missing databricks CLI. Install it to use this target.", file=sys.stderr)
        return 1

    cluster_id = os.environ.get("DATABRICKS_CLUSTER_ID")
    if not cluster_id:
        print(
            "Missing cluster id. Set DATABRICKS_CLUSTER_ID or include it in the cluster spec.",
            file=sys.stderr,
        )
        return 1

    print(f"Terminating cluster {cluster_id}...")
    result = _run(["databricks", "clusters", "delete", "--cluster-id", cluster_id])
    if result.returncode != 0:
        print(result.stderr.strip() or "Failed to terminate cluster.", file=sys.stderr)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
