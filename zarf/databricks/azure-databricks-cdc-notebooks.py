#!/usr/bin/env python3
"""Bootstrap and optionally run CDC Databricks notebooks.

This helper does four things:
1. Loads Databricks credentials and defaults from an env file.
2. Starts the configured cluster and waits until it is RUNNING.
3. Ensures CDC notebooks exist in the Databricks workspace (optionally uploads them).
4. Optionally submits producer/consumer notebooks as one-time Databricks runs.
"""

from __future__ import annotations

import argparse
import configparser
import datetime as dt
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.compute import State
from databricks.sdk.service.jobs import NotebookTask, Run, RunResultState, Source, SubmitTask
from databricks.sdk.service.workspace import ImportFormat, Language


def _write(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{message}\n")


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description="Start Databricks cluster and prepare CDC notebooks")
    parser.add_argument(
        "--env-file",
        default=str(repo_root / "zarf/databricks/dev/.env.local"),
        help="Path to env file (default: zarf/databricks/dev/.env.local)",
    )
    parser.add_argument(
        "--cluster-id",
        default="",
        help="Databricks cluster id override (default: DATABRICKS_CLUSTER_ID from env file)",
    )
    parser.add_argument(
        "--workspace-repo-root",
        default="",
        help=(
            "Workspace repo root override, e.g. /Workspace/Users/<user>/iNatInq "
            "(default derives from INATINQ_SRC_DIR parent)"
        ),
    )
    parser.add_argument(
        "--workspace-env-file",
        default="",
        help="Optional workspace-visible env file path passed to notebook widget env_file",
    )
    parser.add_argument(
        "--cluster-wait-minutes",
        type=int,
        default=30,
        help="Minutes to wait for cluster start (default: 30)",
    )
    parser.add_argument(
        "--configure-cli",
        action="store_true",
        help="Write Databricks CLI profile (~/.databrickscfg) from env credentials",
    )
    parser.add_argument(
        "--configure-cli-via-command",
        action="store_true",
        help="Try `databricks configure --host ... --token` instead of writing config file directly",
    )
    parser.add_argument(
        "--profile",
        default="DEFAULT",
        help="CLI profile name when --configure-cli is used (default: DEFAULT)",
    )
    parser.add_argument(
        "--upload-notebooks",
        action="store_true",
        help="Upload local notebook sources to workspace paths (overwrite existing)",
    )
    parser.add_argument(
        "--open-notebooks",
        action="store_true",
        help="Open producer/consumer notebook URLs in browser after setup",
    )
    parser.add_argument(
        "--run-notebooks",
        action="store_true",
        help="Submit producer and consumer notebooks as one-time Databricks runs",
    )
    parser.add_argument(
        "--only",
        choices=["producer", "consumer", "both"],
        default="both",
        help="Which notebook runs to submit when --run-notebooks is set (default: both)",
    )
    parser.add_argument(
        "--catalog",
        default="",
        help="Notebook widget override: catalog",
    )
    parser.add_argument(
        "--schema",
        default="",
        help="Notebook widget override: schema",
    )
    parser.add_argument(
        "--test-run-suffix",
        default="",
        help="Notebook widget override: test_run_suffix (used to build ephemeral test tables)",
    )
    parser.add_argument(
        "--progress-id",
        default="",
        help="Notebook widget override: progress_id",
    )
    parser.add_argument(
        "--collection",
        default="",
        help="Notebook widget override: collection",
    )
    parser.add_argument(
        "--window-size",
        default="",
        help="Notebook widget override: window_size",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=int,
        default=3600,
        help="Timeout for each notebook run when --run-notebooks is set (default: 3600)",
    )
    return parser.parse_args()


def _validate_required(env_values: dict[str, str], *, cluster_override: str) -> tuple[str, str, str]:
    host = env_values.get("DATABRICKS_HOST", "").strip()
    token = env_values.get("DATABRICKS_TOKEN", "").strip()
    cluster_id = cluster_override.strip() or env_values.get("DATABRICKS_CLUSTER_ID", "").strip()

    if not host:
        raise ValueError("Missing required Databricks env var: DATABRICKS_HOST")
    if not token:
        raise ValueError("Missing required Databricks env var: DATABRICKS_TOKEN")
    if not cluster_id:
        raise ValueError("Missing required Databricks env var: DATABRICKS_CLUSTER_ID")
    return host, token, cluster_id


def _resolve_workspace_repo_root(env_values: dict[str, str], *, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip().rstrip("/")

    src_hint = env_values.get("INATINQ_SRC_DIR", "").strip()
    if not src_hint:
        raise ValueError(
            "Could not determine workspace repo root. Set INATINQ_SRC_DIR in env file "
            "or pass --workspace-repo-root."
        )

    src_path = Path(src_hint)
    return str(src_path.parent if src_path.name == "src" else src_path).rstrip("/")


def _configure_cli_profile(*, host: str, token: str, profile: str, via_command: bool) -> None:
    if via_command:
        if shutil.which("databricks") is None:
            raise RuntimeError("databricks CLI not found; install CLI or omit --configure-cli-via-command")
        cmd = ["databricks", "configure", "--host", host, "--token"]
        if profile != "DEFAULT":
            cmd.extend(["--profile", profile])
        result = subprocess.run(  # noqa: S603
            cmd,
            input=f"{token}\n",
            text=True,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "databricks configure failed"
            raise RuntimeError(details)
        _write(f"Configured CLI profile via command: {profile}")
        return

    cfg_path = Path.home() / ".databrickscfg"
    parser = configparser.RawConfigParser()
    if cfg_path.exists():
        parser.read(cfg_path, encoding="utf-8")
    if not parser.has_section(profile):
        parser.add_section(profile)
    parser.set(profile, "host", host)
    parser.set(profile, "token", token)
    with cfg_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    _write(f"Configured CLI profile in {cfg_path}: {profile}")


def _start_cluster(client: WorkspaceClient, *, cluster_id: str, wait_minutes: int) -> None:
    details = client.clusters.get(cluster_id=cluster_id)
    state = details.state

    _write(f"Cluster {cluster_id} current state: {state}")
    if state == State.RUNNING:
        _write("Cluster already RUNNING")
        return

    timeout = dt.timedelta(minutes=max(1, wait_minutes))
    _write(f"Starting cluster {cluster_id} (timeout={timeout})...")
    client.clusters.start_and_wait(cluster_id=cluster_id, timeout=timeout)

    details = client.clusters.get(cluster_id=cluster_id)
    if details.state != State.RUNNING:
        raise RuntimeError(f"Cluster did not reach RUNNING state (state={details.state})")
    _write("Cluster is RUNNING")


def _workspace_notebook_paths(workspace_repo_root: str) -> dict[str, str]:
    notebooks_root = f"{workspace_repo_root}/zarf/databricks/dev/notebooks"
    return {
        "common": f"{notebooks_root}/cdc_test_common.py",
        "producer": f"{notebooks_root}/cdc_producer_test.py",
        "consumer": f"{notebooks_root}/cdc_consumer_test.py",
    }


def _local_notebook_paths(repo_root: Path) -> dict[str, Path]:
    notebooks_root = repo_root / "zarf/databricks/dev/notebooks"
    return {
        "common": notebooks_root / "cdc_test_common.py",
        "producer": notebooks_root / "cdc_producer_test.py",
        "consumer": notebooks_root / "cdc_consumer_test.py",
    }


def _notebook_exists(client: WorkspaceClient, *, workspace_path: str) -> bool:
    try:
        client.workspace.get_status(path=workspace_path)
    except NotFound:
        return False
    return True


def _workspace_path_candidates(workspace_path: str) -> tuple[str, ...]:
    """Return candidate workspace notebook paths across extension variants."""
    candidates = [workspace_path]
    if workspace_path.endswith(".py"):
        base = workspace_path[: -len(".py")]
        candidates.extend([base, f"{workspace_path}.py"])
    else:
        candidates.append(f"{workspace_path}.py")
    return tuple(dict.fromkeys(candidates))


def _resolve_existing_workspace_path(client: WorkspaceClient, *, workspace_path: str) -> str | None:
    """Resolve the first existing workspace path among known extension variants."""
    for candidate in _workspace_path_candidates(workspace_path):
        if _notebook_exists(client, workspace_path=candidate):
            return candidate
    return None


def _upload_notebook(
    client: WorkspaceClient,
    *,
    local_path: Path,
    workspace_path: str,
) -> str:
    if not local_path.exists():
        raise FileNotFoundError(f"Local notebook missing: {local_path}")

    # Databricks SOURCE import can append language extension. Upload to a
    # normalized base path to avoid accidental *.py.py notebook names.
    upload_path = workspace_path.removesuffix(".py")

    parent = str(Path(upload_path).parent)
    client.workspace.mkdirs(path=parent)
    client.workspace.upload(
        path=upload_path,
        content=local_path.read_bytes(),
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=True,
    )

    resolved = _resolve_existing_workspace_path(client, workspace_path=workspace_path)
    if resolved is None:
        raise RuntimeError(
            "Notebook upload completed but no workspace object found at expected paths: "
            + ", ".join(_workspace_path_candidates(workspace_path))
        )
    return resolved


def _ensure_notebooks(
    client: WorkspaceClient,
    *,
    workspace_paths: dict[str, str],
    local_paths: dict[str, Path],
    upload_missing: bool,
) -> dict[str, str]:
    resolved_paths: dict[str, str] = {}
    for key, workspace_path in workspace_paths.items():
        if upload_missing:
            _write(f"Uploading notebook: {local_paths[key]} -> {workspace_path}")
            resolved_paths[key] = _upload_notebook(
                client,
                local_path=local_paths[key],
                workspace_path=workspace_path,
            )
            _write(f"Resolved workspace notebook path: {resolved_paths[key]}")
            continue

        resolved = _resolve_existing_workspace_path(client, workspace_path=workspace_path)
        if resolved is not None:
            resolved_paths[key] = resolved
            _write(f"Found workspace notebook: {resolved}")
            continue

        if not upload_missing:
            raise FileNotFoundError(
                f"Workspace notebook not found: {workspace_path}. "
                "Re-run with --upload-notebooks or sync your Databricks Repo."
            )
    return resolved_paths


def _collect_run_failure_details(client: WorkspaceClient, *, run_id: int) -> str:
    """Return concise task-level failure details for a submitted run."""
    try:
        run = client.jobs.get_run(run_id=run_id)
    except Exception as exc:
        return f"Unable to fetch run details for run_id={run_id}: {exc}"

    details: list[str] = []
    tasks = run.tasks or []
    for task in tasks:
        task_run_id = task.run_id
        task_key = task.task_key or "task"
        state = task.state.state_message if task.state else ""
        if not task_run_id:
            if state:
                details.append(f"{task_key}: {state}")
            continue
        try:
            out = client.jobs.get_run_output(run_id=task_run_id)
        except Exception as exc:
            details.append(f"{task_key}: unable to fetch run output ({exc})")
            continue

        if out.error:
            details.append(f"{task_key}: {out.error}")
        elif out.error_trace:
            details.append(f"{task_key}: {out.error_trace.splitlines()[0][:500]}")
        elif state:
            details.append(f"{task_key}: {state}")

    if not details:
        state = run.state.state_message if run.state else ""
        return state or f"No task output available for run_id={run_id}"
    return " | ".join(details)


def _wait_for_submit_run(
    submit_waiter: Any,
    *,
    timeout_seconds: int,
) -> Run:
    """Wait for a submitted run to finish with explicit timeout handling."""
    return submit_waiter.result(timeout=dt.timedelta(seconds=max(60, timeout_seconds)))


def _build_base_params(
    *,
    args: argparse.Namespace,
    is_consumer: bool,
) -> dict[str, str]:
    params: dict[str, str] = {}

    if args.workspace_env_file.strip():
        params["env_file"] = args.workspace_env_file.strip()

    if args.catalog.strip():
        params["catalog"] = args.catalog.strip()
    if args.schema.strip():
        params["schema"] = args.schema.strip()
    if args.test_run_suffix.strip():
        params["test_run_suffix"] = args.test_run_suffix.strip()

    if is_consumer:
        if args.progress_id.strip():
            params["progress_id"] = args.progress_id.strip()
        if args.collection.strip():
            params["collection"] = args.collection.strip()
        if args.window_size.strip():
            params["window_size"] = args.window_size.strip()

    return params


def _run_notebook(
    client: WorkspaceClient,
    *,
    cluster_id: str,
    notebook_path: str,
    task_key: str,
    run_name: str,
    base_parameters: dict[str, str],
    timeout_seconds: int,
) -> None:
    _write(f"Submitting notebook run: {notebook_path}")

    submit_waiter = client.jobs.submit(
        run_name=run_name,
        tasks=[
            SubmitTask(
                task_key=task_key,
                existing_cluster_id=cluster_id,
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                    base_parameters=base_parameters or None,
                    source=Source.WORKSPACE,
                ),
            )
        ],
    )
    run_id = int(submit_waiter.run_id)
    _write(f"Submitted run_id={run_id}")

    try:
        run = _wait_for_submit_run(submit_waiter, timeout_seconds=timeout_seconds)
    except Exception as exc:
        details = _collect_run_failure_details(client, run_id=run_id)
        raise RuntimeError(f"Notebook run failed for {notebook_path} (run_id={run_id}). {details}") from exc

    state = run.state
    if state is None:
        raise RuntimeError(f"Notebook run returned no state: {notebook_path}")

    result = state.result_state
    _write(
        "Notebook run finished: "
        f"life_cycle={state.life_cycle_state}, result={result}, "
        f"message={state.state_message or ''}"
    )

    if result not in {RunResultState.SUCCESS, RunResultState.SUCCESS_WITH_FAILURES}:
        details = _collect_run_failure_details(client, run_id=run_id)
        raise RuntimeError(
            f"Notebook run failed for {notebook_path} (run_id={run_id}) with result {result}. {details}"
        )


def _workspace_url(host: str, workspace_path: str) -> str:
    return f"{host.rstrip('/')}/#workspace{workspace_path}"


def main() -> int:
    """Run cluster bootstrap + CDC notebook setup workflow."""
    args = _parse_args()
    env_file = Path(args.env_file).expanduser().resolve()
    if not env_file.exists():
        _write(f"Missing env file: {env_file}", error=True)
        _write("Create it from zarf/databricks/dev/env.local.example", error=True)
        return 1

    env_values = _read_env_file(env_file)
    os.environ.update(env_values)

    try:
        host, token, cluster_id = _validate_required(env_values, cluster_override=args.cluster_id)
        workspace_repo_root = _resolve_workspace_repo_root(
            env_values,
            explicit=args.workspace_repo_root,
        )
    except ValueError as exc:
        _write(str(exc), error=True)
        return 1

    _write(f"Using env file: {env_file}")
    _write(f"Workspace repo root: {workspace_repo_root}")
    _write(f"Cluster id: {cluster_id}")

    if args.configure_cli:
        try:
            _configure_cli_profile(
                host=host,
                token=token,
                profile=args.profile,
                via_command=args.configure_cli_via_command,
            )
        except Exception as exc:
            _write(f"Failed to configure CLI profile: {exc}", error=True)
            return 1

    client = WorkspaceClient(host=host, token=token)

    try:
        _start_cluster(client, cluster_id=cluster_id, wait_minutes=args.cluster_wait_minutes)
    except Exception as exc:
        _write(f"Cluster start failed: {exc}", error=True)
        return 1

    workspace_paths = _workspace_notebook_paths(workspace_repo_root)
    local_paths = _local_notebook_paths(_repo_root())

    try:
        resolved_workspace_paths = _ensure_notebooks(
            client,
            workspace_paths=workspace_paths,
            local_paths=local_paths,
            upload_missing=args.upload_notebooks,
        )
    except Exception as exc:
        _write(f"Notebook validation/upload failed: {exc}", error=True)
        return 1

    producer_url = _workspace_url(host, resolved_workspace_paths["producer"])
    consumer_url = _workspace_url(host, resolved_workspace_paths["consumer"])

    _write("CDC notebook URLs:")
    _write(f"  Producer: {producer_url}")
    _write(f"  Consumer: {consumer_url}")

    if args.open_notebooks:
        webbrowser.open(producer_url)
        webbrowser.open(consumer_url)

    if args.run_notebooks:
        timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        try:
            if args.only in {"producer", "both"}:
                _run_notebook(
                    client,
                    cluster_id=cluster_id,
                    notebook_path=resolved_workspace_paths["producer"],
                    task_key="cdc_producer_test",
                    run_name=f"cdc-producer-test-{timestamp}",
                    base_parameters=_build_base_params(args=args, is_consumer=False),
                    timeout_seconds=args.run_timeout_seconds,
                )
            if args.only in {"consumer", "both"}:
                _run_notebook(
                    client,
                    cluster_id=cluster_id,
                    notebook_path=resolved_workspace_paths["consumer"],
                    task_key="cdc_consumer_test",
                    run_name=f"cdc-consumer-test-{timestamp}",
                    base_parameters=_build_base_params(args=args, is_consumer=True),
                    timeout_seconds=args.run_timeout_seconds,
                )
        except Exception as exc:
            _write(f"Notebook run failed: {exc}", error=True)
            return 1

    _write("Done. Attach notebooks to the started cluster in Databricks UI if you want interactive runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
