#!/usr/bin/env python3
"""Configure Databricks secret scope + cluster Spark S3A conf for MinIO access.

This helper is intended as a one-time bootstrap:
1. Create (or reuse) a Databricks secret scope.
2. Store MinIO access/secret keys in the scope.
3. Update cluster Spark conf to reference secret-backed S3A credentials.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


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


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _api_request(
    *,
    host: str,
    token: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{host.rstrip('/')}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
    }
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            body = resp.read().decode("utf-8").strip()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        detail: dict[str, Any]
        try:
            detail = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            detail = {}
        code = detail.get("error_code") or f"HTTP_{exc.code}"
        message = detail.get("message") or raw or exc.reason
        raise RuntimeError(f"{code}: {message}") from exc


def _create_scope_if_needed(*, host: str, token: str, scope: str) -> None:
    try:
        _api_request(
            host=host,
            token=token,
            method="POST",
            path="/api/2.0/secrets/scopes/create",
            payload={"scope": scope},
        )
        print(f"Created secret scope: {scope}")
    except RuntimeError as exc:
        if "RESOURCE_ALREADY_EXISTS" in str(exc):
            print(f"Secret scope already exists: {scope}")
            return
        raise


def _put_secret(*, host: str, token: str, scope: str, key: str, value: str) -> None:
    _api_request(
        host=host,
        token=token,
        method="POST",
        path="/api/2.0/secrets/put",
        payload={
            "scope": scope,
            "key": key,
            "string_value": value,
        },
    )


def _build_cluster_edit_payload(cluster: dict[str, Any], spark_conf: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cluster_id": cluster["cluster_id"],
        "spark_conf": spark_conf,
    }

    # Include the primary shape attributes to satisfy clusters/edit requirements.
    for key in (
        "cluster_name",
        "spark_version",
        "node_type_id",
        "driver_node_type_id",
        "num_workers",
        "autoscale",
        "autotermination_minutes",
        "aws_attributes",
        "azure_attributes",
        "gcp_attributes",
        "instance_pool_id",
        "driver_instance_pool_id",
        "spark_env_vars",
        "custom_tags",
        "cluster_log_conf",
        "init_scripts",
        "enable_elastic_disk",
        "policy_id",
        "data_security_mode",
        "single_user_name",
        "runtime_engine",
    ):
        value = cluster.get(key)
        if value is not None:
            payload[key] = value

    return payload


def _remove_stale_s3a_class_conf(spark_conf: dict[str, str]) -> None:
    """Drop class-name S3A conf keys that vary across Databricks runtimes."""
    spark_conf.pop("spark.hadoop.fs.s3a.impl", None)
    spark_conf.pop("spark.hadoop.fs.s3a.aws.credentials.provider", None)


def main() -> int:
    """Create/update Databricks secret scope and cluster Spark S3A settings."""
    repo_root = Path(__file__).resolve().parents[2]
    env_file = Path(os.environ.get("ENV_FILE", repo_root / "zarf/databricks/dev/.env.local"))

    if not env_file.exists():
        print(f"Missing env file: {env_file}", file=sys.stderr)
        print(
            "Create it from zarf/databricks/dev/env.local.example or set ENV_FILE.",
            file=sys.stderr,
        )
        return 1

    env_data = _read_env_file(env_file)
    os.environ.update(env_data)

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")
    s3_endpoint = os.getenv("S3_ENDPOINT")
    s3_access_key = os.getenv("S3_ACCESS_KEY_ID")
    s3_secret_key = os.getenv("S3_SECRET_ACCESS_KEY")

    required = {
        "DATABRICKS_HOST": host,
        "DATABRICKS_TOKEN": token,
        "DATABRICKS_CLUSTER_ID": cluster_id,
        "S3_ENDPOINT": s3_endpoint,
        "S3_ACCESS_KEY_ID": s3_access_key,
        "S3_SECRET_ACCESS_KEY": s3_secret_key,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        print(f"Missing required config: {', '.join(missing)}", file=sys.stderr)
        return 1

    scope = os.getenv("DATABRICKS_S3_SECRET_SCOPE", "inatinq-minio")
    access_key_name = os.getenv("DATABRICKS_S3_ACCESS_KEY_NAME", "s3-access-key")
    secret_key_name = os.getenv("DATABRICKS_S3_SECRET_KEY_NAME", "s3-secret-key")
    s3_use_ssl = _parse_bool(os.getenv("S3_USE_SSL"), default=False)
    s3_path_style = _parse_bool(os.getenv("S3_PATH_STYLE"), default=True)
    should_restart = _parse_bool(os.getenv("DATABRICKS_RESTART_CLUSTER_AFTER_S3A_CONFIG"), default=False)

    _create_scope_if_needed(host=host, token=token, scope=scope)
    _put_secret(host=host, token=token, scope=scope, key=access_key_name, value=s3_access_key)
    _put_secret(host=host, token=token, scope=scope, key=secret_key_name, value=s3_secret_key)
    print(f"Updated secrets in scope '{scope}' ({access_key_name}, {secret_key_name})")

    cluster = _api_request(
        host=host,
        token=token,
        method="GET",
        path=f"/api/2.0/clusters/get?cluster_id={urllib.parse.quote(cluster_id)}",
    )
    existing_spark_conf = dict(cluster.get("spark_conf") or {})
    _remove_stale_s3a_class_conf(existing_spark_conf)
    existing_spark_conf.update(
        {
            "spark.hadoop.fs.s3a.endpoint": s3_endpoint,
            "spark.hadoop.fs.s3a.path.style.access": str(s3_path_style).lower(),
            "spark.hadoop.fs.s3a.connection.ssl.enabled": str(s3_use_ssl).lower(),
            "spark.hadoop.fs.s3a.access.key": f"{{{{secrets/{scope}/{access_key_name}}}}}",
            "spark.hadoop.fs.s3a.secret.key": f"{{{{secrets/{scope}/{secret_key_name}}}}}",
        }
    )

    edit_payload = _build_cluster_edit_payload(cluster, existing_spark_conf)
    _api_request(
        host=host,
        token=token,
        method="POST",
        path="/api/2.0/clusters/edit",
        payload=edit_payload,
    )
    print(f"Updated cluster Spark conf for S3A on cluster {cluster_id}")

    if should_restart:
        _api_request(
            host=host,
            token=token,
            method="POST",
            path="/api/2.0/clusters/restart",
            payload={"cluster_id": cluster_id},
        )
        print(f"Cluster restart requested: {cluster_id}")
    else:
        print(
            "Cluster restart not requested. Set DATABRICKS_RESTART_CLUSTER_AFTER_S3A_CONFIG=true to auto-restart."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
