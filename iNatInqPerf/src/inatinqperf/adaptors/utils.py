"""Utils for vector database adaptors."""

import subprocess

from loguru import logger


def _is_safe_container_name(name: str) -> bool:
    """Check if the name of the container is safe to run in a cluster."""
    _allowed_container_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return name and all(char in _allowed_container_chars for char in name)


def log_single_container_tail(docker_cmd: str, container_name: str) -> None:
    """Log the output tail of running a single container `container_name` via the `docker_cmd`."""
    if not _is_safe_container_name(container_name):
        logger.warning(f"Skipping unsafe container name: {container_name!r}")
        return

    try:
        result = subprocess.run(  # noqa: S603
            [docker_cmd, "logs", "--tail", "20", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"Failed to fetch logs for container '{container_name}': {exc}")
        return

    output = result.stdout.strip()
    if output:
        logger.warning(f"[{container_name}] {output}")
    error_output = result.stderr.strip()
    if error_output:
        logger.warning(f"[{container_name}][stderr] {error_output}")
