"""Checkpoint management for tracking processed items.

This module provides utilities for saving and loading checkpoints of processed
items (e.g., S3 keys, file paths, job IDs), enabling job recovery and avoiding
reprocessing of already-processed items. Supports both local filesystem and S3
storage for checkpoints.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import attrs
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger("pipeline.checkpoint")

__all__ = ["CheckpointManager", "is_s3_path"]


def is_s3_path(path: str | Path) -> bool:
    """Check if path is an S3 URI.

    Args:
        path: Path to check.

    Returns:
        True if path is an S3 URI (s3:// or s3a://), False otherwise.
    """
    path_str = str(path)
    return path_str.startswith("s3://") or path_str.startswith("s3a://")


def _parse_s3_path(s3_uri: str) -> tuple[str, str]:
    """Parse S3 URI into bucket and key.

    Args:
        s3_uri: S3 URI (e.g., s3://bucket/path/to/file.json).

    Returns:
        Tuple of (bucket, key).
    """
    # Remove s3:// or s3a:// prefix
    path = s3_uri.replace("s3://", "").replace("s3a://", "")
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return (bucket, key)


@attrs.define(frozen=True, slots=True)
class CheckpointManager:
    """Manager for loading and saving checkpoints of processed items.

    Supports both local filesystem and S3 storage for checkpoint persistence.
    Can be used with or without an S3 client, allowing flexibility in different
    execution contexts.

    Attributes:
        s3_client: Optional S3ClientWrapper instance for S3 checkpoint operations.
            If None, only local filesystem checkpoints are supported.

    Example:
        ```python
        from pathlib import Path
        from core.ingestion.checkpoint import CheckpointManager
        from clients import create_s3_client

        # Local filesystem only
        manager = CheckpointManager()
        processed = manager.load(Path("/tmp/checkpoint.json"))
        manager.save(Path("/tmp/checkpoint.json"), processed)

        # With S3 support
        s3_client = create_s3_client()
        manager = CheckpointManager(s3_client=s3_client)
        processed = manager.load("s3://bucket/checkpoints/job.json")
        manager.save("s3://bucket/checkpoints/job.json", processed)
        ```
    """

    s3_client: Any | None = attrs.field(default=None)

    def load(self, checkpoint_path: str | Path) -> set[str]:
        """Load checkpoint of processed items.

        Supports both local filesystem and S3 storage. If checkpoint_path is an S3 URI
        (s3:// or s3a://), s3_client must be provided either via constructor or
        as an instance attribute.

        Args:
            checkpoint_path: Path to checkpoint file (local path or S3 URI).

        Returns:
            Set of item identifiers that have been successfully processed
            (e.g., S3 keys, file paths, job IDs).

        Example:
            ```python
            from pathlib import Path
            from core.ingestion.checkpoint import CheckpointManager

            manager = CheckpointManager()
            processed = manager.load(Path("/tmp/checkpoint.json"))
            # Returns: {"item1", "item2", ...}
            ```
        """
        checkpoint_str = str(checkpoint_path)

        # Handle S3 paths
        if is_s3_path(checkpoint_path):
            if self.s3_client is None:
                logger.warning(
                    "S3 checkpoint path provided but no S3 client, starting fresh",
                    extra={"path": checkpoint_str},
                )
                return set()

            try:
                bucket, key = _parse_s3_path(checkpoint_str)
                content_bytes = self.s3_client.get_object(bucket=bucket, key=key)
                data = json.loads(content_bytes.decode("utf-8"))
                processed_keys = set(data.get("processed_keys", []))
                logger.info(
                    "Loaded checkpoint from S3",
                    extra={"path": checkpoint_str, "count": len(processed_keys)},
                )
                return processed_keys
            except (json.JSONDecodeError, KeyError, ClientError, BotoCoreError, OSError) as e:
                logger.warning(
                    "Failed to load checkpoint from S3, starting fresh",
                    extra={"path": checkpoint_str, "error": str(e)},
                )
                return set()

        # Handle local filesystem paths
        path = Path(checkpoint_path)
        if not path.exists():
            logger.debug("No checkpoint file found", extra={"path": checkpoint_str})
            return set()

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                processed_keys = set(data.get("processed_keys", []))
                logger.info(
                    "Loaded checkpoint",
                    extra={"path": checkpoint_str, "count": len(processed_keys)},
                )
                return processed_keys
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(
                "Failed to load checkpoint, starting fresh",
                extra={"path": checkpoint_str, "error": str(e)},
            )
            return set()

    def save(self, checkpoint_path: str | Path, processed_keys: set[str]) -> None:
        """Save checkpoint of processed items.

        Supports both local filesystem and S3 storage. If checkpoint_path is an S3 URI
        (s3:// or s3a://), s3_client must be provided either via constructor or
        as an instance attribute.

        Args:
            checkpoint_path: Path to checkpoint file (local path or S3 URI).
            processed_keys: Set of item identifiers that have been successfully processed
                (e.g., S3 keys, file paths, job IDs).

        Example:
            ```python
            from pathlib import Path
            from core.ingestion.checkpoint import CheckpointManager

            manager = CheckpointManager()
            processed = {"inputs/file1.txt", "inputs/file2.txt"}
            manager.save(Path("/tmp/checkpoint.json"), processed)
            ```

        Note:
            The checkpoint file is saved as JSON with the following structure:
            ```json
            {
              "processed_keys": ["key1", "key2", ...],
              "last_updated": 1234567890.0
            }
            ```
        """
        checkpoint_str = str(checkpoint_path)
        data = {"processed_keys": sorted(processed_keys), "last_updated": time.time()}
        json_content = json.dumps(data, indent=2).encode("utf-8")

        # Handle S3 paths
        if is_s3_path(checkpoint_path):
            if self.s3_client is None:
                logger.warning(
                    "S3 checkpoint path provided but no S3 client, skipping save",
                    extra={"path": checkpoint_str},
                )
                return

            try:
                bucket, key = _parse_s3_path(checkpoint_str)
                self.s3_client.put_object(bucket=bucket, key=key, body=json_content)
                logger.debug(
                    "Saved checkpoint to S3",
                    extra={"path": checkpoint_str, "count": len(processed_keys)},
                )
            except (ClientError, BotoCoreError, OSError, ValueError) as e:
                logger.warning(
                    "Failed to save checkpoint to S3",
                    extra={"path": checkpoint_str, "error": str(e)},
                )
            return

        # Handle local filesystem paths
        try:
            path = Path(checkpoint_path)
            # Ensure checkpoint directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save checkpoint as JSON
            with path.open("wb") as f:
                f.write(json_content)

            logger.debug(
                "Saved checkpoint",
                extra={"path": checkpoint_str, "count": len(processed_keys)},
            )
        except OSError as e:
            logger.warning(
                "Failed to save checkpoint",
                extra={"path": checkpoint_str, "error": str(e)},
            )

